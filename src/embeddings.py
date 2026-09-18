"""
BanglaBERT Frozen Feature Extraction and Caching Module.
Extracts mean-pooled contextual sentence embeddings using sagorsarker/bangla-bert-base.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import BERT_BATCH_SIZE, BERT_MAX_LENGTH, BERT_MODEL_NAME, MODELS_CACHE

def _get_device() -> torch.device:
    """Determine the optimal compute device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        # Utilize CPU cores for multi-threaded inference
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)
        return torch.device("cpu")


_DEVICE = _get_device()
_TOKENIZER: Optional[Any] = None
_MODEL: Optional[Any] = None


def load_banglabert() -> Tuple[Any, Any]:
    """
    Lazily load and cache BanglaBERT tokenizer and model in eval mode.
    Returns: (tokenizer, model)
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        if model is not None:
            model.to(_DEVICE)
            model.eval()
        _TOKENIZER = tokenizer
        _MODEL = model
    return _TOKENIZER, _MODEL


def get_bert_features(
    texts: Union[List[str], pd.Series, str],
    batch_size: int = BERT_BATCH_SIZE,
    max_length: int = BERT_MAX_LENGTH
) -> np.ndarray:
    """
    Extract frozen 768-dimensional mean-pooled BanglaBERT embeddings for a list of texts.
    Uses optimized PyTorch inference mode.
    Returns: np.ndarray of shape (len(texts), 768)
    """
    if isinstance(texts, str):
        text_list = [texts]
    elif isinstance(texts, pd.Series):
        text_list = [str(t) for t in texts.fillna("").tolist()]
    else:
        text_list = [str(t) if not isinstance(t, str) else t for t in texts]

    if len(text_list) == 0:
        return np.zeros((0, 768), dtype=np.float32)

    tokenizer, model = load_banglabert()
    all_embeddings = []

    with torch.inference_mode():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            batch_cleaned = [t if t.strip() else "ভালো" for t in batch_texts]

            inputs = tokenizer(
                batch_cleaned,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # shape: (batch, seq_len, 768)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            
            # Mean pooling over non-padded tokens
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()

            all_embeddings.append(mean_pooled)

    return np.vstack(all_embeddings).astype(np.float32)



def get_or_cache_bert_features(
    texts: Union[List[str], pd.Series],
    cache_path: Union[str, Path],
    batch_size: int = BERT_BATCH_SIZE,
    max_length: int = BERT_MAX_LENGTH
) -> np.ndarray:
    """
    Load precomputed BERT features from cache if available and matching sample count,
    otherwise compute using get_bert_features and save to cache.
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    expected_len = len(texts) if hasattr(texts, "__len__") else len(list(texts))

    if cache_file.exists():
        try:
            cached = np.load(cache_file)
            if len(cached) == expected_len:
                return cached
        except Exception:
            pass

    embeddings = get_bert_features(texts, batch_size=batch_size, max_length=max_length)
    np.save(cache_file, embeddings)
    return embeddings

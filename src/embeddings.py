import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    AutoModel = None
    AutoTokenizer = None
    HAS_TORCH = False
    _DEVICE = "cpu"

from src.config import BERT_BATCH_SIZE, BERT_MAX_LENGTH, BERT_MODEL_NAME

_TOKENIZER: Optional[Any] = None
_MODEL: Optional[Any] = None


def load_banglabert() -> Tuple[Any, Any]:
    """Lazily load and cache BanglaBERT model and tokenizer."""
    global _TOKENIZER, _MODEL
    if not HAS_TORCH or AutoTokenizer is None or AutoModel is None:
        raise ImportError("PyTorch & Transformers required: pip install torch transformers")

    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME).to(_DEVICE).eval()
    return _TOKENIZER, _MODEL


def get_bert_features(
    texts: Union[List[str], pd.Series, str],
    batch_size: int = BERT_BATCH_SIZE,
    max_length: int = BERT_MAX_LENGTH
) -> np.ndarray:
    """Extract frozen mean-pooled 768-dim BanglaBERT embeddings."""
    if isinstance(texts, str):
        text_list = [texts]
    elif isinstance(texts, pd.Series):
        text_list = [str(t) for t in texts.fillna("").tolist()]
    else:
        text_list = [str(t) if not isinstance(t, str) else t for t in texts]

    if not text_list:
        return np.zeros((0, 768), dtype=np.float32)

    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch & Transformers required: pip install torch transformers")

    tokenizer, model = load_banglabert()
    all_embeddings = []

    with torch.inference_mode():
        for i in range(0, len(text_list), batch_size):
            batch_texts = [t if t.strip() else " " for t in text_list[i : i + batch_size]]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            all_embeddings.append((sum_emb / sum_mask).cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def get_or_cache_bert_features(
    texts: Union[List[str], pd.Series],
    cache_path: Union[str, Path],
    batch_size: int = BERT_BATCH_SIZE,
    max_length: int = BERT_MAX_LENGTH
) -> np.ndarray:
    """Load precomputed BERT embeddings from cache or compute and save."""
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

    if not HAS_TORCH:
        if cache_file.exists():
            return np.load(cache_file)
        raise ImportError("PyTorch & Transformers required to extract BERT embeddings: pip install torch transformers")

    embeddings = get_bert_features(texts, batch_size=batch_size, max_length=max_length)
    np.save(cache_file, embeddings)
    return embeddings

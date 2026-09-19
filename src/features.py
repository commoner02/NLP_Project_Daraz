"""Feature engineering: TF-IDF vectorizers, vocabulary, PyTorch models, dataloaders."""
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from torch.utils.data import DataLoader, Dataset

from src.config import (
    LSTM_EMBED_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    TFIDF_CHAR_NGRAMS,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_WORD_NGRAMS,
)


# ── 1. TF-IDF ────────────────────────────────────────────────────────────────
def build_tfidf() -> FeatureUnion:
    """Word (1,2) + char (3,5) n-grams, sublinear TF."""
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[\u0980-\u09FFA-Za-z0-9]+",
            ngram_range=TFIDF_WORD_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_CHAR_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
        )),
    ])


# ── 2. Vocabulary ────────────────────────────────────────────────────────────
class BanglaVocab:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_size: int = 15000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {self.PAD: 0, self.UNK: 1}

    def fit(self, texts: List[str]) -> None:
        freqs = Counter(w for t in texts for w in str(t).split())
        for word, count in freqs.most_common(self.max_size - 2):
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def transform(self, text: str, max_length: int) -> List[int]:
        ids = [self.word2idx.get(w, 1) for w in str(text).split()][:max_length]
        return ids + [0] * (max_length - len(ids))

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


# ── 3. PyTorch models ────────────────────────────────────────────────────────
class _LSTMBase(nn.Module):
    """Shared bi-LSTM trunk. Subclasses set `head` output size."""

    def __init__(self, vocab_size: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, LSTM_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            LSTM_EMBED_DIM,
            LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=0.5 if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(LSTM_HIDDEN_DIM * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        _, (h, _) = self.lstm(emb)
        return self.fc(self.dropout(torch.cat((h[-2], h[-1]), dim=1)))


class SentimentLSTM(_LSTMBase):
    CLASSES = ["Negative", "Neutral", "Positive"]

    def __init__(self, vocab_size: int):
        super().__init__(vocab_size, output_dim=len(self.CLASSES))
        self.classes_ = list(self.CLASSES)


class AspectLSTM(_LSTMBase):
    def __init__(self, vocab_size: int, num_aspects: int):
        super().__init__(vocab_size, output_dim=num_aspects)


# ── 4. DataLoader ────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts: Any, labels: Any, vocab: BanglaVocab, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocab.transform(self.texts[index], self.max_length)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[index])


def create_dataloader(
    texts: Any,
    labels: Any,
    vocab: BanglaVocab,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Helper to create a DataLoader from raw texts and labels."""
    return DataLoader(
        TextDataset(texts, labels, vocab, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
    )

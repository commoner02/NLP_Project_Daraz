# Bangla Daraz Review Analytics & Aspect-Based Sentiment Analysis (ABSA)
## Complete Codebase & System Architecture Reference Document

---

## 1. System Overview & Architecture

This document serves as an all-in-one AI reading context containing the complete project architecture, directory structure, and full source code for the Bangla Daraz Aspect-Based Sentiment Analysis (ABSA) system.

### NLP Formulation:
1. **3-Class Sentiment Analysis**: Classifies reviews into Positive, Neutral, or Negative.
2. **Multi-Label Aspect Detection**: Detects mentions of 5 core aspects: Product Quality, Price, Delivery, Packaging, Seller Service.
3. **Aspect-Level Binary Polarity**: Resolves Positive vs. Negative polarity for each detected aspect.
4. **Dual Model Benchmarking**: Compares TF-IDF FeatureUnion (Word + Char n-grams) with a custom PyTorch BiLSTM network.
5. **Preprocessing**: Normalizes Unicode NFC, strips noise, preserves negations (e.g., না, নাই, নয়, নেই), and standardizes composite labels.


---

## 2. Complete Project Directory Structure

```
nlp_daraz/
├── .streamlit/
│   └── config.toml                  # Streamlit theme configuration
├── data/
│   ├── original_data/               # Raw source data from Mendeley
│   └── processed_data/
│       ├── annotated_bangla.csv     # Raw input corpus (review_id, original_text, label)
│       ├── clean_annotated_bangla.csv # Clean dataset with parsed targets
│       └── dataset_summary.json     # Class and aspect distribution stats
├── models/
│   ├── lstm/                        # PyTorch BiLSTM binaries (sentiment_model.pt, issue_model.pt, vocab.pkl)
│   └── tfidf/                       # TF-IDF binaries (sentiment_model.pkl, issue_model.pkl, polarity_*.pkl)
├── results/
│   ├── metrics_summary.json         # Evaluation metrics JSON
│   ├── model_comparison.csv         # Comparative benchmark metrics
│   ├── sentiment_lstm.png           # BiLSTM confusion matrix heatmap
│   └── sentiment_tfidf.png          # TF-IDF confusion matrix heatmap
├── src/
│   ├── __init__.py
│   ├── config.py                    # Constants, paths, and hyperparameters
│   ├── preprocessing.py             # Bangla text normalization
│   ├── data_processing.py           # Parsing labels and dataset splits
│   ├── features.py                  # Vocab, PyTorch Datasets & LSTM models
│   ├── train.py                     # TF-IDF & BiLSTM training routines
│   ├── predict.py                   # Clean inference routines
│   ├── evaluate.py                  # Metric computation and plot savers
│   └── persistence.py               # Model save/load helpers
├── app.py                           # Interactive Streamlit dashboard
├── pipeline.ipynb                   # Single end-to-end training notebook
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore configuration
├── README.md                        # Documentation
└── PROJECT_CODEBASE.md              # Complete codebase reference
```


---

## 3. Complete Source Code by File

### `requirements.txt`
```text
--extra-index-url https://download.pytorch.org/whl/cpu
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.30.0
plotly>=5.15.0
torch>=2.0.0
ipykernel>=6.25.0

```

### `.gitignore`
```text
# Byte-compiled / cache files
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.ipynb_checkpoints/

# Virtual Environments
.venv/
venv/
.nlp_venv/
env/

# Model binaries & caches
models/**/*.pkl
models/cache/*.npy
*.bin
*.pt

# Results (regenerable artifacts)
results/*.png
results/*.csv
results/metrics_summary.json

# Keep essential directory tracking
!models/**/.gitkeep
!results/**/.gitkeep
!data/**/.gitkeep

# IDE / System files
.DS_Store
.vscode/
.idea/
*.swp

```

### `.streamlit/config.toml`
```toml
[server]
fileWatcherType = "poll"
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#3B82F6"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#0F172A"
font = "sans serif"

```

### `src/__init__.py`
```python
"""
Bangla Daraz Review Analytics NLP Package.
"""
__version__ = "1.0.0"

```

### `src/config.py`
```python
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_TFIDF = MODELS_DIR / "tfidf"
MODELS_LSTM = MODELS_DIR / "lstm"
RESULTS_DIR = BASE_DIR / "results"

# Tasks & Aspects
ALL_ASPECTS = [
    "Product Quality",
    "Price",
    "Delivery",
    "Packaging",
    "Seller Service",
]

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

ASPECT_MAPPING = {
    "product_quality": "Product Quality",
    "price": "Price",
    "delivery": "Delivery",
    "packaging": "Packaging",
    "seller_service": "Seller Service",
}

# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# BiLSTM settings
LSTM_VOCAB_SIZE = 15000
LSTM_EMBED_DIM = 300
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_BATCH_SIZE = 32
LSTM_MAX_LENGTH = 128
LSTM_EPOCHS = 10
LSTM_LR = 1e-3

# TF-IDF settings
TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95


def ensure_dirs() -> None:
    """Create required project directories if missing."""
    for directory in [DATA_DIR, MODELS_TFIDF, MODELS_LSTM, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)

```

### `src/preprocessing.py`
```python
import re
import unicodedata

# Pre-compiled regex patterns
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_ELONGATION = re.compile(r"(.)\1{2,}")
# Preserve Bangla, English letters (code-switching / loanwords), digits, and standard punctuation/spaces
RE_NOISE = re.compile(r"[^\u0980-\u09FFA-Za-z0-9\s.,!?|।॥\-_\']")
ZERO_WIDTH_CHARS = ["\u200c", "\u200d", "\ufeff", "\u200b", "\u200e", "\u200f"]


def normalize_unicode(text: str) -> str:
    """Normalize text to NFC form and strip zero-width characters."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, "")
    return text


def clean_text(text: str) -> str:
    """Clean Bangla review text preserving negations, loanwords, and sentiment context."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_ELONGATION.sub(r"\1", text)
    text = RE_NOISE.sub(" ", text)
    return " ".join(text.split())

```

### `src/data_processing.py`
```python
"""Data loading and train/test splitting. Single source of truth: `label` column."""
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ALL_ASPECTS, ASPECT_MAPPING, DATA_DIR, RANDOM_STATE, TEST_SIZE
from src.preprocessing import clean_text

import json

RAW_FILE = DATA_DIR / "processed_data" / "annotated_bangla.csv"
CLEAN_FILE = DATA_DIR / "processed_data" / "clean_annotated_bangla.csv"
MIN_POLARITY_SAMPLES = 50  # Skip aspects with fewer samples than this threshold


def _parse_label(label: str) -> Tuple[List[str], Dict[str, str], str]:
    """Parse `label` → (aspects, aspect_polarities, overall_sentiment).

    Handles both '#' and ';' delimiters. Example:
        'packaging_negative#product_quality_negative'
        → (['Packaging', 'Product Quality'],
           {'Packaging': 'negative', 'Product Quality': 'negative'},
           'Negative')
    """
    if not isinstance(label, str) or not label.strip():
        return [], {}, "Neutral"

    aspects: List[str] = []
    polarities: Dict[str, str] = {}
    for token in re.split(r"[#;]", label):
        token = token.strip()
        if "_" not in token:
            continue
        key, polarity = token.rsplit("_", 1)
        aspect = ASPECT_MAPPING.get(key.lower(), key.replace("_", " ").title())
        aspects.append(aspect)
        polarities[aspect] = polarity.lower()

    pos = sum(1 for p in polarities.values() if p == "positive")
    neg = sum(1 for p in polarities.values() if p == "negative")
    overall = "Positive" if pos > neg else "Negative" if neg > pos else "Neutral"
    return sorted(list(set(aspects))), polarities, overall


def load_data(save_clean: bool = True) -> pd.DataFrame:
    """Load raw CSV, clean text, parse labels into targets, and optionally save clean CSV.

    Input (annotated_bangla.csv): review_id | original_text | label
    Output df: review_id | original_text | cleaned_text | label | sentiment | aspects | aspect_polarities
    """
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_FILE}")

    df = pd.read_csv(RAW_FILE, index_col=False)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if "review_id" in df.columns:
        df = df.drop_duplicates(subset=["review_id"])

    # Clean raw Bangla text
    source = df["original_text"] if "original_text" in df.columns else df["cleaned_text"]
    df["cleaned_text"] = source.fillna("").apply(clean_text)

    # Parse `label` into structured targets
    parsed = df["label"].apply(_parse_label)
    df["aspects"] = parsed.apply(lambda x: x[0])
    df["aspect_polarities"] = parsed.apply(lambda x: x[1])
    df["sentiment"] = parsed.apply(lambda x: x[2])

    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)

    if save_clean:
        clean_df = df.copy()
        clean_df["aspects"] = clean_df["aspects"].apply(lambda x: ", ".join(x))
        clean_df["aspect_polarities"] = clean_df["aspect_polarities"].apply(json.dumps)
        cols = [c for c in ["review_id", "original_text", "cleaned_text", "label", "sentiment", "aspects", "aspect_polarities"] if c in clean_df.columns]
        clean_df = clean_df[cols]
        clean_df.to_csv(CLEAN_FILE, index=False)

    return df


def get_sentiment_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split for 3-class sentiment."""
    return train_test_split(
        df["cleaned_text"],
        df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )


def get_aspect_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[List[str]], List[List[str]]]:
    """80/20 train/test split for multi-label aspect detection."""
    tr, te = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return (
        df.loc[tr, "cleaned_text"],
        df.loc[te, "cleaned_text"],
        df.loc[tr, "aspects"].tolist(),
        df.loc[te, "aspects"].tolist(),
    )


def get_polarity_split(df: pd.DataFrame, aspect: str) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """Binary split for one aspect. Returns None if too few samples."""
    rows = [
        {"text": row["cleaned_text"], "polarity": row["aspect_polarities"][aspect].title()}
        for _, row in df.iterrows()
        if aspect in row["aspect_polarities"]
        and row["aspect_polarities"][aspect] in ("positive", "negative")
    ]
    if len(rows) < MIN_POLARITY_SAMPLES:
        return None

    sub = pd.DataFrame(rows)
    counts = sub["polarity"].value_counts()
    stratify = sub["polarity"] if counts.min() >= 2 else None
    return train_test_split(
        sub["text"],
        sub["polarity"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

```

### `src/features.py`
```python
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

```

### `src/train.py`
```python
"""Model training routines for TF-IDF and PyTorch BiLSTM classifiers."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    ALL_ASPECTS,
    LSTM_EPOCHS,
    LSTM_LR,
    RANDOM_STATE,
)
from src.data_processing import get_polarity_split
from src.evaluate import _evaluate
from src.features import AspectLSTM, SentimentLSTM, build_tfidf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- TF-IDF trainers ----------------------------------------------------------
def _train_logreg_classifier(
    X_tr: Any,
    y_tr: Any,
    X_te: Any,
    y_te: Any,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Helper to train a balanced Logistic Regression classifier on TF-IDF features."""
    vec = build_tfidf()
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=max_iter, random_state=RANDOM_STATE, solver="lbfgs"
    )
    model.fit(vec.fit_transform(X_tr), y_tr)
    return model, vec, _evaluate(y_te, model.predict(vec.transform(X_te)))


def train_sentiment(
    X_tr: Any,
    y_tr: Any,
    X_te: Any,
    y_te: Any,
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Train 3-class sentiment classifier using TF-IDF."""
    return _train_logreg_classifier(X_tr, y_tr, X_te, y_te)


def train_aspects(
    X_tr: Any,
    y_tr: List[List[str]],
    X_te: Any,
    y_te: List[List[str]],
) -> Tuple[OneVsRestClassifier, FeatureUnion, MultiLabelBinarizer, Dict[str, Any]]:
    """Train multi-label aspect classifier using TF-IDF."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_tr_bin = mlb.fit_transform(y_tr)
    y_te_bin = mlb.transform(y_te)

    vec = build_tfidf()
    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        )
    )
    model.fit(vec.fit_transform(X_tr), y_tr_bin)
    return model, vec, mlb, _evaluate(y_te_bin, model.predict(vec.transform(X_te)), multilabel=True)


def train_polarities(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Train one binary classifier per aspect. Skips aspects with < MIN samples."""
    out: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        split = get_polarity_split(df, aspect)
        if split is None:
            print(f"  [Skipped] {aspect}: too few samples (< MIN_POLARITY_SAMPLES)")
            continue
        X_tr, X_te, y_tr, y_te = split
        model, vec, m = _train_logreg_classifier(X_tr, y_tr, X_te, y_te)
        out[aspect] = {
            "model": model,
            "vectorizer": vec,
            "metrics": m,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }
    return out


# -- LSTM trainers ------------------------------------------------------------
def _train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    name: str = "Model",
) -> nn.Module:
    """Shared PyTorch BiLSTM training loop."""
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LSTM_LR)
    model.train()
    print(f"    Training BiLSTM {name}...")
    for epoch in range(1, LSTM_EPOCHS + 1):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y.float() if isinstance(criterion, nn.BCEWithLogitsLoss) else y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"      Epoch [{epoch:02d}/{LSTM_EPOCHS:02d}] - Loss: {avg_loss:.4f}", flush=True)
    return model


def train_lstm_sentiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    y_te: List[str],
    class_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[SentimentLSTM, Dict[str, Any]]:
    """Train 3-class sentiment classifier using BiLSTM."""
    model = SentimentLSTM(vocab_size)
    _train_lstm(model, train_loader, nn.CrossEntropyLoss(), name="Sentiment Model")

    model.eval()
    idx2class = {i: c for i, c in enumerate(SentimentLSTM.CLASSES)}
    preds: List[str] = []
    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x.to(DEVICE))
            preds.extend([idx2class[i] for i in logits.argmax(1).cpu().numpy().tolist()])
    return model, _evaluate(y_te, preds)


def train_lstm_aspects(
    train_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    y_te_bin: np.ndarray,
    mlb: MultiLabelBinarizer,
) -> Tuple[AspectLSTM, MultiLabelBinarizer, Dict[str, Any]]:
    """Train multi-label aspect classifier using BiLSTM."""
    model = AspectLSTM(vocab_size, len(ALL_ASPECTS))
    _train_lstm(model, train_loader, nn.BCEWithLogitsLoss(), name="Multi-Label Aspect Model")

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            probs = torch.sigmoid(model(x.to(DEVICE)))
            preds.extend((probs > 0.5).int().cpu().numpy())
    return model, mlb, _evaluate(y_te_bin, np.array(preds), multilabel=True)

```

### `src/predict.py`
```python
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

from src.config import ALL_ASPECTS, LSTM_MAX_LENGTH
from src.features import AspectLSTM, SentimentLSTM
from src.preprocessing import clean_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_sentiment(
    text: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    vocab: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """3-class sentiment prediction. Pass vectorizer for TF-IDF or vocab for LSTM."""
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in SentimentLSTM.CLASSES},
        }

    if vocab is not None:  # LSTM path
        ids = torch.tensor([vocab.transform(cleaned, LSTM_MAX_LENGTH)], dtype=torch.long).to(DEVICE)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(ids), dim=1)[0].cpu().numpy()
        classes = SentimentLSTM.CLASSES
    else:  # TF-IDF path
        feat = vectorizer.transform([cleaned]) if vectorizer is not None else cleaned
        probs = model.predict_proba(feat)[0]
        classes = model.classes_

    pred = str(classes[int(np.argmax(probs))])
    return {
        "sentiment": pred,
        "confidence": float(max(probs)),
        "probabilities": {str(c): float(p) for c, p in zip(classes, probs)},
    }


def predict_aspects(
    text: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    vocab: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Multi-label aspect detection."""
    cleaned = clean_text(text)
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    if not cleaned:
        return {"aspects": [], "confidences": {a: 0.0 for a in ALL_ASPECTS}}

    if vocab is not None:  # LSTM path
        ids = torch.tensor([vocab.transform(cleaned, LSTM_MAX_LENGTH)], dtype=torch.long).to(DEVICE)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(ids))[0].cpu().numpy()
    else:  # TF-IDF path
        feat = vectorizer.transform([cleaned]) if vectorizer is not None else cleaned
        probs = model.predict_proba(feat)[0]

    detected = [binarizer.classes_[i] for i, p in enumerate(probs) if p > 0.5]
    return {
        "aspects": detected,
        "confidences": {str(a): float(p) for a, p in zip(binarizer.classes_, probs)},
    }


def predict_aspect_polarity(
    text: str,
    aspect: str,
    polarity_models: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.60,
) -> Dict[str, Any]:
    """Binary polarity prediction for a specific aspect."""
    default = {
        "aspect": aspect,
        "polarity": "Positive",
        "confidence": 0.50,
        "is_low_confidence": True,
        "probabilities": {"Positive": 0.5, "Negative": 0.5},
    }

    cleaned = clean_text(text)
    if not cleaned or aspect not in polarity_models:
        return default

    entry = polarity_models[aspect]
    vec = entry.get("vectorizer")
    model = entry["model"]
    if vec is None:
        return default

    feat = vec.transform([cleaned])
    pred = str(model.predict(feat)[0])
    probs = model.predict_proba(feat)[0]
    conf = float(max(probs))

    return {
        "aspect": aspect,
        "polarity": pred,
        "confidence": conf,
        "is_low_confidence": conf < min_confidence,
        "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probs)},
    }


def predict_hierarchical(
    text: str,
    aspect_model: Any,
    polarity_models: Dict[str, Dict[str, Any]],
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    vocab: Optional[Any] = None,
    min_confidence: float = 0.60,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hierarchical ABSA: Detect aspects → binary polarity per aspect."""
    aspect_res = predict_aspects(
        text, aspect_model, vectorizer=vectorizer, binarizer=binarizer, vocab=vocab
    )
    aspects = aspect_res["aspects"]
    details: List[Dict[str, Any]] = []

    for asp in aspects:
        info = predict_aspect_polarity(text, asp, polarity_models, min_confidence=min_confidence)
        pol, conf, low = info["polarity"], info["confidence"], info["is_low_confidence"]
        if low:
            status, color, bg = "low_confidence", "#D97706", "#FFFBEB"
        elif pol == "Positive":
            status, color, bg = "positive", "#10B981", "#ECFDF5"
        else:
            status, color, bg = "negative", "#EF4444", "#FEF2F2"

        details.append({
            "aspect": asp,
            "polarity": pol,
            "confidence": conf,
            "is_low_confidence": low,
            "status": status,
            "color": color,
            "bg_color": bg,
        })

    return {
        "aspects": aspects,
        "aspect_details": details,
        "confidences": aspect_res["confidences"],
    }

```

### `src/evaluate.py`
```python
"""Metrics, confusion matrices, and benchmark CSV/JSON export."""
import json
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
)

from src.config import ALL_ASPECTS, RESULTS_DIR


def _evaluate(y_true: Any, y_pred: Any, multilabel: bool = False) -> Dict[str, Any]:
    """Standardized evaluation metrics for single-label and multi-label tasks."""
    if multilabel:
        y_pred_arr = np.array(y_pred)
        return {
            "hamming_loss": float(hamming_loss(y_true, y_pred_arr)),
            "micro_f1": float(f1_score(y_true, y_pred_arr, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred_arr, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred_arr, average="weighted", zero_division=0)),
            "predictions": y_pred_arr,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "predictions": y_pred,
    }


def save_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: List[Any],
    title: str,
    filename: str,
) -> None:
    """Generate and save Seaborn confusion matrix heatmap."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics_summary_json(
    summary_dict: Dict[str, Any],
    filename: str = "metrics_summary.json",
) -> None:
    """Save summary metrics to JSON in results/ with safe type conversion."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False, default=_convert)


def save_model_comparison_csv(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    filename: str = "model_comparison.csv",
) -> pd.DataFrame:
    """Save benchmark summary rows to CSV in results/."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df

```

### `src/persistence.py`
```python
from typing import Any, Dict, Optional, Tuple
import joblib
import torch

from src.config import ALL_ASPECTS, MODELS_LSTM, MODELS_TFIDF
from src.features import AspectLSTM, SentimentLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DIRS = {"tfidf": MODELS_TFIDF, "lstm": MODELS_LSTM}


def save_artifacts(
    task: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[Any] = None,
    model_type: str = "tfidf",
    vocab: Optional[Any] = None,
) -> None:
    """Save one trained task artifact. task ∈ {'sentiment', 'issue'}."""
    d = _DIRS[model_type]
    d.mkdir(parents=True, exist_ok=True)

    if model_type == "lstm":
        torch.save(model.state_dict(), d / f"{task}_model.pt")
        if vocab is not None:
            joblib.dump(vocab, d / "vocab.pkl")
    else:
        joblib.dump(model, d / f"{task}_model.pkl")
        if vectorizer is not None:
            joblib.dump(vectorizer, d / f"{task}_vectorizer.pkl")

    if binarizer is not None:
        joblib.dump(binarizer, d / f"{task}_binarizer.pkl")


def load_artifacts(
    task: str,
    model_type: str = "tfidf",
) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Any]]:
    """Load one task artifact. Returns (model, vectorizer, binarizer, vocab)."""
    d = _DIRS[model_type]
    bin_path = d / f"{task}_binarizer.pkl"
    binarizer = joblib.load(bin_path) if bin_path.exists() else None

    if model_type == "lstm":
        vocab_path = d / "vocab.pkl"
        model_path = d / f"{task}_model.pt"
        if not vocab_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Missing LSTM artifacts in {d}. Run pipeline.ipynb first.")

        vocab = joblib.load(vocab_path)
        model = (
            SentimentLSTM(vocab.vocab_size)
            if task == "sentiment"
            else AspectLSTM(vocab.vocab_size, len(ALL_ASPECTS))
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model, None, binarizer, vocab

    # TF-IDF
    model_path = d / f"{task}_model.pkl"
    vec_path = d / f"{task}_vectorizer.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing TF-IDF model at {model_path}. Run pipeline.ipynb first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path) if vec_path.exists() else None
    return model, vectorizer, binarizer, None


def save_polarity_artifacts(
    polarity_models: Dict[str, Dict[str, Any]],
    model_type: str = "tfidf",
) -> None:
    """Save aspect-specific polarity models and vectorizers."""
    d = _DIRS[model_type]
    d.mkdir(parents=True, exist_ok=True)
    for aspect, entry in polarity_models.items():
        key = aspect.lower().replace(" ", "_")
        joblib.dump(entry["model"], d / f"polarity_{key}_model.pkl")
        if "vectorizer" in entry and entry["vectorizer"] is not None:
            joblib.dump(entry["vectorizer"], d / f"polarity_{key}_vectorizer.pkl")


def load_polarity_artifacts(
    model_type: str = "tfidf",
) -> Dict[str, Dict[str, Any]]:
    """Load all saved aspect-specific polarity models."""
    d = _DIRS[model_type]
    out: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        key = aspect.lower().replace(" ", "_")
        mp, vp = d / f"polarity_{key}_model.pkl", d / f"polarity_{key}_vectorizer.pkl"
        if mp.exists():
            entry: Dict[str, Any] = {"model": joblib.load(mp)}
            if vp.exists():
                entry["vectorizer"] = joblib.load(vp)
            out[aspect] = entry
    return out

```

### `app.py`
```python
import os
import sys
from typing import Any, Dict, Optional
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

sys.path.insert(0, os.path.abspath("."))

from src.config import RESULTS_DIR
from src.persistence import (
    load_artifacts,
    load_polarity_artifacts,
)
from src.predict import (
    predict_hierarchical,
    predict_sentiment,
)
from src.preprocessing import clean_text


@st.cache_resource
def get_model_bundle(family: str) -> Dict[str, Any]:
    """Load only the selected model family artifacts."""
    polarity_models = load_polarity_artifacts("tfidf")
    if family == "TF-IDF":
        s_m, s_v, _, _ = load_artifacts("sentiment", "tfidf")
        a_m, a_v, a_b, _ = load_artifacts("issue", "tfidf")
        return {
            "sentiment_model": s_m,
            "sentiment_vectorizer": s_v,
            "aspect_model": a_m,
            "aspect_vectorizer": a_v,
            "aspect_binarizer": a_b,
            "polarity_models": polarity_models,
            "vocab": None,
        }

    s_m, _, _, s_v = load_artifacts("sentiment", "lstm")
    a_m, _, a_b, _ = load_artifacts("issue", "lstm")
    return {
        "sentiment_model": s_m,
        "sentiment_vectorizer": None,
        "aspect_model": a_m,
        "aspect_vectorizer": None,
        "aspect_binarizer": a_b,
        "polarity_models": polarity_models,
        "vocab": s_v,
    }


def load_dataset() -> pd.DataFrame:
    """Load preprocessed ABSA dataset."""
    from src.data_processing import load_data
    try:
        return load_data()
    except Exception:
        return pd.DataFrame()


def load_benchmarks() -> pd.DataFrame:
    """Load model comparison benchmarks."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        try:
            return pd.read_csv(comp_path)
        except Exception:
            pass
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(
        page_title="Bangla Daraz ABSA Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styling
    st.markdown("""
    <style>
        .main-header { font-size: 2.1rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
        .kpi-card { 
            background: #FFFFFF; 
            border: 1px solid #E2E8F0; 
            border-radius: 12px; 
            padding: 1.25rem; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .kpi-title { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
        .kpi-value { font-size: 1.6rem; font-weight: 800; margin-top: 0.25rem; }
        .sentiment-pos { color: #10B981; }
        .sentiment-neg { color: #EF4444; }
        .sentiment-neu { color: #F59E0B; }
        .aspect-row { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 8px 12px; 
            background: #F8FAFC; 
            border: 1px solid #E2E8F0; 
            border-radius: 8px; 
            margin-bottom: 6px; 
        }
        .aspect-name { font-weight: 700; color: #1E293B; font-size: 0.95rem; }
        .aspect-pill { 
            font-weight: 600; 
            font-size: 0.85rem; 
            padding: 6px 12px; 
            border-radius: 999px;
            display: flex;
            align-items: center;
            gap: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    load_data_cached = st.cache_data(load_dataset)
    load_benchmarks_cached = st.cache_data(load_benchmarks)

    dataset_df = load_data_cached()
    comparison_df = load_benchmarks_cached()

    # Sidebar
    st.sidebar.markdown("<h2>Daraz ABSA</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='color: #64748B; font-weight: 500; margin-top: -10px; margin-bottom: 20px;'>Sentiment & Aspect NLP Platform</div>", unsafe_allow_html=True)

    view_mode = st.sidebar.radio(
        "Navigation",
        ["Review Analyzer", "Benchmarks & Data Insights"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Pipeline")
    selected_model = str(st.sidebar.radio(
        "Select Model Architecture:",
        ["TF-IDF", "LSTM"],
        help="Toggle between TF-IDF (N-gram Union) and LSTM representations."
    ) or "TF-IDF")

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Corpus**: Mendeley Bangla Daraz ABSA\n\n"
        f"**Total Reviews**: {len(dataset_df):,} rows\n\n"
        f"**Aspects**: 5 Dimensions\n(Quality, Price, Delivery, Packaging, Seller)"
    )

    # 1. REVIEW ANALYZER
    if view_mode == "Review Analyzer":
        st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Hierarchical ABSA live inference via <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

        try:
            active = get_model_bundle(selected_model)
        except Exception as e:
            st.error(f"Models missing: {str(e)}. Please run `pipeline.ipynb` first.")
            st.stop()

        presets = {
            "— Choose Sample Preset —": "",
            "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
            "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
            "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
            "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
            "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল।"
        }

        selected_preset = st.selectbox("Quick Test Presets:", list(presets.keys()))
        default_text = presets.get(selected_preset or "", "")

        user_text = st.text_area(
            "Enter Bangla Review Text:",
            value=default_text,
            height=95,
            placeholder="দারাজ রিভিউ এখানে লিখুন... (e.g. প্রোডাক্ট ভালো ছিলো কিন্তু ডেলিভারি দেরি হয়েছে)"
        )

        if st.button("Analyze Review", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                try:
                    with st.spinner(f"Analyzing with {selected_model}..."):
                        s_res = predict_sentiment(
                            user_text,
                            active["sentiment_model"],
                            vectorizer=active.get("sentiment_vectorizer"),
                            vocab=active.get("vocab"),
                        )
                        pol_map: Dict[str, Dict[str, Any]] = active.get("polarity_models") or {}
                        asp_bin: Optional[MultiLabelBinarizer] = active.get("aspect_binarizer")
                        a_res = predict_hierarchical(
                            user_text,
                            active["aspect_model"],
                            polarity_models=pol_map,
                            vectorizer=active.get("aspect_vectorizer"),
                            binarizer=asp_bin,
                            vocab=active.get("vocab"),
                        )
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
                    st.stop()

                st.markdown("---")
                st.subheader("NLP Prediction Results")

                c1, c2 = st.columns([1, 1.2])

                # Sentiment Box
                with c1:
                    s_lbl = s_res["sentiment"]
                    s_cls = "sentiment-pos" if s_lbl == "Positive" else ("sentiment-neg" if s_lbl == "Negative" else "sentiment-neu")

                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Overall Predicted Sentiment</div>
                        <div class="kpi-value {s_cls}">{s_lbl}</div>
                        <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Confidence: {s_res['confidence']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if s_res["probabilities"]:
                        prob_df = pd.DataFrame(list(s_res["probabilities"].items()), columns=["Sentiment", "Probability"])
                        prob_df["Percentage"] = prob_df["Probability"] * 100
                        fig_s = px.bar(
                            prob_df, x="Percentage", y="Sentiment", orientation="h",
                            color="Sentiment",
                            color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
                            text=prob_df["Percentage"].apply(lambda p: f"{p:.1f}%")
                        )
                        fig_s.update_layout(height=160, margin=dict(t=8, b=8, l=8, r=8), showlegend=False, xaxis=dict(range=[0, 100]))
                        st.plotly_chart(fig_s, use_container_width=True)

                # Aspect Box
                with c2:
                    st.markdown("""
                    <div class="kpi-card">
                        <div class="kpi-title">Detected Aspects & Polarities</div>
                        <div style="margin-top: 10px;">
                    """, unsafe_allow_html=True)

                    aspect_details = a_res.get("aspect_details", [])
                    if aspect_details:
                        asp_html = ""
                        for item in aspect_details:
                            asp = item["aspect"]
                            pol = item["polarity"]
                            conf = item["confidence"] * 100
                            color = item["color"]
                            bg_color = item.get("bg_color", "#ECFDF5" if pol == "Positive" else "#FEF2F2")
                            asp_html += (
                                f'<div class="aspect-row">'
                                f'<div class="aspect-name">{asp}</div>'
                                f'<div class="aspect-pill" style="color: {color}; background-color: {bg_color};">'
                                f'{pol} ({conf:.0f}%)</div>'
                                f'</div>'
                            )
                        st.markdown(asp_html, unsafe_allow_html=True)
                    else:
                        st.info("No specific product aspects detected in this review.")

                    st.markdown("</div></div>", unsafe_allow_html=True)

    # 2. BENCHMARKS
    elif view_mode == "Benchmarks & Data Insights":
        st.markdown('<div class="main-header">Model Benchmarks & Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluation metrics across TF-IDF and LSTM pipelines</div>', unsafe_allow_html=True)

        if not comparison_df.empty:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Benchmark comparison file `model_comparison.csv` not found. Please run the training pipeline first.")

        st.markdown("---")
        st.subheader("Dataset Distribution & Insights")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### Sentiment Class Distribution")
            if "sentiment" in dataset_df.columns:
                sent_counts = dataset_df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                fig_dist = px.pie(
                    sent_counts, values="Count", names="Sentiment", hole=0.4,
                    color="Sentiment",
                    color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}
                )
                fig_dist.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_dist, use_container_width=True)

        with col_d2:
            st.markdown("#### Top Detected Review Aspects")
            if "aspects" in dataset_df.columns:
                all_asps = [asp for sublist in dataset_df["aspects"] for asp in sublist]
                asp_counts = pd.Series(all_asps).value_counts().reset_index()
                asp_counts.columns = ["Aspect", "Count"]
                fig_asp = px.bar(
                    asp_counts, x="Count", y="Aspect", orientation="h",
                    color="Count", color_continuous_scale="Teal"
                )
                fig_asp.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                st.plotly_chart(fig_asp, use_container_width=True)


if __name__ == "__main__":
    main()

```

### `pipeline.ipynb` (Notebook Structure & Code Cells)

#### Cell 1 [Markdown]
# Bangla Daraz ABSA Pipeline
### End-to-End Sentiment & Hierarchical Aspect-Based Sentiment Analysis (TF-IDF + PyTorch BiLSTM)
---


#### Cell 2 [Markdown]
## 1. Environment & Setup
Initialize project directories, import dependencies, and set execution paths.


#### Cell 3 [Python Code]
```python
import sys, os, time
sys.path.insert(0, os.path.abspath("."))

from src.config import ensure_dirs, SENTIMENT_LABELS, ALL_ASPECTS
from src.data_processing import load_data, get_sentiment_split, get_aspect_split
from src.features import build_tfidf, BanglaVocab, create_dataloader, SentimentLSTM
from src.train import train_sentiment, train_aspects, train_polarities, train_lstm_sentiment, train_lstm_aspects
from src.predict import predict_hierarchical, predict_sentiment
from src.evaluate import save_confusion_matrix, save_metrics_summary_json, save_model_comparison_csv
from src.persistence import save_artifacts, save_polarity_artifacts, load_artifacts, load_polarity_artifacts

ensure_dirs()
START = time.time()
print("Environment ready")

```

#### Cell 4 [Markdown]
## 2. Data Loading & Preprocessing
Load raw reviews (`annotated_bangla.csv`), clean Bangla text, parse labels into targets, and export `clean_annotated_bangla.csv`.


#### Cell 5 [Python Code]
```python
df = load_data()
Xs_tr, Xs_te, ys_tr, ys_te = get_sentiment_split(df)
Xa_tr, Xa_te, ya_tr, ya_te = get_aspect_split(df)
print(f"Loaded {len(df):,} reviews | train={len(Xs_tr):,} test={len(Xs_te):,}")

```

#### Cell 6 [Markdown]
## 3. TF-IDF Sentiment Classifier
Train 3-class Sentiment Classifier (`Positive`, `Negative`, `Neutral`) using Word + Character TF-IDF FeatureUnion and balanced Logistic Regression.


#### Cell 7 [Python Code]
```python
s_model, s_vec, s_metrics = train_sentiment(Xs_tr, ys_tr, Xs_te, ys_te)
save_artifacts("sentiment", s_model, s_vec, model_type="tfidf")
save_confusion_matrix(ys_te, s_metrics["predictions"], SENTIMENT_LABELS,
                      "Sentiment (TF-IDF)", "sentiment_tfidf.png")
print(f"Acc={s_metrics['accuracy']:.4f}  Macro-F1={s_metrics['macro_f1']:.4f}  Weighted-F1={s_metrics['weighted_f1']:.4f}")

```

#### Cell 8 [Markdown]
## 4. TF-IDF Aspect Detection & Dedicated Polarity Models
Train multi-label OneVsRest aspect classifier and dedicated binary polarity classifiers for each aspect.


#### Cell 9 [Python Code]
```python
a_model, a_vec, a_bin, a_metrics = train_aspects(Xa_tr, ya_tr, Xa_te, ya_te)
save_artifacts("issue", a_model, a_vec, a_bin, model_type="tfidf")

pol_models = train_polarities(df)
save_polarity_artifacts(pol_models, model_type="tfidf")
print(f"Aspects Micro-F1={a_metrics['micro_f1']:.4f}  Macro-F1={a_metrics['macro_f1']:.4f}  Hamming-Loss={a_metrics['hamming_loss']:.4f}")

```

#### Cell 10 [Markdown]
## 5. Vocabulary & PyTorch DataLoaders
Build Bangla vocabulary token mapping and prepare PyTorch DataLoaders for Sentiment and Multi-Label Aspect models.


#### Cell 11 [Python Code]
```python
from sklearn.preprocessing import MultiLabelBinarizer
from src.config import LSTM_BATCH_SIZE, LSTM_MAX_LENGTH

vocab = BanglaVocab()
vocab.fit(df["cleaned_text"].tolist())
print(f"Vocab size: {vocab.vocab_size}")

ys_tr_int = [SentimentLSTM.CLASSES.index(l) for l in ys_tr]
ys_te_int = [SentimentLSTM.CLASSES.index(l) for l in ys_te]
s_train = create_dataloader(Xs_tr, ys_tr_int, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE)
s_test = create_dataloader(Xs_te, ys_te_int, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE, shuffle=False)

mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
ya_tr_bin = mlb.fit_transform(ya_tr)
ya_te_bin = mlb.transform(ya_te)
a_train = create_dataloader(Xa_tr, ya_tr_bin, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE)
a_test = create_dataloader(Xa_te, ya_te_bin, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE, shuffle=False)

```

#### Cell 12 [Markdown]
## 6. PyTorch BiLSTM Sentiment & Aspect Models
Train 2-layer Bidirectional LSTM models for Sentiment Analysis and Multi-Label Aspect Detection.


#### Cell 13 [Python Code]
```python
s_lstm, s_metrics_l = train_lstm_sentiment(s_train, s_test, vocab.vocab_size, ys_te.tolist())
save_artifacts("sentiment", s_lstm, model_type="lstm", vocab=vocab)
save_confusion_matrix(ys_te, s_metrics_l["predictions"], SENTIMENT_LABELS,
                      "Sentiment (LSTM)", "sentiment_lstm.png")

a_lstm, a_bin_l, a_metrics_l = train_lstm_aspects(a_train, a_test, vocab.vocab_size, ya_te_bin, mlb)
save_artifacts("issue", a_lstm, binarizer=a_bin_l, model_type="lstm", vocab=vocab)
print(f"LSTM Sent Acc={s_metrics_l['accuracy']:.4f} | Aspect Micro-F1={a_metrics_l['micro_f1']:.4f}")

```

#### Cell 14 [Markdown]
## 7. Export Benchmark Comparison & Metrics Summary
Export comprehensive benchmark table (`model_comparison.csv`) and evaluation summary (`metrics_summary.json`).


#### Cell 15 [Python Code]
```python
rows = [
    {"Task": "Sentiment Analysis", "Model": "TF-IDF + LogReg",
     "Accuracy": round(s_metrics["accuracy"], 4), "Macro F1": round(s_metrics["macro_f1"], 4),
     "Weighted F1": round(s_metrics["weighted_f1"], 4), "Additional Metric": "N/A"},
    {"Task": "Sentiment Analysis", "Model": "LSTM",
     "Accuracy": round(s_metrics_l["accuracy"], 4), "Macro F1": round(s_metrics_l["macro_f1"], 4),
     "Weighted F1": round(s_metrics_l["weighted_f1"], 4), "Additional Metric": "N/A"},
    {"Task": "Aspect Detection", "Model": "TF-IDF + OvR",
     "Accuracy": round(1 - a_metrics["hamming_loss"], 4), "Macro F1": round(a_metrics["macro_f1"], 4),
     "Weighted F1": round(a_metrics["weighted_f1"], 4),
     "Additional Metric": f"Micro-F1: {a_metrics['micro_f1']:.4f}"},
    {"Task": "Aspect Detection", "Model": "LSTM",
     "Accuracy": round(1 - a_metrics_l["hamming_loss"], 4), "Macro F1": round(a_metrics_l["macro_f1"], 4),
     "Weighted F1": round(a_metrics_l["weighted_f1"], 4),
     "Additional Metric": f"Micro-F1: {a_metrics_l['micro_f1']:.4f}"},
]
for asp, e in pol_models.items():
    m = e["metrics"]
    rows.append({"Task": f"Polarity: {asp}", "Model": "TF-IDF + Binary LogReg",
                 "Accuracy": round(m["accuracy"], 4), "Macro F1": round(m["macro_f1"], 4),
                 "Weighted F1": round(m["weighted_f1"], 4),
                 "Additional Metric": f"n_train={e['n_train']}"})

save_model_comparison_csv(rows, "model_comparison.csv")
save_metrics_summary_json({
    "dataset": {"total": len(df)},
    "sentiment": {"tfidf": {k: v for k, v in s_metrics.items() if k != "predictions"},
                  "lstm": {k: v for k, v in s_metrics_l.items() if k != "predictions"}},
    "aspects": {"tfidf": {k: v for k, v in a_metrics.items() if k != "predictions"},
                "lstm": {k: v for k, v in a_metrics_l.items() if k != "predictions"}},
}, "metrics_summary.json")
print(f"Pipeline finished in {time.time()-START:.1f}s")

```

#### Cell 16 [Markdown]
## 8. Live Inference Demo
Test Hierarchical ABSA inference on sample Bangla e-commerce customer reviews.


#### Cell 17 [Python Code]
```python
s_m, s_v, _, _ = load_artifacts("sentiment", "tfidf")
a_m, a_v, a_b, _ = load_artifacts("issue", "tfidf")
pol = load_polarity_artifacts("tfidf")

samples = [
    "প্রোডাক্ট ভালো কিন্তু ডেলিভারি দেরি হয়েছে।",
    "দাম অনেক বেশি, সেলার রিপ্লাই দেয় না।",
    "ব্যাটারি ভালো না একদমই।",
]
for text in samples:
    s = predict_sentiment(text, s_m, vectorizer=s_v)
    h = predict_hierarchical(text, a_m, pol, vectorizer=a_v, binarizer=a_b)
    print(f"\nReview: {text}")
    print(f"   Sentiment: {s['sentiment']} ({s['confidence']*100:.0f}%)")
    for d in h["aspect_details"]:
        print(f"   Aspect: {d['aspect']} | Polarity: {d['polarity']} ({d['confidence']*100:.0f}%)")

```

# 🛒 Bangla Daraz Review Analytics & Aspect-Based Sentiment Analysis (ABSA)
## Streamlined Project Architecture, Directory Structure & Complete Codebase

---

## 📑 Table of Contents
1. [Project Overview & System Architecture](#-1-project-overview--system-architecture)
2. [Complete Directory & File Structure](#-2-complete-directory--file-structure)
3. [Environment Configuration & Dependencies](#-3-environment-configuration--dependencies)
   - [`requirements.txt`](#31-requirementstxt)
   - [`.streamlit/config.toml`](#32-streamlitconfigtoml)
   - [`.gitignore`](#33-gitignore)
4. [Core NLP Library (`src/`)](#-4-core-nlp-library-src)
   - [`src/__init__.py`](#41-src__init__py)
   - [`src/config.py`](#42-srcconfigpy)
   - [`src/preprocessing.py`](#43-srcpreprocessingpy)
   - [`src/data_processing.py`](#44-srcdata_processingpy)
   - [`src/embeddings.py`](#45-srcembeddingspy)
   - [`src/models.py`](#46-srcmodelspy)
   - [`src/evaluation.py`](#47-srcevaluationpy)
5. [Orchestration & Training Pipeline](#-5-orchestration--training-pipeline)
   - [`train_models.py`](#51-train_modelspy)
6. [Interactive Web Dashboard](#-6-interactive-web-dashboard)
   - [`app.py`](#61-apppy)
7. [Benchmark Comparison & Empirical Results](#-7-benchmark-comparison--empirical-results)
8. [Quickstart & Execution Guide](#-8-quickstart--execution-guide)

---

## 🔍 1. Project Overview & System Architecture

This project is a streamlined, pure Natural Language Processing (NLP) intelligence platform designed for **Aspect-Based Sentiment Analysis (ABSA)** on Daraz Bangladesh customer reviews.

### Core Objectives:
1. **Sentiment Analysis**: 3-class classification (`Positive`, `Neutral`, `Negative`).
2. **Aspect Detection**: Multi-label classification across 5 critical product & service dimensions:
   - **Product Quality**
   - **Price**
   - **Delivery**
   - **Packaging**
   - **Seller Service**
3. **Dual Feature Representation Benchmarking**:
   - **TF-IDF Pipeline**: Hybrid word n-grams (1, 2) + character subword n-grams (3, 5) with balanced Logistic Regression.
   - **BanglaBERT Pipeline**: 768-dimensional contextual sentence embeddings extracted from frozen `sagorsarker/bangla-bert-base` with mean pooling and balanced Logistic Regression.

```
                                  ┌───────────────────────────────┐
                                  │   Raw Bangla Daraz Reviews    │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │   Pure Bangla Text Cleaner    │
                                  │  (Unicode, Noise, Stopwords)  │
                                  └──────────────┬────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
         ┌─────────────────────────────┐                   ┌─────────────────────────────┐
         │     TF-IDF Vectorizer       │                   │    BanglaBERT Embeddings    │
         │  (Word (1,2) + Char (3,5))  │                   │  (sagorsarker/bangla-bert)  │
         └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                        │                                                 │
         ┌──────────────┴──────────────┐                   ┌──────────────┴──────────────┐
         ▼                             ▼                   ▼                             ▼
  ┌──────────────┐              ┌──────────────┐    ┌──────────────┐              ┌──────────────┐
  │  Sentiment   │              │    Aspect    │    │  Sentiment   │              │    Aspect    │
  │  Classifier  │              │ (Multi-Label)│    │  Classifier  │              │ (Multi-Label)│
  └──────────────┘              └──────────────┘    └──────────────┘              └──────────────┘
```

---

## 📁 2. Complete Directory & File Structure

```
NLP_Project/
│
├── .nlp_venv/                         # Python virtual environment
│
└── nlp_daraz/                         # Primary project root
    │
    ├── .streamlit/                    # Streamlit UI configuration
    │   └── config.toml                # Theme, port, and server settings
    │
    ├── data/                          # Dataset repository
    │   ├── bangla_stopwords.txt       # Curated 142 Bangla stopword dictionary
    │   │
    │   ├── original_data/             # Intact Mendeley raw source files
    │   │   ├── annotated_dataset.csv  # 3,587 composite labeled reviews
    │   │   ├── preprocessed_dataset.csv# 10,657 reviews (bn, banglish, mix)
    │   │   └── original_dataset.csv   # 19,638 raw Daraz reviews
    │   │
    │   └── processed_data/            # Standardized, cleaned Bangla datasets
    │       └── annotated_bangla.csv   # 2,016 cleaned Bangla reviews with ground-truth ABSA
    │
    ├── models/                        # Serialized model binaries & caches
    │   ├── tfidf/                     # TF-IDF model weights & vectorizers (*.pkl)
    │   │   ├── sentiment_model.pkl    # 3-class sentiment classifier
    │   │   ├── sentiment_vectorizer.pkl# Word + character TF-IDF union
    │   │   ├── issue_model.pkl        # Multi-label OneVsRest aspect classifier
    │   │   ├── issue_vectorizer.pkl   # Aspect TF-IDF vectorizer
    │   │   └── issue_binarizer.pkl    # MultiLabelBinarizer instance
    │   │
    │   ├── bert/                      # BanglaBERT classifier weights (*.pkl)
    │   │   ├── sentiment_model.pkl    # Logistic regression on BERT embeddings
    │   │   ├── issue_model.pkl        # OneVsRest classifier on BERT embeddings
    │   │   └── issue_binarizer.pkl    # MultiLabelBinarizer instance
    │   │
    │   └── cache/                     # Cached 768-dim sentence embeddings (*.npy)
    │       ├── sentiment_train.npy    # Precomputed training embeddings (Sentiment)
    │       ├── sentiment_test.npy     # Precomputed testing embeddings (Sentiment)
    │       ├── issue_train.npy        # Precomputed training embeddings (Aspect)
    │       └── issue_test.npy         # Precomputed testing embeddings (Aspect)
    │
    ├── results/                       # Empirical evaluation outputs
    │   ├── sentiment_tfidf.png        # Confusion matrix heatmap (TF-IDF Sentiment)
    │   ├── sentiment_bert.png         # Confusion matrix heatmap (BanglaBERT Sentiment)
    │   ├── sentiment_tfidf.csv        # Classification report (TF-IDF Sentiment)
    │   ├── sentiment_bert.csv         # Classification report (BanglaBERT Sentiment)
    │   ├── issue_tfidf.csv            # Multi-label classification report (TF-IDF Aspect)
    │   ├── issue_bert.csv             # Multi-label classification report (BERT Aspect)
    │   ├── model_comparison.csv       # 4-row consolidated benchmark table
    │   └── metrics_summary.json       # JSON export of all metrics and parameters
    │
    ├── src/                           # Clean, modular Python library
    │   ├── __init__.py                # Package declaration & version
    │   ├── config.py                  # Paths, hyper-parameters, and aspect constants
    │   ├── preprocessing.py           # Pure Bangla text cleaning pipeline
    │   ├── data_processing.py         # ABSA label parsing & stratified train/test splits
    │   ├── embeddings.py              # BanglaBERT feature extraction with mean-pooling
    │   ├── models.py                  # Model builders, training routines, save/load API
    │   └── evaluation.py              # Metrics calculation, plot generation & report export
    │
    ├── pipeline.ipynb                 # Master interactive Jupyter Notebook
    ├── train_models.py                # 1-Click end-to-end training & benchmark script
    ├── app.py                         # Interactive Streamlit Web Application
    ├── requirements.txt               # Pinned Python dependencies (CPU PyTorch default)
    ├── .gitignore                     # Git ignore rules
    └── README.md                      # Project documentation
```

---

## 📦 3. Environment Configuration & Dependencies

### 3.1 `requirements.txt`
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
torchvision>=0.15.0
transformers>=4.35.0
ipykernel>=6.25.0
```

### 3.2 `.streamlit/config.toml`
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

### 3.3 `.gitignore`
```gitignore
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

---

## 🧠 4. Core NLP Library (`src/`)

### 4.1 `src/__init__.py`
```python
"""
Bangla Daraz Review Analytics NLP Package.
"""
__version__ = "1.0.0"
```

---

### 4.2 `src/config.py`
```python
"""
Centralized Configuration and Path Management for Bangla Review Analytics.
Aspect-Based Sentiment Analysis (ABSA) on Daraz Reviews.
"""

import os
from pathlib import Path

# Project root directory (directory containing src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
ORIGINAL_DATA_DIR = DATA_DIR / "original_data"
MODELS_DIR = BASE_DIR / "models"
MODELS_TFIDF = MODELS_DIR / "tfidf"
MODELS_BERT = MODELS_DIR / "bert"
MODELS_CACHE = MODELS_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# Data Files
ANNOTATED_CSV = "annotated_bangla.csv"
STOPWORDS_FILE = "bangla_stopwords.txt"

# Tasks & Aspects
ALL_ASPECTS = [
    "Product Quality",
    "Price",
    "Delivery",
    "Packaging",
    "Seller Service"
]

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

ASPECT_MAPPING = {
    "product_quality": "Product Quality",
    "price": "Price",
    "delivery": "Delivery",
    "packaging": "Packaging",
    "seller_service": "Seller Service"
}

# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# BanglaBERT Configuration
BERT_MODEL_NAME = "sagorsarker/bangla-bert-base"
BERT_BATCH_SIZE = 32
BERT_MAX_LENGTH = 128

# TF-IDF Configuration
TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95


def ensure_dirs() -> None:
    """Create all required project directories if they do not exist."""
    directories = [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        ORIGINAL_DATA_DIR,
        MODELS_DIR,
        MODELS_TFIDF,
        MODELS_BERT,
        MODELS_CACHE,
        RESULTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
```

---

### 4.3 `src/preprocessing.py`
```python
"""
Bangla Text Preprocessing Pipeline for E-Commerce Reviews.
Implements Unicode normalization, noise reduction, Bangla-only filtering,
punctuation removal, and stopword removal.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Set

from src.config import DATA_DIR, STOPWORDS_FILE

# Pre-compiled regular expressions for high-performance batch text cleaning
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_ELONGATION = re.compile(r"(.)\1{2,}")
RE_NON_BANGLA = re.compile(r"[^\u0980-\u09FF0-9\s]")
RE_PUNCTUATION = re.compile(r"[।॥.,!?;:\"'()\[\]{}~`_/\-+=*&^%$#@<>\\]")
RE_WHITESPACE = re.compile(r"\s+")

# Bangla diacritics / zero-width characters to strip
ZERO_WIDTH_CHARS = ["\u200c", "\u200d", "\ufeff", "\u200b", "\u200e", "\u200f"]

# Cached stopwords set
_CACHED_STOPWORDS: Optional[Set[str]] = None


def normalize_unicode(text: str) -> str:
    """Normalize text to NFC form and strip zero-width characters."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, "")
    return text


def remove_html_urls(text: str) -> str:
    """Strip HTML tags, URLs, and emails from text."""
    if not isinstance(text, str):
        return ""
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    return text


def reduce_elongation(text: str) -> str:
    """Collapse 3 or more repeated characters to a single character."""
    if not isinstance(text, str):
        return ""
    return RE_ELONGATION.sub(r"\1", text)


def remove_non_bangla(text: str) -> str:
    """
    Keep only Bangla Unicode characters (\u0980-\u09FF), Bangla/English digits, and spaces.
    """
    if not isinstance(text, str):
        return ""
    return RE_NON_BANGLA.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Remove Bangla punctuation (।, ॥) and standard punctuation marks."""
    if not isinstance(text, str):
        return ""
    return RE_PUNCTUATION.sub(" ", text)


def load_bangla_stopwords(path: Optional[str | Path] = None) -> Set[str]:
    """Read Bangla stopwords from data file and return as a set."""
    global _CACHED_STOPWORDS
    if path is None:
        path = DATA_DIR / STOPWORDS_FILE

    path_obj = Path(path)
    if not path_obj.exists():
        return set()

    with open(path_obj, "r", encoding="utf-8") as f:
        stopwords = {line.strip() for line in f if line.strip() and not line.startswith("#")}

    _CACHED_STOPWORDS = stopwords
    return stopwords


def remove_stopwords(text: str, stopwords_set: Optional[Set[str]] = None) -> str:
    """Filter out stopwords from a tokenized string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    if stopwords_set is None:
        global _CACHED_STOPWORDS
        if _CACHED_STOPWORDS is None:
            _CACHED_STOPWORDS = load_bangla_stopwords()
        stopwords_set = _CACHED_STOPWORDS

    tokens = text.split()
    filtered = [t for t in tokens if t not in stopwords_set]
    return " ".join(filtered)


def clean_text(text: str, remove_sw: bool = True) -> str:
    """
    Full Bangla text preprocessing pipeline:
    1. Unicode normalization (NFC + zero-width removal)
    2. HTML / URL / Email removal
    3. Character elongation reduction (3+ to 1)
    4. Non-Bangla character removal (keeps Bangla characters & digits)
    5. Punctuation removal
    6. Whitespace collapse and trim
    7. Stopword removal (optional, default True)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalize
    text = normalize_unicode(text)

    # 2. Remove HTML, URLs, and emails
    text = remove_html_urls(text)

    # 3. Reduce elongation
    text = reduce_elongation(text)

    # 4. Remove non-Bangla
    text = remove_non_bangla(text)

    # 5. Remove punctuation
    text = remove_punctuation(text)

    # 6. Collapse whitespace & trim
    text = RE_WHITESPACE.sub(" ", text).strip()

    # 7. Remove stopwords
    if remove_sw:
        text = remove_stopwords(text)

    return text
```

---

### 4.4 `src/data_processing.py`
```python
"""
Data Processing and Splitting Pipeline for Bangla Daraz Review Analytics.
Handles loading annotated ABSA reviews and stratified 80/20 train/test splitting.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_ASPECTS,
    ANNOTATED_CSV,
    ASPECT_MAPPING,
    DATA_DIR,
    ORIGINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE
)
from src.preprocessing import clean_text


def parse_absa_labels(label_str: str) -> Dict[str, Any]:
    """
    Parse composite ABSA label strings (e.g., 'delivery_positive#product_quality_negative').
    Returns:
      - aspects: list of canonical aspect names
      - overall_sentiment: 'Positive', 'Negative', or 'Neutral'
    """
    if not isinstance(label_str, str) or not label_str.strip():
        return {"aspects": [], "overall_sentiment": "Neutral"}

    parts = [p.strip() for p in label_str.split("#") if p.strip()]
    aspects = set()
    sentiments = []

    for part in parts:
        if "_" in part:
            aspect_key, polarity = part.rsplit("_", 1)
            canonical_aspect = ASPECT_MAPPING.get(aspect_key, aspect_key.replace("_", " ").title())
            aspects.add(canonical_aspect)
            sentiments.append(polarity.lower())

    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")

    if pos_count > neg_count:
        overall_sentiment = "Positive"
    elif neg_count > pos_count:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {
        "aspects": sorted(list(aspects)),
        "overall_sentiment": overall_sentiment
    }


def load_annotated_data() -> pd.DataFrame:
    """
    Load the clean annotated Bangla ABSA dataset (2,016 rows).
    Self-heals and builds from original source files if missing.
    """
    processed_path = PROCESSED_DATA_DIR / ANNOTATED_CSV
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        if "aspects_str" in df.columns and "aspects" not in df.columns:
            df["aspects"] = df["aspects_str"].fillna("").apply(
                lambda s: [x.strip() for x in str(s).split(";") if x.strip()]
            )
        return df

    # Build from original dataset if processed file doesn't exist
    anno_orig = ORIGINAL_DATA_DIR / "annotated_dataset.csv"
    prep_orig = ORIGINAL_DATA_DIR / "preprocessed_dataset.csv"

    if not anno_orig.exists():
        raise FileNotFoundError(f"Source annotated dataset not found at {anno_orig.resolve()}")

    df = pd.read_csv(anno_orig)
    if prep_orig.exists() and "language" not in df.columns:
        prep_df = pd.read_csv(prep_orig, usecols=["review_id", "language"]).drop_duplicates(subset=["review_id"])
        df = df.merge(prep_df, on="review_id", how="left")
        df = df[df["language"] == "bn"].copy()

    raw_col = "original_text" if "original_text" in df.columns else ("text" if "text" in df.columns else "content")
    df = df.dropna(subset=["label", raw_col]).drop_duplicates(subset=["review_id"]).copy()

    parsed = [parse_absa_labels(lbl) for lbl in df["label"]]
    df["sentiment"] = [p["overall_sentiment"] for p in parsed]
    df["aspects"] = [p["aspects"] for p in parsed]
    df["aspects_str"] = [";".join(p["aspects"]) for p in parsed]
    df["cleaned_text"] = df[raw_col].apply(clean_text)
    df = df[df["cleaned_text"].str.strip().str.len() > 0].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df


def get_sentiment_split(
    anno_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratified 80/20 train/test split for sentiment analysis.
    Returns: X_train, X_test, y_train, y_test
    """
    valid = anno_df.dropna(subset=["sentiment", "cleaned_text"]).copy()
    X = valid["cleaned_text"]
    y = valid["sentiment"]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def get_aspect_split(
    anno_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, List[List[str]], List[List[str]]]:
    """
    80/20 train/test split for multi-label aspect detection.
    Returns: X_train, X_test, y_train, y_test
    """
    valid = anno_df.dropna(subset=["cleaned_text"]).copy()
    if "aspects" in valid.columns:
        y_aspects = valid["aspects"].tolist()
    else:
        y_aspects = valid["aspects_str"].fillna("").apply(
            lambda s: [a.strip() for a in str(s).split(";") if a.strip()]
        ).tolist()

    y_clean = [
        item if isinstance(item, list) else [a.strip() for a in str(item).split(";") if a.strip()]
        for item in y_aspects
    ]

    X = valid["cleaned_text"].reset_index(drop=True)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = [y_clean[i] for i in train_idx]
    y_test = [y_clean[i] for i in test_idx]

    return X_train, X_test, y_train, y_test
```

---

### 4.5 `src/embeddings.py`
```python
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
        _TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
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
```

---

### 4.6 `src/models.py`
```python
"""
Unified Machine Learning Models and Artifact Management for Bangla Review Analytics.
Aspect-Based Sentiment Analysis: Sentiment (3-class) & Aspect Detection (Multi-Label).
Supports dual feature representations: TF-IDF (Word + Char Union) & BanglaBERT.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss
)

from src.config import (
    ALL_ASPECTS,
    MODELS_BERT,
    MODELS_TFIDF,
    RANDOM_STATE,
    TFIDF_CHAR_NGRAMS,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_WORD_NGRAMS
)
from src.preprocessing import clean_text


# -----------------------------------------------------------------------------
# EVALUATION & FEATURE EXTRACTION HELPERS
# -----------------------------------------------------------------------------
def _evaluate_multiclass(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute standard multi-class classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "predictions": y_pred,
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }


def _evaluate_multilabel(
    y_true_bin: Any,
    y_pred_bin: Any,
    target_names: List[str] = ALL_ASPECTS
) -> Dict[str, Any]:
    """Compute multi-label classification metrics and hamming loss."""
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    return {
        "accuracy": float(1.0 - h_loss),
        "micro_f1": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        "hamming_loss": h_loss,
        "predictions": y_pred_bin,
        "report": classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
    }


def _extract_features(
    text_or_features: Union[str, np.ndarray],
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Tuple[Any, str]:
    """Uniformly extract features from raw Bangla text or precomputed representations."""
    if not isinstance(text_or_features, str):
        return text_or_features, ""

    raw_text = text_or_features
    cleaned = clean_text(raw_text)

    if use_bert:
        from src.embeddings import get_bert_features
        feat = get_bert_features([cleaned if cleaned else "ভালো"])
    else:
        if not cleaned or vectorizer is None:
            return None, ""
        feat = vectorizer.transform([cleaned])

    return feat, cleaned


# -----------------------------------------------------------------------------
# TF-IDF FEATURE BUILDER
# -----------------------------------------------------------------------------
def build_tfidf_union() -> FeatureUnion:
    """Build hybrid FeatureUnion combining word n-grams and character n-grams."""
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[\u0980-\u09FF\w]+",
        ngram_range=TFIDF_WORD_NGRAMS,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=TFIDF_CHAR_NGRAMS,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True
    )
    return FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer)
    ])


# -----------------------------------------------------------------------------
# 1. SENTIMENT ANALYSIS
# -----------------------------------------------------------------------------
def train_sentiment_tfidf(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Train TF-IDF + Balanced Logistic Regression for Sentiment Analysis."""
    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    metrics = _evaluate_multiclass(y_test, preds)
    return model, vectorizer, metrics


def train_sentiment_bert(
    X_train_bert: np.ndarray,
    y_train: pd.Series,
    X_test_bert: np.ndarray,
    y_test: pd.Series
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train Balanced Logistic Regression on frozen BanglaBERT embeddings."""
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    model.fit(X_train_bert, y_train)
    preds = model.predict(X_test_bert)
    metrics = _evaluate_multiclass(y_test, preds)
    return model, metrics


def predict_sentiment(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict sentiment ('Negative', 'Neutral', 'Positive') and class probabilities."""
    feat, _ = _extract_features(text_or_features, vectorizer, use_bert=use_bert)

    if feat is None:
        return {
            "sentiment": "Neutral",
            "confidence": 0.34,
            "probabilities": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}
        }

    pred = str(model.predict(feat)[0])
    probs_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        for cls, prob in zip(model.classes_, probs):
            probs_dict[str(cls)] = float(prob)
        confidence = float(np.max(probs))
    else:
        confidence = 1.0

    return {
        "sentiment": pred,
        "confidence": confidence,
        "probabilities": probs_dict
    }


# -----------------------------------------------------------------------------
# 2. ASPECT DETECTION (MULTI-LABEL)
# -----------------------------------------------------------------------------
def train_aspect_tfidf(
    X_train: pd.Series,
    y_train: List[List[str]],
    X_test: pd.Series,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, FeatureUnion, MultiLabelBinarizer, Dict[str, Any]]:
    """Train OneVsRestClassifier for multi-label aspect detection with TF-IDF."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_vec, y_train_bin)

    preds_bin = ovr_model.predict(X_test_vec)
    metrics = _evaluate_multilabel(y_test_bin, preds_bin)
    return ovr_model, vectorizer, mlb, metrics


def train_aspect_bert(
    X_train_bert: np.ndarray,
    y_train: List[List[str]],
    X_test_bert: np.ndarray,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, MultiLabelBinarizer, Dict[str, Any]]:
    """Train OneVsRestClassifier for multi-label aspect detection on BanglaBERT features."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    base_lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_bert, y_train_bin)

    preds_bin = ovr_model.predict(X_test_bert)
    metrics = _evaluate_multilabel(y_test_bin, preds_bin)
    return ovr_model, mlb, metrics


# Aliases for backward compatibility
train_issue_tfidf = train_aspect_tfidf
train_issue_bert = train_aspect_bert


def predict_aspects(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """
    Predict multi-label aspect tags and confidence scores purely using Machine Learning.
    Returns: {"aspects": [...], "confidences": {...}}
    """
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    feat, _ = _extract_features(text_or_features, vectorizer, use_bert=use_bert)

    if feat is None:
        return {
            "aspects": ["Product Quality"],
            "confidences": {"Product Quality": 0.5}
        }

    preds_bin = model.predict(feat)
    detected = list(binarizer.inverse_transform(preds_bin)[0])

    if len(detected) == 0:
        detected = ["Product Quality"]

    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        for i, asp in enumerate(binarizer.classes_):
            confidences[str(asp)] = float(probs[i])
    else:
        for asp in ALL_ASPECTS:
            confidences[asp] = 1.0 if asp in detected else 0.0

    return {
        "aspects": detected,
        "confidences": confidences
    }


# Alias for backward compatibility
predict_issue = predict_aspects


# -----------------------------------------------------------------------------
# 3. ARTIFACT SAVE & LOAD
# -----------------------------------------------------------------------------
def save_artifacts(
    task: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    model_type: str = "tfidf"
) -> None:
    """Save trained model artifacts to models/{model_type}/ directory."""
    target_dir = MODELS_TFIDF if model_type == "tfidf" else MODELS_BERT
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, target_dir / f"{task}_model.pkl")

    if vectorizer is not None:
        joblib.dump(vectorizer, target_dir / f"{task}_vectorizer.pkl")

    if binarizer is not None:
        joblib.dump(binarizer, target_dir / f"{task}_binarizer.pkl")


def load_artifacts(
    task: str,
    model_type: str = "tfidf"
) -> Tuple[Any, Optional[Any], Optional[MultiLabelBinarizer]]:
    """Load model artifacts from models/{model_type}/ directory."""
    target_dir = MODELS_TFIDF if model_type == "tfidf" else MODELS_BERT
    model_path = target_dir / f"{task}_model.pkl"
    vec_path = target_dir / f"{task}_vectorizer.pkl"
    bin_path = target_dir / f"{task}_binarizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path.resolve()}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path) if vec_path.exists() else None
    binarizer = joblib.load(bin_path) if bin_path.exists() else None

    return model, vectorizer, binarizer
```

---

### 4.7 `src/evaluation.py`
```python
"""
Evaluation and Metrics Reporting Module for Bangla Review Analytics.
Plots confusion matrices, writes classification reports to CSV, and saves JSON summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import RESULTS_DIR


def plot_and_save_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: List[Any],
    title: str,
    filename: str
) -> None:
    """Generate and save a high-resolution Seaborn confusion matrix heatmap directly to results/."""
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
        linewidths=0.5
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report_csv(
    report_dict: Dict[str, Any],
    filename: str
) -> None:
    """Save a Scikit-learn classification report dictionary as a CSV directly to results/."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(output_path, index=True)


def save_metrics_summary_json(
    summary_dict: Dict[str, Any],
    filename: str = "metrics_summary.json"
) -> None:
    """Save summary metrics JSON directly to results/."""
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
    data_or_s_tf: Any,
    *args: Any,
    filename: str = "model_comparison.csv"
) -> pd.DataFrame:
    """
    Save model comparison table to results/ directory and return DataFrame.
    Supports:
      1. save_model_comparison_csv(comparison_rows_or_df, filename="model_comparison.csv")
      2. save_model_comparison_csv(s_metrics_tfidf, s_metrics_bert, a_metrics_tfidf, a_metrics_bert)
    """
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data_or_s_tf, pd.DataFrame):
        df = data_or_s_tf
    elif isinstance(data_or_s_tf, list):
        df = pd.DataFrame(data_or_s_tf)
    elif len(args) >= 3:
        s_tf = data_or_s_tf
        s_bt, a_tf, a_bt = args[0], args[1], args[2]
        rows = [
            {
                "Task": "Sentiment Analysis",
                "Model": "TF-IDF + Logistic Regression",
                "Accuracy": round(s_tf.get("accuracy", 0.0), 4),
                "Macro F1": round(s_tf.get("macro_f1", 0.0), 4),
                "Weighted F1": round(s_tf.get("weighted_f1", 0.0), 4),
                "Additional Metric": "N/A"
            },
            {
                "Task": "Sentiment Analysis",
                "Model": "BanglaBERT + Logistic Regression",
                "Accuracy": round(s_bt.get("accuracy", 0.0), 4),
                "Macro F1": round(s_bt.get("macro_f1", 0.0), 4),
                "Weighted F1": round(s_bt.get("weighted_f1", 0.0), 4),
                "Additional Metric": "N/A"
            },
            {
                "Task": "Aspect Detection",
                "Model": "TF-IDF + OneVsRest LogReg",
                "Accuracy": round(a_tf.get("accuracy", 0.0), 4),
                "Macro F1": round(a_tf.get("macro_f1", 0.0), 4),
                "Weighted F1": round(a_tf.get("weighted_f1", 0.0), 4),
                "Additional Metric": f"Hamming Loss: {a_tf.get('hamming_loss', 0.0):.4f}"
            },
            {
                "Task": "Aspect Detection",
                "Model": "BanglaBERT + OneVsRest LogReg",
                "Accuracy": round(a_bt.get("accuracy", 0.0), 4),
                "Macro F1": round(a_bt.get("macro_f1", 0.0), 4),
                "Weighted F1": round(a_bt.get("weighted_f1", 0.0), 4),
                "Additional Metric": f"Hamming Loss: {a_bt.get('hamming_loss', 0.0):.4f}"
            }
        ]
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()

    df.to_csv(output_path, index=False)
    return df
```

---

## 🚀 5. Orchestration & Training Pipeline

### 5.1 `train_models.py`
```python
"""
Orchestration Training Pipeline for Bangla Review Analytics.
Trains and benchmarks both TF-IDF and BanglaBERT model variants across:
1. Sentiment Analysis (3-class)
2. Aspect Detection (Multi-Label)
"""

import os
import sys
import time
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.config import (
    ALL_ASPECTS,
    MODELS_CACHE,
    MODELS_DIR,
    RESULTS_DIR,
    SENTIMENT_LABELS,
    ensure_dirs
)
from src.data_processing import (
    get_aspect_split,
    get_sentiment_split,
    load_annotated_data
)
from src.embeddings import get_or_cache_bert_features
from src.evaluation import (
    plot_and_save_confusion_matrix,
    save_classification_report_csv,
    save_metrics_summary_json,
    save_model_comparison_csv
)
from src.models import (
    save_artifacts,
    train_aspect_bert,
    train_aspect_tfidf,
    train_sentiment_bert,
    train_sentiment_tfidf
)


def main():
    start_time = time.time()
    print("=" * 70)
    print("   BANGLA DARAZ REVIEW ANALYTICS - ABSA TRAINING PIPELINE   ")
    print("=" * 70)

    ensure_dirs()

    # 1. Load Data
    print("\n[Step 1/5] Loading annotated Bangla ABSA dataset...")
    df = load_annotated_data()
    print(f"  ✓ Loaded {len(df):,} expert-annotated reviews.")

    # 2. Train/Test Splits
    print("\n[Step 2/5] Creating stratified 80/20 train/test splits...")
    X_train_s, X_test_s, y_train_s, y_test_s = get_sentiment_split(df)
    X_train_a, X_test_a, y_train_a, y_test_a = get_aspect_split(df)
    print(f"  ✓ Sentiment Split: {len(X_train_s):,} train | {len(X_test_s):,} test")
    print(f"  ✓ Aspect Split:    {len(X_train_a):,} train | {len(X_test_a):,} test")

    # 3. TF-IDF Models
    print("\n[Step 3/5] Training TF-IDF (Word + Char Union) models...")
    s_model_tf, s_vec_tf, s_metrics_tf = train_sentiment_tfidf(X_train_s, y_train_s, X_test_s, y_test_s)
    save_artifacts("sentiment", s_model_tf, s_vec_tf, model_type="tfidf")
    plot_and_save_confusion_matrix(
        y_test_s, s_metrics_tf["predictions"],
        labels=SENTIMENT_LABELS,
        title="Sentiment Confusion Matrix (TF-IDF)",
        filename="sentiment_tfidf.png"
    )
    save_classification_report_csv(s_metrics_tf["report"], "sentiment_tfidf.csv")
    print(f"  ✓ [TF-IDF] Sentiment -> Acc: {s_metrics_tf['accuracy']:.4f} | Macro-F1: {s_metrics_tf['macro_f1']:.4f}")

    a_model_tf, a_vec_tf, a_mlb_tf, a_metrics_tf = train_aspect_tfidf(X_train_a, y_train_a, X_test_a, y_test_a)
    save_artifacts("issue", a_model_tf, a_vec_tf, a_mlb_tf, model_type="tfidf")
    save_classification_report_csv(a_metrics_tf["report"], "issue_tfidf.csv")
    print(f"  ✓ [TF-IDF] Aspects   -> Micro-F1: {a_metrics_tf['micro_f1']:.4f} | Hamming Loss: {a_metrics_tf['hamming_loss']:.4f}")

    # 4. BanglaBERT Models
    print("\n[Step 4/5] Extracting BanglaBERT embeddings & training classifiers...")
    s_train_bert = get_or_cache_bert_features(X_train_s, MODELS_CACHE / "sentiment_train.npy")
    s_test_bert = get_or_cache_bert_features(X_test_s, MODELS_CACHE / "sentiment_test.npy")
    a_train_bert = get_or_cache_bert_features(X_train_a, MODELS_CACHE / "issue_train.npy")
    a_test_bert = get_or_cache_bert_features(X_test_a, MODELS_CACHE / "issue_test.npy")

    s_model_bt, s_metrics_bt = train_sentiment_bert(s_train_bert, y_train_s, s_test_bert, y_test_s)
    save_artifacts("sentiment", s_model_bt, model_type="bert")
    plot_and_save_confusion_matrix(
        y_test_s, s_metrics_bt["predictions"],
        labels=SENTIMENT_LABELS,
        title="Sentiment Confusion Matrix (BanglaBERT)",
        filename="sentiment_bert.png"
    )
    save_classification_report_csv(s_metrics_bt["report"], "sentiment_bert.csv")
    print(f"  ✓ [BanglaBERT] Sentiment -> Acc: {s_metrics_bt['accuracy']:.4f} | Macro-F1: {s_metrics_bt['macro_f1']:.4f}")

    a_model_bt, a_mlb_bt, a_metrics_bt = train_aspect_bert(a_train_bert, y_train_a, a_test_bert, y_test_a)
    save_artifacts("issue", a_model_bt, binarizer=a_mlb_bt, model_type="bert")
    save_classification_report_csv(a_metrics_bt["report"], "issue_bert.csv")
    print(f"  ✓ [BanglaBERT] Aspects   -> Micro-F1: {a_metrics_bt['micro_f1']:.4f} | Hamming Loss: {a_metrics_bt['hamming_loss']:.4f}")

    # 5. Export Benchmark Reports
    print("\n[Step 5/5] Exporting benchmark comparison tables & summaries...")
    comparison_rows = [
        {
            "Task": "Sentiment Analysis",
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": round(s_metrics_tf["accuracy"], 4),
            "Macro F1": round(s_metrics_tf["macro_f1"], 4),
            "Weighted F1": round(s_metrics_tf["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Sentiment Analysis",
            "Model": "BanglaBERT + Logistic Regression",
            "Accuracy": round(s_metrics_bt["accuracy"], 4),
            "Macro F1": round(s_metrics_bt["macro_f1"], 4),
            "Weighted F1": round(s_metrics_bt["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Aspect Detection",
            "Model": "TF-IDF + OneVsRest LogReg",
            "Accuracy": round(a_metrics_tf["accuracy"], 4),
            "Macro F1": round(a_metrics_tf["macro_f1"], 4),
            "Weighted F1": round(a_metrics_tf["weighted_f1"], 4),
            "Additional Metric": f"Hamming Loss: {a_metrics_tf['hamming_loss']:.4f}"
        },
        {
            "Task": "Aspect Detection",
            "Model": "BanglaBERT + OneVsRest LogReg",
            "Accuracy": round(a_metrics_bt["accuracy"], 4),
            "Macro F1": round(a_metrics_bt["macro_f1"], 4),
            "Weighted F1": round(a_metrics_bt["weighted_f1"], 4),
            "Additional Metric": f"Hamming Loss: {a_metrics_bt['hamming_loss']:.4f}"
        }
    ]
    save_model_comparison_csv(comparison_rows, "model_comparison.csv")

    metrics_summary = {
        "dataset_statistics": {
            "total_reviews": len(df),
            "language": "bn",
            "sentiment_distribution": df["sentiment"].value_counts().to_dict()
        },
        "sentiment_analysis": {
            "tfidf": s_metrics_tf,
            "banglabert": s_metrics_bt
        },
        "aspect_detection": {
            "tfidf": a_metrics_tf,
            "banglabert": a_metrics_bt
        }
    }
    save_metrics_summary_json(metrics_summary, "metrics_summary.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"          TRAINING PIPELINE COMPLETED IN {elapsed:.1f}s!         ")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---

## 🌐 6. Interactive Web Dashboard

### 6.1 `app.py`
```python
"""
Streamlit Web Application: Bangla Review Sentiment & Aspect Analytics Platform
Aspect-Based Sentiment Analysis (ABSA) for Daraz Bangladesh.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.config import ALL_ASPECTS, RESULTS_DIR
from src.models import load_artifacts, predict_aspects, predict_sentiment
from src.preprocessing import clean_text

# Page Config
st.set_page_config(
    page_title="Bangla Daraz ABSA Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #4B5563; margin-bottom: 1.2rem; }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px 18px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-title { font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .sentiment-pos { color: #10B981; }
    .sentiment-neg { color: #EF4444; }
    .sentiment-neu { color: #F59E0B; }
    .aspect-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 6px 14px;
        border-radius: 16px;
        font-size: 0.92rem;
        font-weight: 600;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load both TF-IDF and BanglaBERT model artifacts."""
    loaded = {"status": "ready", "tfidf": {}, "bert": {}}
    try:
        s_m_tf, s_v_tf, _ = load_artifacts("sentiment", "tfidf")
        a_m_tf, a_v_tf, a_b_tf = load_artifacts("issue", "tfidf")
        loaded["tfidf"] = {
            "sentiment_model": s_m_tf,
            "sentiment_vectorizer": s_v_tf,
            "aspect_model": a_m_tf,
            "aspect_vectorizer": a_v_tf,
            "aspect_binarizer": a_b_tf
        }

        s_m_bt, _, _ = load_artifacts("sentiment", "bert")
        a_m_bt, _, a_b_bt = load_artifacts("issue", "bert")
        loaded["bert"] = {
            "sentiment_model": s_m_bt,
            "aspect_model": a_m_bt,
            "aspect_binarizer": a_b_bt or a_b_tf
        }
    except Exception as e:
        loaded["status"] = "error"
        loaded["message"] = str(e)
    return loaded


@st.cache_data
def load_dataset():
    """Load annotated ABSA dataset."""
    from src.data_processing import load_annotated_data
    try:
        return load_annotated_data()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_benchmarks():
    """Load model comparison benchmarks from results/."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path)
    return pd.DataFrame()


models_data = load_all_models()
dataset_df = load_dataset()
comparison_df = load_benchmarks()

# Sidebar
st.sidebar.title("🛒 Daraz ABSA Analytics")
st.sidebar.markdown("**Sentiment & Aspect NLP Platform**")

view_mode = st.sidebar.radio(
    "Navigation",
    ["🔍 Review Analyzer", "📈 Benchmarks & Data Insights"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Architecture")
selected_model = st.sidebar.radio(
    "Select Model:",
    ["TF-IDF", "BanglaBERT"],
    help="Toggle between TF-IDF (N-gram Union) and BanglaBERT feature representations."
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Corpus**: Mendeley ABSA Dataset\n\n"
    f"**Total Annotated Reviews**: {len(dataset_df):,} rows\n\n"
    f"**Aspects**: 5 Product Dimensions"
)


# -----------------------------------------------------------------------------
# 1. REVIEW ANALYZER
# -----------------------------------------------------------------------------
if view_mode == "🔍 Review Analyzer":
    st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Live NLP inference using <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

    if models_data["status"] != "ready":
        st.error(f"Models not loaded: {models_data.get('message')}. Please run `python train_models.py` first.")
        st.stop()

    use_bert = (selected_model == "BanglaBERT")
    active = models_data["bert"] if use_bert else models_data["tfidf"]

    presets = {
        "Select a sample review preset...": "",
        "Preset 1 (Positive Quality & Fast Delivery)": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
        "Preset 2 (Negative Delay & Poor Quality)": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
        "Preset 3 (Damaged Packaging & Good Product)": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
        "Preset 4 (Excellent Seller Service & Accurate)": "সেলার খুব ভালো ব্যবহার করেছে, প্রোডাক্ট যেমন দেখেছি তেমনই পেয়েছি।",
        "Preset 5 (Price Concern & Quality Failure)": "দাম অনেক বেশি কিন্তু কোয়ালিটি একদম বাজে।"
    }

    selected_preset = st.selectbox("💡 Quick Test Presets:", list(presets.keys()))
    default_text = presets[selected_preset]

    user_text = st.text_area(
        "Enter Bangla Review Text:",
        value=default_text,
        height=95,
        placeholder="দারাজ রিভিউ এখানে লিখুন... (e.g. প্রোডাক্ট ভালো ছিলো কিন্তু ডেলিভারি দেরি হয়েছে)"
    )

    if st.button("🚀 Analyze Review", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner(f"Analyzing with {selected_model}..."):
                cleaned = clean_text(user_text)

                if use_bert:
                    s_res = predict_sentiment(user_text, active["sentiment_model"], use_bert=True)
                    a_res = predict_aspects(
                        user_text,
                        active["aspect_model"],
                        binarizer=active["aspect_binarizer"],
                        use_bert=True
                    )
                else:
                    s_res = predict_sentiment(
                        user_text,
                        active["sentiment_model"],
                        vectorizer=active["sentiment_vectorizer"],
                        use_bert=False
                    )
                    a_res = predict_aspects(
                        user_text,
                        active["aspect_model"],
                        vectorizer=active["aspect_vectorizer"],
                        binarizer=active["aspect_binarizer"],
                        use_bert=False
                    )

            st.markdown("---")
            st.subheader("NLP Prediction Results")

            c1, c2 = st.columns([1, 1.2])

            # Sentiment Box
            with c1:
                s_lbl = s_res["sentiment"]
                s_cls = "sentiment-pos" if s_lbl == "Positive" else ("sentiment-neg" if s_lbl == "Negative" else "sentiment-neu")
                s_ico = "😊" if s_lbl == "Positive" else ("😡" if s_lbl == "Negative" else "😐")

                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Predicted Sentiment</div>
                    <div class="kpi-value {s_cls}">{s_ico} {s_lbl}</div>
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
                    st.plotly_chart(fig_s, width="stretch")

            # Aspect Box
            with c2:
                st.markdown("""
                <div class="kpi-card">
                    <div class="kpi-title">Detected Aspects</div>
                    <div style="margin-top: 10px;">
                """, unsafe_allow_html=True)

                asp_html = ""
                for asp in a_res["aspects"]:
                    asp_html += f'<span class="aspect-badge">🏷️ {asp}</span>'
                st.markdown(asp_html, unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            with st.expander("🔍 Cleaned Bangla Tokens"):
                st.code(cleaned, language="text")


# -----------------------------------------------------------------------------
# 2. BENCHMARKS & DATA INSIGHTS
# -----------------------------------------------------------------------------
elif view_mode == "📈 Benchmarks & Data Insights":
    st.markdown('<div class="main-header">Model Performance & Empirical Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Direct evaluation on held-out 20% test sets (404 reviews).</div>', unsafe_allow_html=True)

    if not comparison_df.empty:
        st.subheader("1. Dual-Model Benchmark Summary (4 Configurations)")
        st.dataframe(comparison_df, width="stretch")
    else:
        st.info("Run `python train_models.py` to generate the benchmark table.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrices
    st.subheader("2. Sentiment Confusion Matrices")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("**TF-IDF Confusion Matrix**")
        p_tf = RESULTS_DIR / "sentiment_tfidf.png"
        if p_tf.exists():
            st.image(str(p_tf), width="stretch")
    with c_m2:
        st.markdown("**BanglaBERT Confusion Matrix**")
        p_bt = RESULTS_DIR / "sentiment_bert.png"
        if p_bt.exists():
            st.image(str(p_bt), width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Explorer
    st.subheader("3. Dataset Distribution & Explorer")
    d1, d2 = st.columns([1, 1.3])

    if not dataset_df.empty and "sentiment" in dataset_df.columns:
        with d1:
            sent_counts = dataset_df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig_p = px.pie(
                sent_counts, names="Sentiment", values="Count", hole=0.4,
                color="Sentiment",
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}
            )
            fig_p.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_p, width="stretch")

        with d2:
            aspect_items = []
            for item in dataset_df["aspects_str"].dropna():
                for a in str(item).split(";"):
                    if a.strip():
                        aspect_items.append(a.strip())
            asp_s = pd.Series(aspect_items).value_counts().reset_index()
            asp_s.columns = ["Aspect", "Mentions"]
            fig_b = px.bar(asp_s, x="Mentions", y="Aspect", orientation="h", color="Mentions", color_continuous_scale="Blues")
            fig_b.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_b, width="stretch")

    st.dataframe(dataset_df[["cleaned_text", "sentiment", "aspects_str"]].head(10).rename(columns={
        "cleaned_text": "Review (Bangla)",
        "sentiment": "Sentiment",
        "aspects_str": "Aspects"
    }), width="stretch")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
    "Bangla Daraz Review Analytics • Sentiment & Aspect NLP Platform"
    "</div>",
    unsafe_allow_html=True
)

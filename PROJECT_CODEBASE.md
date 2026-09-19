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

This project is a streamlined, production-grade Natural Language Processing (NLP) intelligence platform designed for **Hierarchical Aspect-Based Sentiment Analysis (ABSA)** on Daraz Bangladesh customer reviews.

### Core Objectives:
1. **Sentiment Analysis**: 3-class classification (`Positive`, `Neutral`, `Negative`).
2. **Aspect Detection**: Multi-label classification across 5 critical product & service dimensions:
   - **Product Quality**
   - **Price**
   - **Delivery**
   - **Packaging**
   - **Seller Service**
3. **Aspect-Level Polarity Classification (Approach C)**: Dedicated binary classifiers (`Positive` vs. `Negative`) for each detected aspect, eliminating the extreme class sparsity of neutral polarity.
4. **Dual Feature Representation Benchmarking**:
   - **TF-IDF Pipeline**: Hybrid word n-grams (1, 2) + character subword n-grams (3, 5) with balanced Logistic Regression.
   - **BanglaBERT Pipeline**: 768-dimensional contextual sentence embeddings extracted from `sagorsarker/bangla-bert-base` with mean pooling and balanced Logistic Regression.

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
         ┌──────────────┼──────────────┐                   ┌──────────────┴──────────────┐
         ▼              ▼              ▼                   ▼                             ▼
  ┌──────────────┐┌──────────────┐┌──────────────┐  ┌──────────────┐              ┌──────────────┐
  │  Sentiment   ││    Aspect    ││Aspect-Level  │  │  Sentiment   │              │    Aspect    │
  │  Classifier  ││ (Multi-Label)││Polarity (x5) │  │  Classifier  │              │ (Multi-Label)│
  │  (3-Class)   ││ (5 Aspects)  ││(Pos vs Neg)  │  │  (3-Class)   │              │ (5 Aspects)  │
  └──────────────┘└──────────────┘└──────────────┘  └──────────────┘              └──────────────┘
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
    │   │   ├── issue_model.pkl        # Multi-label aspect detection model
    │   │   ├── issue_vectorizer.pkl   # Aspect TF-IDF vectorizer
    │   │   ├── issue_binarizer.pkl    # MultiLabelBinarizer instance
    │   │   ├── polarity_product_quality_model.pkl
    │   │   ├── polarity_product_quality_vectorizer.pkl
    │   │   ├── polarity_price_model.pkl
    │   │   ├── polarity_price_vectorizer.pkl
    │   │   ├── polarity_delivery_model.pkl
    │   │   ├── polarity_delivery_vectorizer.pkl
    │   │   ├── polarity_packaging_model.pkl
    │   │   ├── polarity_packaging_vectorizer.pkl
    │   │   ├── polarity_seller_service_model.pkl
    │   │   └── polarity_seller_service_vectorizer.pkl
    │   │
    │   ├── bert/                      # BanglaBERT Logistic Regression classifiers
    │   │   ├── sentiment_model.pkl    # 3-class sentiment classifier
    │   │   └── issue_model.pkl        # Multi-label aspect detection model
    │   │
    │   └── cache/                     # Cached precomputed 768-dim embeddings
    │       ├── sentiment_train.npy    # (1612, 768) float32
    │       ├── sentiment_test.npy     # (404, 768) float32
    │       ├── issue_train.npy        # (1612, 768) float32
    │       └── issue_test.npy         # (404, 768) float32
    │
    ├── results/                       # Empirical reports, metrics & plots
    │   ├── model_comparison.csv       # Unified benchmark table
    │   ├── metrics_summary.json       # Complete machine-readable metrics
    │   ├── sentiment_tfidf.png        # 300 DPI Seaborn confusion matrix
    │   ├── sentiment_bert.png         # 300 DPI Seaborn confusion matrix
    │   ├── sentiment_tfidf.csv        # Classification report
    │   ├── sentiment_bert.csv         # Classification report
    │   ├── issue_tfidf.csv            # Aspect classification report
    │   ├── issue_bert.csv             # Aspect classification report
    │   ├── polarity_product_quality_tfidf.csv
    │   ├── polarity_price_tfidf.csv
    │   ├── polarity_delivery_tfidf.csv
    │   ├── polarity_packaging_tfidf.csv
    │   └── polarity_seller_service_tfidf.csv
    │
    ├── src/                           # Pure NLP modular library
    │   ├── __init__.py                # Package initialization & exports
    │   ├── config.py                  # Paths, constants, aspect mappings
    │   ├── preprocessing.py           # Pure regex Bangla cleaner & stopword filter
    │   ├── data_processing.py         # Dataset loading, label parsing, stratified splits
    │   ├── embeddings.py              # Frozen BanglaBERT feature extractor & caching
    │   ├── models.py                  # TF-IDF unions, models, hierarchical ABSA & persistence
    │   └── evaluation.py              # Confusion matrix heatmaps & report exporters
    │
    ├── train_models.py                # Pipeline orchestrator
    ├── app.py                         # Streamlit interactive ABSA analytics app
    ├── requirements.txt               # Minimal environment dependencies
    └── PROJECT_CODEBASE.md            # Comprehensive project documentation
```

---

## ⚙️ 3. Environment Configuration & Dependencies

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

### 4.2 `src/config.py`
```python
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
ORIGINAL_DATA_DIR = DATA_DIR / "original_data"
MODELS_DIR = BASE_DIR / "models"
MODELS_TFIDF = MODELS_DIR / "tfidf"
MODELS_BERT = MODELS_DIR / "bert"
MODELS_CACHE = MODELS_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# Data files
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

# BanglaBERT settings
BERT_MODEL_NAME = "sagorsarker/bangla-bert-base"
BERT_BATCH_SIZE = 32
BERT_MAX_LENGTH = 128

# TF-IDF settings
TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95


def ensure_dirs() -> None:
    """Create required project directories if missing."""
    for directory in [
        DATA_DIR, PROCESSED_DATA_DIR, ORIGINAL_DATA_DIR,
        MODELS_DIR, MODELS_TFIDF, MODELS_BERT, MODELS_CACHE, RESULTS_DIR
    ]:
        os.makedirs(directory, exist_ok=True)
```

### 4.3 `src/preprocessing.py`
```python
import re
import unicodedata
from pathlib import Path
from typing import Optional, Set

from src.config import DATA_DIR, STOPWORDS_FILE

# Pre-compiled regex patterns
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_ELONGATION = re.compile(r"(.)\1{2,}")
RE_NON_BANGLA = re.compile(r"[^\u0980-\u09FF0-9\s]")
RE_PUNCTUATION = re.compile(r"[।॥.,!?;:\"'()\[\]{}~`_/\-+=*&^%$#@<>\\]")
RE_WHITESPACE = re.compile(r"\s+")
ZERO_WIDTH_CHARS = ["\u200c", "\u200d", "\ufeff", "\u200b", "\u200e", "\u200f"]

_CACHED_STOPWORDS: Optional[Set[str]] = None


def normalize_unicode(text: str) -> str:
    """Normalize text to NFC form and strip zero-width characters."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, "")
    return text


def load_bangla_stopwords(path: Optional[str | Path] = None) -> Set[str]:
    """Load Bangla stopwords from disk."""
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


def clean_text(text: str, remove_sw: bool = True) -> str:
    """Clean and normalize Bangla review text."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_ELONGATION.sub(r"\1", text)
    text = RE_NON_BANGLA.sub(" ", text)
    text = RE_PUNCTUATION.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()

    if remove_sw:
        global _CACHED_STOPWORDS
        if _CACHED_STOPWORDS is None:
            _CACHED_STOPWORDS = load_bangla_stopwords()
        tokens = [t for t in text.split() if t not in _CACHED_STOPWORDS]
        text = " ".join(tokens)

    return text
```

### 4.4 `src/data_processing.py`
```python
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ANNOTATED_CSV,
    ASPECT_MAPPING,
    ORIGINAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE
)
from src.preprocessing import clean_text


def parse_absa_labels(label_str: str) -> Dict[str, Any]:
    """Parse composite ABSA labels into aspects, polarities, and overall sentiment."""
    if not isinstance(label_str, str) or not label_str.strip():
        return {"aspects": [], "aspect_polarities": {}, "overall_sentiment": "Neutral"}

    parts = [p.strip() for p in label_str.split("#") if p.strip()]
    aspects = set()
    sentiments = []
    aspect_polarities = {}

    for part in parts:
        if "_" in part:
            aspect_key, polarity = part.rsplit("_", 1)
            canonical = ASPECT_MAPPING.get(aspect_key, aspect_key.replace("_", " ").title())
            aspects.add(canonical)
            pol = polarity.lower()
            sentiments.append(pol)
            aspect_polarities[canonical] = pol

    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")
    if pos_count > neg_count:
        overall = "Positive"
    elif neg_count > pos_count:
        overall = "Negative"
    else:
        overall = "Neutral"

    return {
        "aspects": sorted(list(aspects)),
        "aspect_polarities": aspect_polarities,
        "overall_sentiment": overall
    }


def load_annotated_data() -> pd.DataFrame:
    """Load or self-heal the annotated Bangla dataset (2,016 rows)."""
    processed_path = PROCESSED_DATA_DIR / ANNOTATED_CSV
    if processed_path.exists():
        df = pd.read_csv(processed_path)
        if "aspects_str" in df.columns and "aspects" not in df.columns:
            df["aspects"] = df["aspects_str"].fillna("").apply(
                lambda s: [x.strip() for x in str(s).split(";") if x.strip()]
            )
        if "sentiment" not in df.columns or "aspect_polarities" not in df.columns:
            if "label" in df.columns:
                parsed = [parse_absa_labels(lbl) for lbl in df["label"]]
                if "sentiment" not in df.columns:
                    df["sentiment"] = [p["overall_sentiment"] for p in parsed]
                if "aspect_polarities" not in df.columns:
                    df["aspect_polarities"] = [p["aspect_polarities"] for p in parsed]
        return df

    anno_orig = ORIGINAL_DATA_DIR / "annotated_dataset.csv"
    prep_orig = ORIGINAL_DATA_DIR / "preprocessed_dataset.csv"

    if not anno_orig.exists():
        raise FileNotFoundError(f"Source dataset not found at {anno_orig.resolve()}")

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
    df["aspect_polarities"] = [p["aspect_polarities"] for p in parsed]
    df["aspects_str"] = [";".join(p["aspects"]) for p in parsed]
    df["cleaned_text"] = df[raw_col].apply(clean_text)
    df = df[df["cleaned_text"].str.strip().str.len() > 0].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df


def get_sentiment_split(
    anno_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split for sentiment analysis."""
    valid = anno_df.dropna(subset=["sentiment", "cleaned_text"]).copy()
    return train_test_split(
        valid["cleaned_text"],
        valid["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=valid["sentiment"]
    )


def get_aspect_split(
    anno_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, List[List[str]], List[List[str]]]:
    """80/20 train/test split for multi-label aspect detection."""
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

    return X.iloc[train_idx], X.iloc[test_idx], [y_clean[i] for i in train_idx], [y_clean[i] for i in test_idx]


def get_aspect_polarity_splits(
    anno_df: pd.DataFrame,
    aspect: str
) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """Extract binary (Positive/Negative) split for a specific aspect."""
    valid = anno_df.dropna(subset=["cleaned_text"]).copy()
    if "aspect_polarities" not in valid.columns:
        valid["aspect_polarities"] = valid["label"].apply(lambda l: parse_absa_labels(l)["aspect_polarities"])

    rows = []
    for _, row in valid.iterrows():
        p_dict = row["aspect_polarities"]
        if isinstance(p_dict, dict) and aspect in p_dict:
            pol = p_dict[aspect].lower()
            if pol in ("positive", "negative"):
                rows.append({"text": row["cleaned_text"], "polarity": pol.title()})

    sub_df = pd.DataFrame(rows).dropna()
    if len(sub_df) < 10:
        return None

    counts = sub_df["polarity"].value_counts().to_dict()
    stratify = sub_df["polarity"] if min(counts.values()) >= 2 else None

    return train_test_split(
        sub_df["text"],
        sub_df["polarity"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )
```

### 4.5 `src/embeddings.py`
```python
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    torch = None
    AutoModel = None
    AutoTokenizer = None
    HAS_TORCH = False

from src.config import BERT_BATCH_SIZE, BERT_MAX_LENGTH, BERT_MODEL_NAME


def _get_device() -> Any:
    """Detect available compute device."""
    if not HAS_TORCH or torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(cpu_cores)
    return torch.device("cpu")


_DEVICE: Any = None
_TOKENIZER: Optional[Any] = None
_MODEL: Optional[Any] = None


def load_banglabert() -> Tuple[Any, Any]:
    """Lazily load and cache BanglaBERT model and tokenizer."""
    global _TOKENIZER, _MODEL, _DEVICE
    if not HAS_TORCH or AutoTokenizer is None or AutoModel is None or torch is None:
        raise ImportError("PyTorch & Transformers required: pip install torch transformers")
    if _DEVICE is None:
        _DEVICE = _get_device()
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
            batch_texts = [t if t.strip() else "ভালো" for t in text_list[i : i + batch_size]]
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

    embeddings = get_bert_features(texts, batch_size=batch_size, max_length=max_length)
    np.save(cache_file, embeddings)
    return embeddings
```

### 4.6 `src/models.py`
```python
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss

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
# 1. EVALUATION & FEATURE HELPERS
# -----------------------------------------------------------------------------
def _evaluate_classification(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute accuracy, macro/weighted F1 scores, and classification report."""
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
    """Compute multi-label F1 scores, hamming loss, and classification report."""
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    return {
        "accuracy": float(1.0 - h_loss),
        "micro_f1": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        "hamming_loss": h_loss,
        "predictions": y_pred_bin,
        "report": classification_report(
            y_true_bin, y_pred_bin, target_names=target_names, output_dict=True, zero_division=0
        )
    }


def _extract_features(
    text_or_features: Union[str, np.ndarray],
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Tuple[Any, str]:
    """Extract features from raw Bangla text or precomputed representations."""
    if not isinstance(text_or_features, str):
        return text_or_features, ""

    cleaned = clean_text(text_or_features)
    if use_bert:
        from src.embeddings import get_bert_features
        feat = get_bert_features([cleaned if cleaned else "ভালো"])
    else:
        if not cleaned or vectorizer is None:
            return None, ""
        feat = vectorizer.transform([cleaned])

    return feat, cleaned


def build_tfidf_union() -> FeatureUnion:
    """Build word + character n-gram TF-IDF FeatureUnion."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[\u0980-\u09FF\w]+",
            ngram_range=TFIDF_WORD_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_CHAR_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True
        ))
    ])


# -----------------------------------------------------------------------------
# 2. MODEL TRAINING ROUTINES
# -----------------------------------------------------------------------------
def train_sentiment_tfidf(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Train TF-IDF + Logistic Regression for 3-class sentiment analysis."""
    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    return model, vectorizer, _evaluate_classification(y_test, preds)


def train_sentiment_bert(
    X_train_bert: np.ndarray,
    y_train: pd.Series,
    X_test_bert: np.ndarray,
    y_test: pd.Series
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train Logistic Regression on frozen BanglaBERT embeddings for sentiment analysis."""
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(X_train_bert, y_train)
    preds = model.predict(X_test_bert)
    return model, _evaluate_classification(y_test, preds)


def train_aspect_tfidf(
    X_train: pd.Series,
    y_train: List[List[str]],
    X_test: pd.Series,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, FeatureUnion, MultiLabelBinarizer, Dict[str, Any]]:
    """Train TF-IDF + OneVsRest Logistic Regression for multi-label aspect detection."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_vec, y_train_bin)

    preds_bin = ovr_model.predict(X_test_vec)
    return ovr_model, vectorizer, mlb, _evaluate_multilabel(y_test_bin, preds_bin)


def train_aspect_bert(
    X_train_bert: np.ndarray,
    y_train: List[List[str]],
    X_test_bert: np.ndarray,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, MultiLabelBinarizer, Dict[str, Any]]:
    """Train BanglaBERT + OneVsRest Logistic Regression for multi-label aspect detection."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    base_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_bert, y_train_bin)

    preds_bin = ovr_model.predict(X_test_bert)
    return ovr_model, mlb, _evaluate_multilabel(y_test_bin, preds_bin)


def train_aspect_polarity_tfidf(
    anno_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Train dedicated binary (Positive vs. Negative) classifiers for each aspect."""
    from src.data_processing import get_aspect_polarity_splits

    results: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        split = get_aspect_polarity_splits(anno_df, aspect)
        if split is None:
            continue

        X_train, X_test, y_train, y_test = split
        vectorizer = build_tfidf_union()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        results[aspect] = {
            "model": model,
            "vectorizer": vectorizer,
            "metrics": _evaluate_classification(y_test, preds),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": model.classes_.tolist()
        }

    return results


# -----------------------------------------------------------------------------
# 3. INFERENCE ROUTINES
# -----------------------------------------------------------------------------
def predict_sentiment(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict 3-class sentiment ('Negative', 'Neutral', 'Positive') and confidence."""
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
        probs_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs_dict = {pred: 1.0}

    return {"sentiment": pred, "confidence": confidence, "probabilities": probs_dict}


def predict_aspects(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict present aspect categories using multi-label classification."""
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    feat, _ = _extract_features(text_or_features, vectorizer, use_bert=use_bert)
    if feat is None:
        return {"aspects": [], "confidences": {asp: 0.0 for asp in ALL_ASPECTS}}

    preds_bin = model.predict(feat)
    detected = list(binarizer.inverse_transform(preds_bin)[0])

    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        confidences = {str(asp): float(probs[i]) for i, asp in enumerate(binarizer.classes_)}
    else:
        confidences = {asp: (1.0 if asp in detected else 0.0) for asp in ALL_ASPECTS}

    return {"aspects": detected, "confidences": confidences}


def _polarity_fallback(aspect: str) -> Dict[str, Any]:
    """Default fallback dictionary for missing aspect polarity models."""
    return {
        "aspect": aspect,
        "polarity": "Positive",
        "confidence": 0.50,
        "is_low_confidence": True,
        "probabilities": {"Positive": 0.5, "Negative": 0.5}
    }


def predict_aspect_polarity(
    text: str,
    aspect: str,
    polarity_models: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """Predict binary polarity (Positive / Negative) with confidence and low-confidence flag."""
    if not text or not text.strip() or aspect not in polarity_models:
        return _polarity_fallback(aspect)

    entry = polarity_models[aspect]
    model, vectorizer = entry["model"], entry.get("vectorizer")
    cleaned = clean_text(text)
    if not cleaned or vectorizer is None:
        return _polarity_fallback(aspect)

    feat = vectorizer.transform([cleaned])
    pred = str(model.predict(feat)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        probs_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs_dict = {pred: 1.0}

    is_low_confidence = confidence < min_confidence
    return {
        "aspect": aspect,
        "polarity": pred,
        "confidence": confidence,
        "is_low_confidence": is_low_confidence,
        "probabilities": probs_dict
    }


def predict_aspects_with_polarity(
    text_or_features: Union[str, np.ndarray],
    aspect_model: Any,
    polarity_models: Dict[str, Dict[str, Any]],
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    use_bert: bool = False,
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """Hierarchical ABSA: Detect aspects, then predict specific binary polarity per aspect."""
    aspect_res = predict_aspects(
        text_or_features, aspect_model, vectorizer=vectorizer, binarizer=binarizer, use_bert=use_bert
    )
    raw_text = text_or_features if isinstance(text_or_features, str) else ""
    aspect_details = []

    for asp in aspect_res["aspects"]:
        if raw_text and asp in polarity_models:
            pol_info = predict_aspect_polarity(raw_text, asp, polarity_models, min_confidence=min_confidence)
            pol_lbl = pol_info["polarity"]
            conf = pol_info["confidence"]
            is_low = pol_info["is_low_confidence"]
        else:
            pol_lbl = "Positive"
            conf = 0.70
            is_low = False

        if is_low:
            icon = "⚠️"
            color = "#D97706"
            bg_color = "#FFFBEB"
        elif pol_lbl == "Positive":
            icon = "✅"
            color = "#10B981"
            bg_color = "#ECFDF5"
        else:
            icon = "😡"
            color = "#EF4444"
            bg_color = "#FEF2F2"

        aspect_details.append({
            "aspect": asp,
            "polarity": pol_lbl,
            "confidence": conf,
            "is_low_confidence": is_low,
            "icon": icon,
            "color": color,
            "bg_color": bg_color
        })

    return {
        "aspects": aspect_res["aspects"],
        "aspect_details": aspect_details,
        "confidences": aspect_res["confidences"]
    }


# -----------------------------------------------------------------------------
# 4. ARTIFACT PERSISTENCE
# -----------------------------------------------------------------------------
def save_artifacts(
    task: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    model_type: str = "tfidf"
) -> None:
    """Save trained model artifacts to models/{model_type}/."""
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
    """Load model artifacts from models/{model_type}/."""
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


def save_polarity_artifacts(
    polarity_models: Dict[str, Dict[str, Any]],
    model_type: str = "tfidf"
) -> None:
    """Save aspect-specific polarity models and vectorizers."""
    target_dir = MODELS_TFIDF if model_type == "tfidf" else MODELS_BERT
    target_dir.mkdir(parents=True, exist_ok=True)
    for aspect, entry in polarity_models.items():
        safe_key = aspect.lower().replace(" ", "_")
        joblib.dump(entry["model"], target_dir / f"polarity_{safe_key}_model.pkl")
        if "vectorizer" in entry and entry["vectorizer"] is not None:
            joblib.dump(entry["vectorizer"], target_dir / f"polarity_{safe_key}_vectorizer.pkl")


def load_polarity_artifacts(
    model_type: str = "tfidf"
) -> Dict[str, Dict[str, Any]]:
    """Load all saved aspect-specific polarity models."""
    target_dir = MODELS_TFIDF if model_type == "tfidf" else MODELS_BERT
    polarity_models: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        safe_key = aspect.lower().replace(" ", "_")
        model_path = target_dir / f"polarity_{safe_key}_model.pkl"
        vec_path = target_dir / f"polarity_{safe_key}_vectorizer.pkl"
        if model_path.exists():
            entry: Dict[str, Any] = {"model": joblib.load(model_path)}
            if vec_path.exists():
                entry["vectorizer"] = joblib.load(vec_path)
            polarity_models[aspect] = entry
    return polarity_models
```

### 4.7 `src/evaluation.py`
```python
import json
from typing import Any, Dict, List, Union
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
    """Generate and save Seaborn confusion matrix heatmap."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True, linewidths=0.5)
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
    """Save Scikit-learn classification report as CSV."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(output_path, index=True)


def save_metrics_summary_json(
    summary_dict: Dict[str, Any],
    filename: str = "metrics_summary.json"
) -> None:
    """Save summary metrics to JSON."""
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
    filename: str = "model_comparison.csv"
) -> pd.DataFrame:
    """Save benchmark rows to CSV in results/."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df
```

---

## 🚀 5. Orchestration & Training Pipeline

### 5.1 `train_models.py`
```python
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.config import MODELS_CACHE, SENTIMENT_LABELS, ensure_dirs
from src.data_processing import get_aspect_split, get_sentiment_split, load_annotated_data
from src.embeddings import get_or_cache_bert_features
from src.evaluation import (
    plot_and_save_confusion_matrix,
    save_classification_report_csv,
    save_metrics_summary_json,
    save_model_comparison_csv
)
from src.models import (
    save_artifacts,
    save_polarity_artifacts,
    train_aspect_bert,
    train_aspect_polarity_tfidf,
    train_aspect_tfidf,
    train_sentiment_bert,
    train_sentiment_tfidf
)


def main():
    start_time = time.time()
    print("=" * 60)
    print("   BANGLA DARAZ ABSA - TRAINING & BENCHMARKING PIPELINE   ")
    print("=" * 60)

    ensure_dirs()

    # 1. Load Data & Splits
    print("\n[Step 1/4] Loading dataset and creating 80/20 splits...")
    df = load_annotated_data()
    X_train_s, X_test_s, y_train_s, y_test_s = get_sentiment_split(df)
    X_train_a, X_test_a, y_train_a, y_test_a = get_aspect_split(df)
    print(f"  ✓ Loaded {len(df):,} reviews | Sentiment Split: {len(X_train_s)} train, {len(X_test_s)} test")

    # 2. TF-IDF Models
    print("\n[Step 2/4] Training TF-IDF models (Sentiment, Aspects, Polarities)...")
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

    polarity_models_tf = train_aspect_polarity_tfidf(df)
    save_polarity_artifacts(polarity_models_tf, model_type="tfidf")
    for asp, p_data in polarity_models_tf.items():
        m = p_data["metrics"]
        safe_name = asp.lower().replace(" ", "_")
        save_classification_report_csv(m["report"], f"polarity_{safe_name}_tfidf.csv")
        print(f"  ✓ [TF-IDF] {asp:15s} Polarity -> Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

    # 3. BanglaBERT Models
    print("\n[Step 3/4] Extracting BanglaBERT embeddings & training classifiers...")
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

    # 4. Save Benchmark Summaries
    print("\n[Step 4/4] Exporting benchmark tables and metrics...")
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

    for asp, p_data in polarity_models_tf.items():
        m = p_data["metrics"]
        comparison_rows.append({
            "Task": f"Polarity: {asp}",
            "Model": "TF-IDF + Balanced Binary LogReg",
            "Accuracy": round(m["accuracy"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "Weighted F1": round(m["weighted_f1"], 4),
            "Additional Metric": f"Samples: {p_data['n_train']+p_data['n_test']} (Train:{p_data['n_train']})"
        })

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
        },
        "aspect_polarities": {
            asp: {
                "accuracy": p_data["metrics"]["accuracy"],
                "macro_f1": p_data["metrics"]["macro_f1"],
                "n_train": p_data["n_train"],
                "n_test": p_data["n_test"]
            }
            for asp, p_data in polarity_models_tf.items()
        }
    }
    save_metrics_summary_json(metrics_summary, "metrics_summary.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"       TRAINING PIPELINE COMPLETED IN {elapsed:.1f}s!      ")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 🖥️ 6. Interactive Web Dashboard

### 6.1 `app.py`
```python
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.abspath("."))

from src.config import RESULTS_DIR
from src.models import (
    load_artifacts,
    load_polarity_artifacts,
    predict_aspects_with_polarity,
    predict_sentiment
)
from src.preprocessing import clean_text

# Streamlit Page Config
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
    .aspect-row {
        margin-bottom: 8px;
        padding: 8px 14px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .aspect-name { font-weight: 600; color: #1E293B; font-size: 0.95rem; }
    .aspect-pill { font-weight: 600; font-size: 0.85rem; padding: 3px 10px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load model artifacts and polarity classifiers."""
    loaded = {"status": "ready", "tfidf": {}, "bert": {}}
    try:
        s_m_tf, s_v_tf, _ = load_artifacts("sentiment", "tfidf")
        a_m_tf, a_v_tf, a_b_tf = load_artifacts("issue", "tfidf")
        p_models_tf = load_polarity_artifacts("tfidf")

        loaded["tfidf"] = {
            "sentiment_model": s_m_tf,
            "sentiment_vectorizer": s_v_tf,
            "aspect_model": a_m_tf,
            "aspect_vectorizer": a_v_tf,
            "aspect_binarizer": a_b_tf,
            "polarity_models": p_models_tf
        }

        s_m_bt, _, _ = load_artifacts("sentiment", "bert")
        a_m_bt, _, a_b_bt = load_artifacts("issue", "bert")
        loaded["bert"] = {
            "sentiment_model": s_m_bt,
            "aspect_model": a_m_bt,
            "aspect_binarizer": a_b_bt or a_b_tf,
            "polarity_models": p_models_tf
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
    """Load model comparison benchmarks."""
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
    help="Toggle between TF-IDF (N-gram Union) and BanglaBERT representations."
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Corpus**: Mendeley ABSA Dataset\n\n"
    f"**Total Annotated Reviews**: {len(dataset_df):,} rows\n\n"
    f"**Aspects**: 5 Dimensions (Quality, Price, Delivery, Packaging, Seller)"
)


# -----------------------------------------------------------------------------
# 1. REVIEW ANALYZER
# -----------------------------------------------------------------------------
if view_mode == "🔍 Review Analyzer":
    st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Hierarchical ABSA live inference via <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

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
    default_text = presets.get(selected_preset or "", "")

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
            try:
                with st.spinner(f"Analyzing with {selected_model}..."):
                    cleaned = clean_text(user_text)

                    if use_bert:
                        s_res = predict_sentiment(user_text, active["sentiment_model"], use_bert=True)
                        a_res = predict_aspects_with_polarity(
                            user_text,
                            active["aspect_model"],
                            polarity_models=active.get("polarity_models", {}),
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
                        a_res = predict_aspects_with_polarity(
                            user_text,
                            active["aspect_model"],
                            polarity_models=active.get("polarity_models", {}),
                            vectorizer=active["aspect_vectorizer"],
                            binarizer=active["aspect_binarizer"],
                            use_bert=False
                        )
            except ImportError as ie:
                st.error(f"⚠️ {str(ie)}")
                st.info("💡 Switch to the **TF-IDF** pipeline in the sidebar for instant real-time inference without PyTorch.")
                st.stop()
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
                s_ico = "😊" if s_lbl == "Positive" else ("😡" if s_lbl == "Negative" else "😐")

                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Overall Predicted Sentiment</div>
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
                    <div class="kpi-title">Detected Aspects & Specific Polarities</div>
                    <div style="margin-top: 10px;">
                """, unsafe_allow_html=True)

                aspect_details = a_res.get("aspect_details", [])
                if aspect_details:
                    asp_html = ""
                    for item in aspect_details:
                        asp = item["aspect"]
                        pol = item["polarity"]
                        icon = item["icon"]
                        conf = item["confidence"] * 100
                        color = item["color"]
                        bg_color = item.get("bg_color", "#ECFDF5" if pol == "Positive" else "#FEF2F2")
                        asp_html += (
                            f'<div class="aspect-row">'
                            f'  <span class="aspect-name">🏷️ {asp}</span>'
                            f'  <span class="aspect-pill" style="color: {color}; background-color: {bg_color}; border: 1px solid {color}44;">'
                            f'    {icon} {pol} <span style="font-size: 0.78rem; opacity: 0.85;">({conf:.1f}%)</span>'
                            f'  </span>'
                            f'</div>'
                        )
                    st.markdown(asp_html, unsafe_allow_html=True)
                else:
                    st.markdown("<em>No specific aspect detected.</em>", unsafe_allow_html=True)

                st.markdown("</div></div>", unsafe_allow_html=True)

            with st.expander("🔍 Cleaned Bangla Tokens"):
                st.code(cleaned, language="text")


# -----------------------------------------------------------------------------
# 2. BENCHMARKS & DATA INSIGHTS
# -----------------------------------------------------------------------------
elif view_mode == "📈 Benchmarks & Data Insights":
    st.markdown('<div class="main-header">Model Performance & Empirical Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation on held-out 20% test sets (Stratified ABSA splits).</div>', unsafe_allow_html=True)

    if not comparison_df.empty:
        st.subheader("1. Comprehensive Model Benchmark Summary")
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
    "Bangla Daraz Review Analytics • Hierarchical Aspect-Based Sentiment Analysis (ABSA)"
    "</div>",
    unsafe_allow_html=True
)
```

---

## 📊 7. Benchmark Comparison & Empirical Results

The models are rigorously trained and evaluated on stratified held-out 20% test sets (404 reviews for sentiment and aspect detection, and aspect-specific stratified test splits for polarities).

| Task | Model Architecture | Accuracy | Macro F1 | Weighted F1 | Additional Metric / Sample Size |
|---|---|---|---|---|---|
| **Sentiment Analysis** | TF-IDF + Logistic Regression | **87.62%** | **77.47%** | **87.77%** | 3 Classes (`Pos`, `Neu`, `Neg`) |
| **Sentiment Analysis** | BanglaBERT + Logistic Regression | **82.92%** | **69.66%** | **83.22%** | Frozen 768-dim embeddings |
| **Aspect Detection** | TF-IDF + OneVsRest LogReg | **96.34%** | **78.04%** | **93.53%** | Hamming Loss: `0.0366` |
| **Aspect Detection** | BanglaBERT + OneVsRest LogReg | **92.97%** | **65.01%** | **89.37%** | Hamming Loss: `0.0703` |
| **Polarity: Product Quality** | TF-IDF + Balanced Binary LogReg | **96.43%** | **93.47%** | **96.42%** | Total: 1,816 (Train: 1,452, Test: 364) |
| **Polarity: Price** | TF-IDF + Balanced Binary LogReg | **93.68%** | **60.85%** | **94.44%** | Total: 471 (Train: 376, Test: 95) |
| **Polarity: Delivery** | TF-IDF + Balanced Binary LogReg | **87.72%** | **73.13%** | **85.98%** | Total: 282 (Train: 225, Test: 57) |
| **Polarity: Packaging** | TF-IDF + Balanced Binary LogReg | **86.67%** | **82.95%** | **85.61%** | Total: 72 (Train: 57, Test: 15) |
| **Polarity: Seller Service** | TF-IDF + Balanced Binary LogReg | **95.45%** | **48.84%** | **93.23%** | Total: 108 (Train: 86, Test: 22) |

---

## ⚡ 8. Quickstart & Execution Guide

### 1. Environment Setup
```bash
cd /home/shuvo/Documents/NLP_Project
source .nlp_venv/bin/activate
cd nlp_daraz
pip install -r requirements.txt
```

### 2. Run Training & Benchmarking Pipeline
```bash
python train_models.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

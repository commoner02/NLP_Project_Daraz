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

This project is a streamlined, production-grade Natural Language Processing (NLP) platform designed for **Hierarchical Aspect-Based Sentiment Analysis (ABSA)** on customer reviews from Daraz Bangladesh.

### Core Objectives:
1. **Sentiment Analysis**: 3-class classification (`Positive`, `Neutral`, `Negative`).
2. **Aspect Detection**: Multi-label classification across 5 product & service dimensions:
   - **Product Quality**
   - **Price**
   - **Delivery**
   - **Packaging**
   - **Seller Service**
3. **Aspect-Level Polarity Classification**: Dedicated binary classifiers (`Positive` vs. `Negative`) for each detected aspect, resolving polarity sparsity.
4. **Dual Representation Benchmarking**:
   - **TF-IDF Pipeline**: Hybrid word n-grams (1, 2) + character subword n-grams (3, 5) with balanced Logistic Regression.
   - **BanglaBERT Pipeline**: 768-dimensional contextual embeddings extracted from `sagorsarker/bangla-bert-base` with mean pooling and balanced Logistic Regression.
5. **Robust Preprocessing & Universal Delimiters**:
   - Preserves negations (*"না"*, *"নাই"*, *"নয়"*, *"নেই"*) and contrastive conjunctions (*"কিন্তু"*).
   - Supports mixed-script/code-switching loanwords (*"battery"*, *"delivery"*, *"product"*).
   - Normalizes both `#` and `;` composite label formats.

```
                                  ┌───────────────────────────────┐
                                  │   Raw Bangla Daraz Reviews    │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │   Bangla Text Normalizer      │
                                  │ (Unicode NFC, Zero-Width,     │
                                  │  Noise, Negation-Preserving)  │
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
└── nlp_daraz/                         # Primary project root
    │
    ├── .streamlit/                    # Streamlit UI configuration
    │   └── config.toml                # Theme and server settings
    │
    ├── data/                          # Dataset repository
    │   ├── original_data/             # Mendeley raw source files
    │   │   ├── annotated_dataset.csv  # 3,587 composite labeled reviews
    │   │   ├── preprocessed_dataset.csv# 10,657 reviews
    │   │   └── original_dataset.csv   # 19,638 raw Daraz reviews
    │   │
    │   └── processed_data/            # Standardized Bangla datasets
    │       └── annotated_bangla.csv   # 2,016 annotated Bangla reviews
    │
    ├── models/                        # Serialized model binaries & caches
    │   ├── tfidf/                     # TF-IDF model weights & vectorizers (*.pkl)
    │   │   ├── sentiment_model.pkl
    │   │   ├── sentiment_vectorizer.pkl
    │   │   ├── issue_model.pkl
    │   │   ├── issue_vectorizer.pkl
    │   │   ├── issue_binarizer.pkl
    │   │   └── polarity_*_model.pkl
    │   │
    │   ├── bert/                      # BanglaBERT classification heads (*.pkl)
    │   │   ├── sentiment_model.pkl
    │   │   ├── issue_model.pkl
    │   │   └── issue_binarizer.pkl
    │   │
    │   └── cache/                     # Precomputed 768-dim BERT embeddings (*.npy)
    │       ├── sentiment_train.npy
    │       ├── sentiment_test.npy
    │       ├── issue_train.npy
    │       └── issue_test.npy
    │
    ├── results/                       # Evaluation figures & benchmark tables
    │   ├── metrics_summary.json       # Complete precision/recall/F1 JSON metrics
    │   ├── model_comparison.csv       # Unified benchmark comparison table
    │   ├── sentiment_tfidf.png        # TF-IDF confusion matrix heatmap
    │   └── sentiment_bert.png         # BanglaBERT confusion matrix heatmap
    │
    ├── src/                           # Reusable NLP library package
    │   ├── __init__.py                # Package versioning
    │   ├── config.py                  # Paths, aspects, and model hyperparameters
    │   ├── preprocessing.py           # Unicode NFC, zero-width & text cleaning
    │   ├── data_processing.py         # CSV loading, delimiter parsing & splits
    │   ├── embeddings.py              # BanglaBERT extraction & .npy caching
    │   ├── models.py                  # Training, hierarchical ABSA & persistence
    │   └── evaluation.py              # Heatmap plots & JSON/CSV exporters
    │
    ├── app.py                         # Interactive Streamlit analytics dashboard
    ├── train_models.py                # End-to-end training & benchmarking script
    ├── requirements.txt               # Dependencies with CPU PyTorch index
    ├── .gitignore                     # Git tracking exclusions
    ├── PROJECT_CODEBASE.md            # Comprehensive project documentation
    └── README.md                      # Project overview and quickstart
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

## 📦 4. Core NLP Library (`src/`)

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
MODELS_DIR = BASE_DIR / "models"
MODELS_TFIDF = MODELS_DIR / "tfidf"
MODELS_BERT = MODELS_DIR / "bert"
MODELS_CACHE = MODELS_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

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
    for directory in [DATA_DIR, MODELS_TFIDF, MODELS_BERT, MODELS_CACHE, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
```

### 4.3 `src/preprocessing.py`
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

### 4.4 `src/data_processing.py`
```python
import ast
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_ASPECTS,
    ASPECT_MAPPING,
    DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocessing import clean_text

PROCESSED_FILE = DATA_DIR / "processed_data" / "annotated_bangla.csv"
ASPECT_MIN_SAMPLES = 20


def parse_absa_labels(label_str: str) -> Dict[str, Any]:
    """Parse composite ABSA labels (delimited by '#' or ';') into structured data."""
    if not isinstance(label_str, str) or not label_str.strip():
        return {"aspects": [], "aspect_polarities": {}, "overall_sentiment": "Neutral"}

    parts = [p.strip() for p in re.split(r"[#;]", label_str) if p.strip()]
    aspects = set()
    sentiments = []
    aspect_polarities = {}

    for part in parts:
        if "_" in part:
            aspect_key, polarity = part.rsplit("_", 1)
            aspect_key_clean = aspect_key.strip().lower()
            canonical = ASPECT_MAPPING.get(aspect_key_clean, aspect_key.replace("_", " ").title())
            aspects.add(canonical)
            pol = polarity.lower().strip()
            sentiments.append(pol)
            aspect_polarities[canonical] = pol

    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")

    if neg_count > 0 and pos_count == 0:
        overall = "Negative"
    elif pos_count > 0 and neg_count == 0:
        overall = "Positive"
    elif pos_count > neg_count:
        overall = "Positive"
    elif neg_count > pos_count:
        overall = "Negative"
    else:
        overall = "Neutral"

    return {
        "aspects": sorted(list(aspects)),
        "aspect_polarities": aspect_polarities,
        "overall_sentiment": overall,
    }


def load_data() -> pd.DataFrame:
    """Load and prepare the annotated Bangla review dataset."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_FILE}")

    df = pd.read_csv(PROCESSED_FILE, index_col=False)
    # Remove any Unnamed columns and drop duplicates
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df = df.drop_duplicates(subset=["review_id"]).copy()

    # Ensure clean text preserves negations (re-clean if original_text exists)
    if "original_text" in df.columns:
        df["cleaned_text"] = df["original_text"].apply(clean_text)
    else:
        df["cleaned_text"] = df["cleaned_text"].fillna("").apply(clean_text)

    # Ensure aspects is a list of canonical names
    if "aspects" not in df.columns:
        if "aspects_str" in df.columns:
            df["aspects"] = df["aspects_str"].fillna("").apply(
                lambda s: [a.strip() for a in str(s).split(";") if a.strip()]
            )
        elif "label" in df.columns:
            parsed = [parse_absa_labels(lbl) for lbl in df["label"]]
            df["aspects"] = [p["aspects"] for p in parsed]

    # Parse aspect_polarities dictionary
    if "aspect_polarities" not in df.columns and "label" in df.columns:
        df["aspect_polarities"] = [parse_absa_labels(lbl)["aspect_polarities"] for lbl in df["label"]]
    elif "aspect_polarities" in df.columns and len(df) > 0 and isinstance(df["aspect_polarities"].iloc[0], str):
        df["aspect_polarities"] = df["aspect_polarities"].apply(
            lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("{") else parse_absa_labels(str(s))["aspect_polarities"]
        )

    # Filter out empty texts
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df


# Backward-compatible alias
load_annotated_data = load_data


def get_sentiment_split(
    df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split for 3-class sentiment analysis."""
    valid = df.dropna(subset=["sentiment", "cleaned_text"]).copy()
    return train_test_split(
        valid["cleaned_text"],
        valid["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=valid["sentiment"],
    )


def get_aspect_split(
    df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, List[List[str]], List[List[str]]]:
    """80/20 train/test split for multi-label aspect detection."""
    valid = df.dropna(subset=["cleaned_text"]).copy()
    y_aspects = valid["aspects"].tolist()
    indices = np.arange(len(valid))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return (
        valid["cleaned_text"].iloc[train_idx],
        valid["cleaned_text"].iloc[test_idx],
        [y_aspects[i] for i in train_idx],
        [y_aspects[i] for i in test_idx],
    )


def get_polarity_split(
    df: pd.DataFrame,
    aspect: str
) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """Binary (Positive vs Negative) split for a specific aspect."""
    rows = []
    for _, row in df.iterrows():
        p_dict = row.get("aspect_polarities", {})
        if isinstance(p_dict, dict) and aspect in p_dict:
            pol = str(p_dict[aspect]).lower()
            if pol in ("positive", "negative"):
                rows.append({"text": row["cleaned_text"], "polarity": pol.title()})

    sub_df = pd.DataFrame(rows).dropna()
    if len(sub_df) < ASPECT_MIN_SAMPLES:
        return None

    counts = sub_df["polarity"].value_counts().to_dict()
    min_class_count = min(counts.values()) if counts else 0
    stratify = sub_df["polarity"] if min_class_count >= 2 else None

    return train_test_split(
        sub_df["text"],
        sub_df["polarity"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


# Backward-compatible alias
get_aspect_polarity_splits = get_polarity_split
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
    TFIDF_WORD_NGRAMS,
)
from src.preprocessing import clean_text


# -----------------------------------------------------------------------------
# 1. FEATURE & EVALUATION HELPERS
# -----------------------------------------------------------------------------
def build_tfidf() -> FeatureUnion:
    """Build word + character n-gram TF-IDF FeatureUnion."""
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


# Backward-compatible alias
build_tfidf_union = build_tfidf


def _evaluate(y_true: Any, y_pred: Any, multilabel: bool = False) -> Dict[str, Any]:
    """Standardized evaluation for single-label and multi-label tasks."""
    if multilabel:
        return {
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "predictions": y_pred,
            "report": classification_report(
                y_true, y_pred, target_names=ALL_ASPECTS, output_dict=True, zero_division=0
            ),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "predictions": y_pred,
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


# -----------------------------------------------------------------------------
# 2. MODEL TRAINING ROUTINES
# -----------------------------------------------------------------------------
def train_sentiment(
    X_tr: Any,
    y_tr: Any,
    X_te: Any,
    y_te: Any,
    use_bert: bool = False
) -> Tuple[LogisticRegression, Optional[FeatureUnion], Dict[str, Any]]:
    """Train 3-class sentiment classifier using TF-IDF or BanglaBERT embeddings."""
    if use_bert:
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        return model, None, _evaluate(y_te, preds, multilabel=False)

    vec = build_tfidf()
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
    )
    model.fit(X_tr_vec, y_tr)
    preds = model.predict(X_te_vec)
    return model, vec, _evaluate(y_te, preds, multilabel=False)


def train_aspects(
    X_tr: Any,
    y_tr: List[List[str]],
    X_te: Any,
    y_te: List[List[str]],
    use_bert: bool = False
) -> Tuple[OneVsRestClassifier, Optional[FeatureUnion], MultiLabelBinarizer, Dict[str, Any]]:
    """Train multi-label aspect classifier using TF-IDF or BanglaBERT embeddings."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_tr_bin = mlb.fit_transform(y_tr)
    y_te_bin = mlb.transform(y_te)

    base = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
    )
    ovr = OneVsRestClassifier(base)

    if use_bert:
        ovr.fit(X_tr, y_tr_bin)
        preds = ovr.predict(X_te)
        return ovr, None, mlb, _evaluate(y_te_bin, preds, multilabel=True)

    vec = build_tfidf()
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)
    ovr.fit(X_tr_vec, y_tr_bin)
    preds = ovr.predict(X_te_vec)
    return ovr, vec, mlb, _evaluate(y_te_bin, preds, multilabel=True)


def train_polarities(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Train dedicated binary (Positive vs Negative) classifiers per aspect."""
    from src.data_processing import get_polarity_split

    results: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        split = get_polarity_split(df, aspect)
        if split is None:
            continue

        X_tr, X_te, y_tr, y_te = split
        vec = build_tfidf()
        X_tr_vec = vec.fit_transform(X_tr)
        X_te_vec = vec.transform(X_te)

        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
        )
        model.fit(X_tr_vec, y_tr)
        preds = model.predict(X_te_vec)

        results[aspect] = {
            "model": model,
            "vectorizer": vec,
            "metrics": _evaluate(y_te, preds, multilabel=False),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "classes": model.classes_.tolist(),
        }

    return results


# Backward-compatible alias
train_aspect_polarity_tfidf = train_polarities


# -----------------------------------------------------------------------------
# 3. INFERENCE ROUTINES
# -----------------------------------------------------------------------------
def predict_sentiment(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict 3-class sentiment with confidence and class probabilities."""
    if isinstance(text_or_features, str):
        cleaned = clean_text(text_or_features)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
            }
        if use_bert:
            from src.embeddings import get_bert_features
            feat = get_bert_features([cleaned])
        else:
            if vectorizer is None:
                return {"sentiment": "Neutral", "confidence": 0.0, "probabilities": {}}
            feat = vectorizer.transform([cleaned])
    else:
        feat = text_or_features

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
    """Predict multi-label aspect categories present in text."""
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    if isinstance(text_or_features, str):
        cleaned = clean_text(text_or_features)
        if not cleaned:
            return {"aspects": [], "confidences": {asp: 0.0 for asp in ALL_ASPECTS}}
        if use_bert:
            from src.embeddings import get_bert_features
            feat = get_bert_features([cleaned])
        else:
            if vectorizer is None:
                return {"aspects": [], "confidences": {asp: 0.0 for asp in ALL_ASPECTS}}
            feat = vectorizer.transform([cleaned])
    else:
        feat = text_or_features

    preds_bin = model.predict(feat)
    detected = list(binarizer.inverse_transform(preds_bin)[0])

    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        confidences = {str(asp): float(probs[i]) for i, asp in enumerate(binarizer.classes_)}
    else:
        confidences = {asp: (1.0 if asp in detected else 0.0) for asp in ALL_ASPECTS}

    return {"aspects": detected, "confidences": confidences}


def predict_aspect_polarity(
    text: str,
    aspect: str,
    polarity_models: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """Predict binary (Positive vs Negative) polarity for a specific aspect."""
    if not text or not text.strip() or aspect not in polarity_models:
        return {
            "aspect": aspect,
            "polarity": "Positive",
            "confidence": 0.50,
            "is_low_confidence": True,
            "probabilities": {"Positive": 0.5, "Negative": 0.5},
        }

    entry = polarity_models[aspect]
    model, vectorizer = entry["model"], entry.get("vectorizer")
    cleaned = clean_text(text)
    if not cleaned or vectorizer is None:
        return {
            "aspect": aspect,
            "polarity": "Positive",
            "confidence": 0.50,
            "is_low_confidence": True,
            "probabilities": {"Positive": 0.5, "Negative": 0.5},
        }

    feat = vectorizer.transform([cleaned])
    pred = str(model.predict(feat)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        probs_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs_dict = {pred: 1.0}

    return {
        "aspect": aspect,
        "polarity": pred,
        "confidence": confidence,
        "is_low_confidence": confidence < min_confidence,
        "probabilities": probs_dict,
    }


def predict_hierarchical(
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
            "bg_color": bg_color,
        })

    return {
        "aspects": aspect_res["aspects"],
        "aspect_details": aspect_details,
        "confidences": aspect_res["confidences"],
    }


# Backward-compatible alias
predict_aspects_with_polarity = predict_hierarchical


# -----------------------------------------------------------------------------
# 4. PERSISTENCE ROUTINES
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


# Backward-compatible alias
save_polarities = save_polarity_artifacts


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


# Backward-compatible alias
load_polarities = load_polarity_artifacts
```

### 4.7 `src/evaluation.py`
```python
import json
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix

from src.config import RESULTS_DIR


def save_confusion_matrix(
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
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=True, linewidths=0.5
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# Backward-compatible alias
plot_and_save_confusion_matrix = save_confusion_matrix


def save_metrics_summary_json(
    summary_dict: Dict[str, Any],
    filename: str = "metrics_summary.json"
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


# Backward-compatible alias
save_json = save_metrics_summary_json


def save_model_comparison_csv(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    filename: str = "model_comparison.csv"
) -> pd.DataFrame:
    """Save benchmark summary rows to CSV in results/."""
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
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))

from src.config import (
    MODELS_CACHE,
    SENTIMENT_LABELS,
    ensure_dirs,
)
from src.data_processing import (
    get_aspect_split,
    get_sentiment_split,
    load_data,
)
from src.embeddings import get_or_cache_bert_features
from src.evaluation import (
    save_confusion_matrix,
    save_metrics_summary_json,
    save_model_comparison_csv,
)
from src.models import (
    save_artifacts,
    save_polarity_artifacts,
    train_aspects,
    train_polarities,
    train_sentiment,
)


def main() -> None:
    start_time = time.time()
    print("=" * 65)
    print("   BANGLA DARAZ ABSA - UNIFIED TRAINING & BENCHMARK PIPELINE   ")
    print("=" * 65)

    ensure_dirs()

    # -------------------------------------------------------------------------
    # 1. LOAD DATASET & SPLITS
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Loading dataset & generating stratified splits...")
    df = load_data()
    Xs_tr, Xs_te, ys_tr, ys_te = get_sentiment_split(df)
    Xa_tr, Xa_te, ya_tr, ya_te = get_aspect_split(df)
    print(f"  ✓ Loaded {len(df):,} reviews | Train: {len(Xs_tr):,} | Test: {len(Xs_te):,}")

    # -------------------------------------------------------------------------
    # 2. TF-IDF MODELS
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Training TF-IDF models (Sentiment, Aspects, Polarities)...")
    s_m_tf, s_v_tf, s_mt_tf = train_sentiment(Xs_tr, ys_tr, Xs_te, ys_te, use_bert=False)
    save_artifacts("sentiment", s_m_tf, s_v_tf, model_type="tfidf")
    save_confusion_matrix(
        ys_te, s_mt_tf["predictions"], SENTIMENT_LABELS,
        "Sentiment Confusion Matrix (TF-IDF)", "sentiment_tfidf.png"
    )
    print(f"  ✓ [TF-IDF] Sentiment -> Acc: {s_mt_tf['accuracy']:.4f} | Macro-F1: {s_mt_tf['macro_f1']:.4f} | Weighted-F1: {s_mt_tf['weighted_f1']:.4f}")

    a_m_tf, a_v_tf, a_b_tf, a_mt_tf = train_aspects(Xa_tr, ya_tr, Xa_te, ya_te, use_bert=False)
    save_artifacts("issue", a_m_tf, a_v_tf, a_b_tf, model_type="tfidf")
    print(f"  ✓ [TF-IDF] Aspects   -> Micro-F1: {a_mt_tf['micro_f1']:.4f} | Macro-F1: {a_mt_tf['macro_f1']:.4f} | Hamming Loss: {a_mt_tf['hamming_loss']:.4f}")

    pol_models_tf = train_polarities(df)
    save_polarity_artifacts(pol_models_tf, model_type="tfidf")
    for asp, p_entry in pol_models_tf.items():
        m = p_entry["metrics"]
        print(f"  ✓ [TF-IDF] {asp:15s} Polarity -> Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 3. BANGLABERT MODELS
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Extracting BanglaBERT embeddings & training classifiers...")
    s_tr_bert = get_or_cache_bert_features(Xs_tr, MODELS_CACHE / "sentiment_train.npy")
    s_te_bert = get_or_cache_bert_features(Xs_te, MODELS_CACHE / "sentiment_test.npy")
    a_tr_bert = get_or_cache_bert_features(Xa_tr, MODELS_CACHE / "issue_train.npy")
    a_te_bert = get_or_cache_bert_features(Xa_te, MODELS_CACHE / "issue_test.npy")

    s_m_bt, _, s_mt_bt = train_sentiment(s_tr_bert, ys_tr, s_te_bert, ys_te, use_bert=True)
    save_artifacts("sentiment", s_m_bt, model_type="bert")
    save_confusion_matrix(
        ys_te, s_mt_bt["predictions"], SENTIMENT_LABELS,
        "Sentiment Confusion Matrix (BanglaBERT)", "sentiment_bert.png"
    )
    print(f"  ✓ [BanglaBERT] Sentiment -> Acc: {s_mt_bt['accuracy']:.4f} | Macro-F1: {s_mt_bt['macro_f1']:.4f} | Weighted-F1: {s_mt_bt['weighted_f1']:.4f}")

    a_m_bt, _, a_b_bt, a_mt_bt = train_aspects(a_tr_bert, ya_tr, a_te_bert, ya_te, use_bert=True)
    save_artifacts("issue", a_m_bt, binarizer=a_b_bt, model_type="bert")
    print(f"  ✓ [BanglaBERT] Aspects   -> Micro-F1: {a_mt_bt['micro_f1']:.4f} | Macro-F1: {a_mt_bt['macro_f1']:.4f} | Hamming Loss: {a_mt_bt['hamming_loss']:.4f}")

    # -------------------------------------------------------------------------
    # 4. EXPORT BENCHMARKS & METRICS
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Exporting model_comparison.csv & metrics_summary.json...")
    comparison_rows: List[Dict[str, Any]] = [
        {
            "Task": "Sentiment Analysis",
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": round(s_mt_tf["accuracy"], 4),
            "Macro F1": round(s_mt_tf["macro_f1"], 4),
            "Weighted F1": round(s_mt_tf["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Sentiment Analysis",
            "Model": "BanglaBERT + Logistic Regression",
            "Accuracy": round(s_mt_bt["accuracy"], 4),
            "Macro F1": round(s_mt_bt["macro_f1"], 4),
            "Weighted F1": round(s_mt_bt["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Aspect Detection",
            "Model": "TF-IDF + OneVsRest LogReg",
            "Accuracy": round(1.0 - a_mt_tf["hamming_loss"], 4),
            "Macro F1": round(a_mt_tf["macro_f1"], 4),
            "Weighted F1": round(a_mt_tf["weighted_f1"], 4),
            "Additional Metric": f"Micro-F1: {a_mt_tf['micro_f1']:.4f} | Hamming Loss: {a_mt_tf['hamming_loss']:.4f}"
        },
        {
            "Task": "Aspect Detection",
            "Model": "BanglaBERT + OneVsRest LogReg",
            "Accuracy": round(1.0 - a_mt_bt["hamming_loss"], 4),
            "Macro F1": round(a_mt_bt["macro_f1"], 4),
            "Weighted F1": round(a_mt_bt["weighted_f1"], 4),
            "Additional Metric": f"Micro-F1: {a_mt_bt['micro_f1']:.4f} | Hamming Loss: {a_mt_bt['hamming_loss']:.4f}"
        }
    ]

    for asp, p_entry in pol_models_tf.items():
        m = p_entry["metrics"]
        comparison_rows.append({
            "Task": f"Polarity: {asp}",
            "Model": "TF-IDF + Balanced Binary LogReg",
            "Accuracy": round(m["accuracy"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "Weighted F1": round(m["weighted_f1"], 4),
            "Additional Metric": f"Samples: {p_entry['n_train']+p_entry['n_test']} (Train:{p_entry['n_train']})"
        })

    save_model_comparison_csv(comparison_rows, "model_comparison.csv")

    metrics_summary = {
        "dataset_statistics": {
            "total_reviews": len(df),
            "language": "bn",
            "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        },
        "sentiment_analysis": {
            "tfidf": {k: v for k, v in s_mt_tf.items() if k != "predictions"},
            "banglabert": {k: v for k, v in s_mt_bt.items() if k != "predictions"},
        },
        "aspect_detection": {
            "tfidf": {k: v for k, v in a_mt_tf.items() if k != "predictions"},
            "banglabert": {k: v for k, v in a_mt_bt.items() if k != "predictions"},
        },
        "aspect_polarities": {
            asp: {
                "accuracy": p_entry["metrics"]["accuracy"],
                "macro_f1": p_entry["metrics"]["macro_f1"],
                "n_train": p_entry["n_train"],
                "n_test": p_entry["n_test"],
            }
            for asp, p_entry in pol_models_tf.items()
        }
    }
    save_metrics_summary_json(metrics_summary, "metrics_summary.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"       ✅ TRAINING PIPELINE COMPLETED IN {elapsed:.1f}s!      ")
    print("=" * 65)


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

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    px = None
    HAS_PLOTLY = False

try:
    import streamlit as st
except ImportError:
    st = None

sys.path.insert(0, os.path.abspath("."))

from src.config import RESULTS_DIR
from src.models import (
    load_artifacts,
    load_polarity_artifacts,
    predict_hierarchical,
    predict_sentiment,
)
from src.preprocessing import clean_text


def load_all_models():
    """Load model artifacts for TF-IDF and BanglaBERT pipelines."""
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
            "polarity_models": p_models_tf,
        }

        s_m_bt, _, _ = load_artifacts("sentiment", "bert")
        a_m_bt, _, a_b_bt = load_artifacts("issue", "bert")
        loaded["bert"] = {
            "sentiment_model": s_m_bt,
            "aspect_model": a_m_bt,
            "aspect_binarizer": a_b_bt or a_b_tf,
            "polarity_models": p_models_tf,
        }
    except Exception as e:
        loaded["status"] = "error"
        loaded["message"] = str(e)
    return loaded


def load_dataset():
    """Load preprocessed ABSA dataset."""
    from src.data_processing import load_data
    try:
        return load_data()
    except Exception:
        return pd.DataFrame()


def load_benchmarks():
    """Load model comparison benchmarks."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path)
    return pd.DataFrame()


# Only execute Streamlit UI flow when running within Streamlit
if st is not None:
    # Page Configuration
    st.set_page_config(
        page_title="Bangla Daraz ABSA Analytics",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
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
            padding: 10px 14px;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .aspect-name { font-weight: 600; color: #1E293B; font-size: 0.95rem; }
        .aspect-pill { font-weight: 600; font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    load_models_cached = st.cache_resource(load_all_models)
    load_data_cached = st.cache_data(load_dataset)
    load_benchmarks_cached = st.cache_data(load_benchmarks)

    models_data = load_models_cached()
    dataset_df = load_data_cached()
    comparison_df = load_benchmarks_cached()

    # Sidebar
    st.sidebar.title("🛒 Daraz ABSA")
    st.sidebar.markdown("**Sentiment & Aspect NLP Platform**")

    view_mode = st.sidebar.radio(
        "Navigation",
        ["🔍 Review Analyzer", "📈 Benchmarks & Data Insights"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Pipeline")
    selected_model = st.sidebar.radio(
        "Select Model Architecture:",
        ["TF-IDF", "BanglaBERT"],
        help="Toggle between TF-IDF (N-gram Union) and BanglaBERT representations."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Corpus**: Mendeley Bangla Daraz ABSA\n\n"
        f"**Total Reviews**: {len(dataset_df):,} rows\n\n"
        f"**Aspects**: 5 Dimensions (Quality, Price, Delivery, Packaging, Seller)"
    )

    # 1. REVIEW ANALYZER
    if view_mode == "🔍 Review Analyzer":
        st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Hierarchical ABSA live inference via <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

        if models_data["status"] != "ready":
            st.error(f"Models missing: {models_data.get('message')}. Please run `python train_models.py` first.")
            st.stop()

        use_bert = (selected_model == "BanglaBERT")
        active = models_data["bert"] if use_bert else models_data["tfidf"]

        presets = {
            "— choose sample preset —": "",
            "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
            "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
            "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
            "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
            "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল।"
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
                            a_res = predict_hierarchical(
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
                            a_res = predict_hierarchical(
                                user_text,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
                                vectorizer=active["aspect_vectorizer"],
                                binarizer=active["aspect_binarizer"],
                                use_bert=False
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
                        if px is not None:
                            fig_s = px.bar(
                                prob_df, x="Percentage", y="Sentiment", orientation="h",
                                color="Sentiment",
                                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
                                text=prob_df["Percentage"].apply(lambda p: f"{p:.1f}%")
                            )
                            fig_s.update_layout(height=160, margin=dict(t=8, b=8, l=8, r=8), showlegend=False, xaxis=dict(range=[0, 100]))
                            st.plotly_chart(fig_s, width="stretch")
                        else:
                            st.bar_chart(prob_df.set_index("Sentiment")["Percentage"])

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

                with st.expander("🔍 Cleaned Tokens"):
                    st.code(cleaned, language="text")

    # 2. BENCHMARKS & DATA INSIGHTS
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

        # Dataset Distribution
        st.subheader("3. Dataset Distribution & Explorer")
        d1, d2 = st.columns([1, 1.3])

        if not dataset_df.empty and "sentiment" in dataset_df.columns:
            with d1:
                sent_counts = dataset_df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                if px is not None:
                    fig_p = px.pie(
                        sent_counts, names="Sentiment", values="Count", hole=0.4,
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}
                    )
                    fig_p.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_p, width="stretch")
                else:
                    st.write(sent_counts)

            with d2:
                aspect_items = []
                for item in dataset_df["aspects_str"].dropna():
                    for a in str(item).split(";"):
                        if a.strip():
                            aspect_items.append(a.strip())
                asp_s = pd.Series(aspect_items).value_counts().reset_index()
                asp_s.columns = ["Aspect", "Mentions"]
                if px is not None:
                    fig_b = px.bar(asp_s, x="Mentions", y="Aspect", orientation="h", color="Mentions", color_continuous_scale="Blues")
                    fig_b.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_b, width="stretch")
                else:
                    st.write(asp_s)

        if not dataset_df.empty:
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

The empirical results from held-out 20% test sets ([`results/model_comparison.csv`](results/model_comparison.csv)):

| Task | Model Architecture | Accuracy | Macro F1 | Weighted F1 | Additional Metrics & Sample Sizes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Sentiment Analysis** | **TF-IDF + Logistic Regression** | **0.9158** | **0.8441** | **0.9182** | 404 test samples (Stratified) |
| **Sentiment Analysis** | BanglaBERT + Logistic Regression | 0.8292 | 0.6966 | 0.8322 | Frozen mean-pooled 768-dim embeddings |
| **Aspect Detection** | **TF-IDF + OneVsRest LogReg** | **0.9668** | **0.7999** | **0.9412** | **Micro-F1: 0.9410** \| **Hamming Loss: 0.0332** |
| **Aspect Detection** | BanglaBERT + OneVsRest LogReg | 0.9297 | 0.6501 | 0.8937 | **Micro-F1: 0.8797** \| **Hamming Loss: 0.0703** |
| **Polarity: Product Quality** | TF-IDF + Balanced Binary LogReg | **0.9753** | **0.9560** | **0.9755** | 1,816 total samples (Train: 1,452) |
| **Polarity: Price** | TF-IDF + Balanced Binary LogReg | **0.9789** | **0.7446** | **0.9738** | 471 total samples (Train: 376) |
| **Polarity: Delivery** | TF-IDF + Balanced Binary LogReg | **0.8947** | **0.8021** | **0.8900** | 282 total samples (Train: 225) |
| **Polarity: Packaging** | TF-IDF + Balanced Binary LogReg | **0.8000** | **0.7205** | **0.7702** | 72 total samples (Train: 57) |
| **Polarity: Seller Service** | TF-IDF + Balanced Binary LogReg | **0.9545** | **0.4884** | **0.9323** | 108 total samples (Train: 86) |

---

## ⚡ 8. Quickstart & Execution Guide

### 8.1 Training All Models & Generating Benchmarks
Run the streamlined training pipeline from the project root:
```bash
python train_models.py
```
This executes:
1. Data loading and 80/20 train/test splitting.
2. TF-IDF sentiment, multi-label aspect, and aspect polarity model training.
3. BanglaBERT feature extraction and head training.
4. Export of `results/model_comparison.csv`, `results/metrics_summary.json`, and confusion matrices.

### 8.2 Launching the Interactive Web Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with the live Hierarchical ABSA inference engine.

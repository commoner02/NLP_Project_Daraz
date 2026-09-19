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

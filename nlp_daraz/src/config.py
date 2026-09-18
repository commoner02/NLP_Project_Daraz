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

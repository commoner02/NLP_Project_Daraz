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

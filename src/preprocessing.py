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

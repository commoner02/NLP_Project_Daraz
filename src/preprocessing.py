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


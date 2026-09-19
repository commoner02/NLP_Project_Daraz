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

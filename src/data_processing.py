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

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

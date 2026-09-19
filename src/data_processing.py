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

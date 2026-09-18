"""
Unified Machine Learning Models and Artifact Management for Bangla Review Analytics.
Aspect-Based Sentiment Analysis: Sentiment (3-class) & Aspect Detection (Multi-Label).
Supports dual feature representations: TF-IDF (Word + Char Union) & BanglaBERT.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss
)

from src.config import (
    ALL_ASPECTS,
    MODELS_BERT,
    MODELS_TFIDF,
    RANDOM_STATE,
    TFIDF_CHAR_NGRAMS,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_WORD_NGRAMS
)
from src.preprocessing import clean_text


# -----------------------------------------------------------------------------
# EVALUATION & FEATURE EXTRACTION HELPERS
# -----------------------------------------------------------------------------
def _evaluate_multiclass(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute standard multi-class classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "predictions": y_pred,
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }


def _evaluate_multilabel(
    y_true_bin: Any,
    y_pred_bin: Any,
    target_names: List[str] = ALL_ASPECTS
) -> Dict[str, Any]:
    """Compute multi-label classification metrics and hamming loss."""
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    return {
        "accuracy": float(1.0 - h_loss),
        "micro_f1": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        "hamming_loss": h_loss,
        "predictions": y_pred_bin,
        "report": classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
    }


def _extract_features(
    text_or_features: Union[str, np.ndarray],
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Tuple[Any, str]:
    """Uniformly extract features from raw Bangla text or precomputed representations."""
    if not isinstance(text_or_features, str):
        return text_or_features, ""

    raw_text = text_or_features
    cleaned = clean_text(raw_text)

    if use_bert:
        from src.embeddings import get_bert_features
        feat = get_bert_features([cleaned if cleaned else "ভালো"])
    else:
        if not cleaned or vectorizer is None:
            return None, ""
        feat = vectorizer.transform([cleaned])

    return feat, cleaned


# -----------------------------------------------------------------------------
# TF-IDF FEATURE BUILDER
# -----------------------------------------------------------------------------
def build_tfidf_union() -> FeatureUnion:
    """Build hybrid FeatureUnion combining word n-grams and character n-grams."""
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[\u0980-\u09FF\w]+",
        ngram_range=TFIDF_WORD_NGRAMS,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=TFIDF_CHAR_NGRAMS,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True
    )
    return FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer)
    ])


# -----------------------------------------------------------------------------
# 1. SENTIMENT ANALYSIS
# -----------------------------------------------------------------------------
def train_sentiment_tfidf(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Train TF-IDF + Balanced Logistic Regression for Sentiment Analysis."""
    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    metrics = _evaluate_multiclass(y_test, preds)
    return model, vectorizer, metrics


def train_sentiment_bert(
    X_train_bert: np.ndarray,
    y_train: pd.Series,
    X_test_bert: np.ndarray,
    y_test: pd.Series
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train Balanced Logistic Regression on frozen BanglaBERT embeddings."""
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    model.fit(X_train_bert, y_train)
    preds = model.predict(X_test_bert)
    metrics = _evaluate_multiclass(y_test, preds)
    return model, metrics


def predict_sentiment(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict sentiment ('Negative', 'Neutral', 'Positive') and class probabilities."""
    feat, _ = _extract_features(text_or_features, vectorizer, use_bert=use_bert)

    if feat is None:
        return {
            "sentiment": "Neutral",
            "confidence": 0.34,
            "probabilities": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}
        }

    pred = str(model.predict(feat)[0])
    probs_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        for cls, prob in zip(model.classes_, probs):
            probs_dict[str(cls)] = float(prob)
        confidence = float(np.max(probs))
    else:
        confidence = 1.0

    return {
        "sentiment": pred,
        "confidence": confidence,
        "probabilities": probs_dict
    }


# -----------------------------------------------------------------------------
# 2. ASPECT DETECTION (MULTI-LABEL)
# -----------------------------------------------------------------------------
def train_aspect_tfidf(
    X_train: pd.Series,
    y_train: List[List[str]],
    X_test: pd.Series,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, FeatureUnion, MultiLabelBinarizer, Dict[str, Any]]:
    """Train OneVsRestClassifier for multi-label aspect detection with TF-IDF."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_vec, y_train_bin)

    preds_bin = ovr_model.predict(X_test_vec)
    metrics = _evaluate_multilabel(y_test_bin, preds_bin)
    return ovr_model, vectorizer, mlb, metrics


def train_aspect_bert(
    X_train_bert: np.ndarray,
    y_train: List[List[str]],
    X_test_bert: np.ndarray,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, MultiLabelBinarizer, Dict[str, Any]]:
    """Train OneVsRestClassifier for multi-label aspect detection on BanglaBERT features."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    base_lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_bert, y_train_bin)

    preds_bin = ovr_model.predict(X_test_bert)
    metrics = _evaluate_multilabel(y_test_bin, preds_bin)
    return ovr_model, mlb, metrics


# Aliases for backward compatibility
train_issue_tfidf = train_aspect_tfidf
train_issue_bert = train_aspect_bert


def predict_aspects(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """
    Predict multi-label aspect tags and confidence scores purely using Machine Learning.
    Returns: {"aspects": [...], "confidences": {...}}
    """
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    feat, _ = _extract_features(text_or_features, vectorizer, use_bert=use_bert)

    if feat is None:
        return {
            "aspects": ["Product Quality"],
            "confidences": {"Product Quality": 0.5}
        }

    preds_bin = model.predict(feat)
    detected = list(binarizer.inverse_transform(preds_bin)[0])

    if len(detected) == 0:
        detected = ["Product Quality"]

    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        for i, asp in enumerate(binarizer.classes_):
            confidences[str(asp)] = float(probs[i])
    else:
        for asp in ALL_ASPECTS:
            confidences[asp] = 1.0 if asp in detected else 0.0

    return {
        "aspects": detected,
        "confidences": confidences
    }


# Alias for backward compatibility
predict_issue = predict_aspects


# -----------------------------------------------------------------------------
# 3. ARTIFACT SAVE & LOAD
# -----------------------------------------------------------------------------
def save_artifacts(
    task: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    model_type: str = "tfidf"
) -> None:
    """Save trained model artifacts to models/{model_type}/ directory."""
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
    """Load model artifacts from models/{model_type}/ directory."""
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

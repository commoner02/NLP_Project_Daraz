from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss

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
# 1. EVALUATION & FEATURE HELPERS
# -----------------------------------------------------------------------------
def _evaluate_classification(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Compute accuracy, macro/weighted F1 scores, and classification report."""
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
    """Compute multi-label F1 scores, hamming loss, and classification report."""
    h_loss = hamming_loss(y_true_bin, y_pred_bin)
    return {
        "accuracy": float(1.0 - h_loss),
        "micro_f1": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        "hamming_loss": h_loss,
        "predictions": y_pred_bin,
        "report": classification_report(
            y_true_bin, y_pred_bin, target_names=target_names, output_dict=True, zero_division=0
        )
    }


def _extract_features(
    text_or_features: Union[str, np.ndarray],
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Tuple[Any, str]:
    """Extract features from raw Bangla text or precomputed representations."""
    if not isinstance(text_or_features, str):
        return text_or_features, ""

    cleaned = clean_text(text_or_features)
    if use_bert:
        from src.embeddings import get_bert_features
        feat = get_bert_features([cleaned if cleaned else "ভালো"])
    else:
        if not cleaned or vectorizer is None:
            return None, ""
        feat = vectorizer.transform([cleaned])

    return feat, cleaned


def build_tfidf_union() -> FeatureUnion:
    """Build word + character n-gram TF-IDF FeatureUnion."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[\u0980-\u09FF\w]+",
            ngram_range=TFIDF_WORD_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_CHAR_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True
        ))
    ])


# -----------------------------------------------------------------------------
# 2. MODEL TRAINING ROUTINES
# -----------------------------------------------------------------------------
def train_sentiment_tfidf(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Train TF-IDF + Logistic Regression for 3-class sentiment analysis."""
    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    return model, vectorizer, _evaluate_classification(y_test, preds)


def train_sentiment_bert(
    X_train_bert: np.ndarray,
    y_train: pd.Series,
    X_test_bert: np.ndarray,
    y_test: pd.Series
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Train Logistic Regression on frozen BanglaBERT embeddings for sentiment analysis."""
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(X_train_bert, y_train)
    preds = model.predict(X_test_bert)
    return model, _evaluate_classification(y_test, preds)


def train_aspect_tfidf(
    X_train: pd.Series,
    y_train: List[List[str]],
    X_test: pd.Series,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, FeatureUnion, MultiLabelBinarizer, Dict[str, Any]]:
    """Train TF-IDF + OneVsRest Logistic Regression for multi-label aspect detection."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    vectorizer = build_tfidf_union()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_vec, y_train_bin)

    preds_bin = ovr_model.predict(X_test_vec)
    return ovr_model, vectorizer, mlb, _evaluate_multilabel(y_test_bin, preds_bin)


def train_aspect_bert(
    X_train_bert: np.ndarray,
    y_train: List[List[str]],
    X_test_bert: np.ndarray,
    y_test: List[List[str]]
) -> Tuple[OneVsRestClassifier, MultiLabelBinarizer, Dict[str, Any]]:
    """Train BanglaBERT + OneVsRest Logistic Regression for multi-label aspect detection."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)

    base_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
    ovr_model = OneVsRestClassifier(base_lr)
    ovr_model.fit(X_train_bert, y_train_bin)

    preds_bin = ovr_model.predict(X_test_bert)
    return ovr_model, mlb, _evaluate_multilabel(y_test_bin, preds_bin)


def train_aspect_polarity_tfidf(
    anno_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Train dedicated binary (Positive vs. Negative) classifiers for each aspect."""
    from src.data_processing import get_aspect_polarity_splits

    results: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        split = get_aspect_polarity_splits(anno_df, aspect)
        if split is None:
            continue

        X_train, X_test, y_train, y_test = split
        vectorizer = build_tfidf_union()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        results[aspect] = {
            "model": model,
            "vectorizer": vectorizer,
            "metrics": _evaluate_classification(y_test, preds),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": model.classes_.tolist()
        }

    return results


# -----------------------------------------------------------------------------
# 3. INFERENCE ROUTINES
# -----------------------------------------------------------------------------
def predict_sentiment(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict 3-class sentiment ('Negative', 'Neutral', 'Positive') and confidence."""
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
        probs_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs_dict = {pred: 1.0}

    return {"sentiment": pred, "confidence": confidence, "probabilities": probs_dict}


def predict_aspects(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict present aspect categories using multi-label classification."""
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    feat, _ = _extract_features(text_or_features, vectorizer, use_bert=use_bert)
    if feat is None:
        return {"aspects": [], "confidences": {asp: 0.0 for asp in ALL_ASPECTS}}

    preds_bin = model.predict(feat)
    detected = list(binarizer.inverse_transform(preds_bin)[0])

    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        confidences = {str(asp): float(probs[i]) for i, asp in enumerate(binarizer.classes_)}
    else:
        confidences = {asp: (1.0 if asp in detected else 0.0) for asp in ALL_ASPECTS}

    return {"aspects": detected, "confidences": confidences}


def _polarity_fallback(aspect: str) -> Dict[str, Any]:
    """Default fallback dictionary for missing aspect polarity models."""
    return {
        "aspect": aspect,
        "polarity": "Positive",
        "confidence": 0.50,
        "is_low_confidence": True,
        "probabilities": {"Positive": 0.5, "Negative": 0.5}
    }


def predict_aspect_polarity(
    text: str,
    aspect: str,
    polarity_models: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """Predict binary polarity (Positive / Negative) with confidence and low-confidence flag."""
    if not text or not text.strip() or aspect not in polarity_models:
        return _polarity_fallback(aspect)

    entry = polarity_models[aspect]
    model, vectorizer = entry["model"], entry.get("vectorizer")
    cleaned = clean_text(text)
    if not cleaned or vectorizer is None:
        return _polarity_fallback(aspect)

    feat = vectorizer.transform([cleaned])
    pred = str(model.predict(feat)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        probs_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs_dict = {pred: 1.0}

    is_low_confidence = confidence < min_confidence
    return {
        "aspect": aspect,
        "polarity": pred,
        "confidence": confidence,
        "is_low_confidence": is_low_confidence,
        "probabilities": probs_dict
    }


def predict_aspects_with_polarity(
    text_or_features: Union[str, np.ndarray],
    aspect_model: Any,
    polarity_models: Dict[str, Dict[str, Any]],
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    use_bert: bool = False,
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """Hierarchical ABSA: Detect aspects, then predict specific binary polarity per aspect."""
    aspect_res = predict_aspects(
        text_or_features, aspect_model, vectorizer=vectorizer, binarizer=binarizer, use_bert=use_bert
    )
    raw_text = text_or_features if isinstance(text_or_features, str) else ""
    aspect_details = []

    for asp in aspect_res["aspects"]:
        if raw_text and asp in polarity_models:
            pol_info = predict_aspect_polarity(raw_text, asp, polarity_models, min_confidence=min_confidence)
            pol_lbl = pol_info["polarity"]
            conf = pol_info["confidence"]
            is_low = pol_info["is_low_confidence"]
        else:
            pol_lbl = "Positive"
            conf = 0.70
            is_low = False

        if is_low:
            icon = "⚠️"
            color = "#D97706"
            bg_color = "#FFFBEB"
        elif pol_lbl == "Positive":
            icon = "✅"
            color = "#10B981"
            bg_color = "#ECFDF5"
        else:
            icon = "😡"
            color = "#EF4444"
            bg_color = "#FEF2F2"

        aspect_details.append({
            "aspect": asp,
            "polarity": pol_lbl,
            "confidence": conf,
            "is_low_confidence": is_low,
            "icon": icon,
            "color": color,
            "bg_color": bg_color
        })

    return {
        "aspects": aspect_res["aspects"],
        "aspect_details": aspect_details,
        "confidences": aspect_res["confidences"]
    }


# -----------------------------------------------------------------------------
# 4. ARTIFACT PERSISTENCE
# -----------------------------------------------------------------------------
def save_artifacts(
    task: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    model_type: str = "tfidf"
) -> None:
    """Save trained model artifacts to models/{model_type}/."""
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
    """Load model artifacts from models/{model_type}/."""
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


def save_polarity_artifacts(
    polarity_models: Dict[str, Dict[str, Any]],
    model_type: str = "tfidf"
) -> None:
    """Save aspect-specific polarity models and vectorizers."""
    target_dir = MODELS_TFIDF if model_type == "tfidf" else MODELS_BERT
    target_dir.mkdir(parents=True, exist_ok=True)
    for aspect, entry in polarity_models.items():
        safe_key = aspect.lower().replace(" ", "_")
        joblib.dump(entry["model"], target_dir / f"polarity_{safe_key}_model.pkl")
        if "vectorizer" in entry and entry["vectorizer"] is not None:
            joblib.dump(entry["vectorizer"], target_dir / f"polarity_{safe_key}_vectorizer.pkl")


def load_polarity_artifacts(
    model_type: str = "tfidf"
) -> Dict[str, Dict[str, Any]]:
    """Load all saved aspect-specific polarity models."""
    target_dir = MODELS_TFIDF if model_type == "tfidf" else MODELS_BERT
    polarity_models: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        safe_key = aspect.lower().replace(" ", "_")
        model_path = target_dir / f"polarity_{safe_key}_model.pkl"
        vec_path = target_dir / f"polarity_{safe_key}_vectorizer.pkl"
        if model_path.exists():
            entry: Dict[str, Any] = {"model": joblib.load(model_path)}
            if vec_path.exists():
                entry["vectorizer"] = joblib.load(vec_path)
            polarity_models[aspect] = entry
    return polarity_models

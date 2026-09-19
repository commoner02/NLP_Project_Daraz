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
    TFIDF_WORD_NGRAMS,
)
from src.preprocessing import clean_text


# -----------------------------------------------------------------------------
# 1. FEATURE & EVALUATION HELPERS
# -----------------------------------------------------------------------------
def build_tfidf() -> FeatureUnion:
    """Build word + character n-gram TF-IDF FeatureUnion."""
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[\u0980-\u09FFA-Za-z0-9]+",
            ngram_range=TFIDF_WORD_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_CHAR_NGRAMS,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
        )),
    ])


# Backward-compatible alias
build_tfidf_union = build_tfidf


def _evaluate(y_true: Any, y_pred: Any, multilabel: bool = False) -> Dict[str, Any]:
    """Standardized evaluation for single-label and multi-label tasks."""
    if multilabel:
        return {
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "predictions": y_pred,
            "report": classification_report(
                y_true, y_pred, target_names=ALL_ASPECTS, output_dict=True, zero_division=0
            ),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "predictions": y_pred,
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


# -----------------------------------------------------------------------------
# 2. MODEL TRAINING ROUTINES
# -----------------------------------------------------------------------------
def train_sentiment(
    X_tr: Any,
    y_tr: Any,
    X_te: Any,
    y_te: Any,
    use_bert: bool = False
) -> Tuple[LogisticRegression, Optional[FeatureUnion], Dict[str, Any]]:
    """Train 3-class sentiment classifier using TF-IDF or BanglaBERT embeddings."""
    if use_bert:
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        return model, None, _evaluate(y_te, preds, multilabel=False)

    vec = build_tfidf()
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
    )
    model.fit(X_tr_vec, y_tr)
    preds = model.predict(X_te_vec)
    return model, vec, _evaluate(y_te, preds, multilabel=False)


def train_aspects(
    X_tr: Any,
    y_tr: List[List[str]],
    X_te: Any,
    y_te: List[List[str]],
    use_bert: bool = False
) -> Tuple[OneVsRestClassifier, Optional[FeatureUnion], MultiLabelBinarizer, Dict[str, Any]]:
    """Train multi-label aspect classifier using TF-IDF or BanglaBERT embeddings."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_tr_bin = mlb.fit_transform(y_tr)
    y_te_bin = mlb.transform(y_te)

    base = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
    )
    ovr = OneVsRestClassifier(base)

    if use_bert:
        ovr.fit(X_tr, y_tr_bin)
        preds = ovr.predict(X_te)
        return ovr, None, mlb, _evaluate(y_te_bin, preds, multilabel=True)

    vec = build_tfidf()
    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)
    ovr.fit(X_tr_vec, y_tr_bin)
    preds = ovr.predict(X_te_vec)
    return ovr, vec, mlb, _evaluate(y_te_bin, preds, multilabel=True)


def train_polarities(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Train dedicated binary (Positive vs Negative) classifiers per aspect."""
    from src.data_processing import get_polarity_split

    results: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        split = get_polarity_split(df, aspect)
        if split is None:
            continue

        X_tr, X_te, y_tr, y_te = split
        vec = build_tfidf()
        X_tr_vec = vec.fit_transform(X_tr)
        X_te_vec = vec.transform(X_te)

        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
        )
        model.fit(X_tr_vec, y_tr)
        preds = model.predict(X_te_vec)

        results[aspect] = {
            "model": model,
            "vectorizer": vec,
            "metrics": _evaluate(y_te, preds, multilabel=False),
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "classes": model.classes_.tolist(),
        }

    return results


# Backward-compatible alias
train_aspect_polarity_tfidf = train_polarities


# -----------------------------------------------------------------------------
# 3. INFERENCE ROUTINES
# -----------------------------------------------------------------------------
def predict_sentiment(
    text_or_features: Union[str, np.ndarray],
    model: Any,
    vectorizer: Optional[Any] = None,
    use_bert: bool = False
) -> Dict[str, Any]:
    """Predict 3-class sentiment with confidence and class probabilities."""
    if isinstance(text_or_features, str):
        cleaned = clean_text(text_or_features)
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0},
            }
        if use_bert:
            from src.embeddings import get_bert_features
            feat = get_bert_features([cleaned])
        else:
            if vectorizer is None:
                return {"sentiment": "Neutral", "confidence": 0.0, "probabilities": {}}
            feat = vectorizer.transform([cleaned])
    else:
        feat = text_or_features

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
    """Predict multi-label aspect categories present in text."""
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    if isinstance(text_or_features, str):
        cleaned = clean_text(text_or_features)
        if not cleaned:
            return {"aspects": [], "confidences": {asp: 0.0 for asp in ALL_ASPECTS}}
        if use_bert:
            from src.embeddings import get_bert_features
            feat = get_bert_features([cleaned])
        else:
            if vectorizer is None:
                return {"aspects": [], "confidences": {asp: 0.0 for asp in ALL_ASPECTS}}
            feat = vectorizer.transform([cleaned])
    else:
        feat = text_or_features

    preds_bin = model.predict(feat)
    detected = list(binarizer.inverse_transform(preds_bin)[0])

    confidences = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        confidences = {str(asp): float(probs[i]) for i, asp in enumerate(binarizer.classes_)}
    else:
        confidences = {asp: (1.0 if asp in detected else 0.0) for asp in ALL_ASPECTS}

    return {"aspects": detected, "confidences": confidences}


def predict_aspect_polarity(
    text: str,
    aspect: str,
    polarity_models: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.60
) -> Dict[str, Any]:
    """Predict binary (Positive vs Negative) polarity for a specific aspect."""
    if not text or not text.strip() or aspect not in polarity_models:
        return {
            "aspect": aspect,
            "polarity": "Positive",
            "confidence": 0.50,
            "is_low_confidence": True,
            "probabilities": {"Positive": 0.5, "Negative": 0.5},
        }

    entry = polarity_models[aspect]
    model, vectorizer = entry["model"], entry.get("vectorizer")
    cleaned = clean_text(text)
    if not cleaned or vectorizer is None:
        return {
            "aspect": aspect,
            "polarity": "Positive",
            "confidence": 0.50,
            "is_low_confidence": True,
            "probabilities": {"Positive": 0.5, "Negative": 0.5},
        }

    feat = vectorizer.transform([cleaned])
    pred = str(model.predict(feat)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        probs_dict = {str(c): float(p) for c, p in zip(model.classes_, probs)}
        confidence = float(np.max(probs))
    else:
        confidence = 1.0
        probs_dict = {pred: 1.0}

    return {
        "aspect": aspect,
        "polarity": pred,
        "confidence": confidence,
        "is_low_confidence": confidence < min_confidence,
        "probabilities": probs_dict,
    }


def predict_hierarchical(
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
            "bg_color": bg_color,
        })

    return {
        "aspects": aspect_res["aspects"],
        "aspect_details": aspect_details,
        "confidences": aspect_res["confidences"],
    }


# Backward-compatible alias
predict_aspects_with_polarity = predict_hierarchical


# -----------------------------------------------------------------------------
# 4. PERSISTENCE ROUTINES
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


# Backward-compatible alias
save_polarities = save_polarity_artifacts


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


# Backward-compatible alias
load_polarities = load_polarity_artifacts

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

from src.config import ALL_ASPECTS, LSTM_MAX_LENGTH
from src.features import AspectLSTM, SentimentLSTM
from src.preprocessing import clean_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_sentiment(
    text: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    vocab: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """3-class sentiment prediction. Pass vectorizer for TF-IDF or vocab for LSTM."""
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in SentimentLSTM.CLASSES},
        }

    if vocab is not None:  # LSTM path
        ids = torch.tensor([vocab.transform(cleaned, LSTM_MAX_LENGTH)], dtype=torch.long).to(DEVICE)
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(ids), dim=1)[0].cpu().numpy()
        classes = SentimentLSTM.CLASSES
    else:  # TF-IDF path
        feat = vectorizer.transform([cleaned]) if vectorizer is not None else cleaned
        probs = model.predict_proba(feat)[0]
        classes = model.classes_

    pred = str(classes[int(np.argmax(probs))])
    return {
        "sentiment": pred,
        "confidence": float(max(probs)),
        "probabilities": {str(c): float(p) for c, p in zip(classes, probs)},
    }


def predict_aspects(
    text: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    vocab: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Multi-label aspect detection."""
    cleaned = clean_text(text)
    if binarizer is None:
        binarizer = MultiLabelBinarizer(classes=ALL_ASPECTS)
        binarizer.fit([ALL_ASPECTS])

    if not cleaned:
        return {"aspects": [], "confidences": {a: 0.0 for a in ALL_ASPECTS}}

    if vocab is not None:  # LSTM path
        ids = torch.tensor([vocab.transform(cleaned, LSTM_MAX_LENGTH)], dtype=torch.long).to(DEVICE)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(ids))[0].cpu().numpy()
    else:  # TF-IDF path
        feat = vectorizer.transform([cleaned]) if vectorizer is not None else cleaned
        probs = model.predict_proba(feat)[0]

    detected = [binarizer.classes_[i] for i, p in enumerate(probs) if p > 0.5]
    return {
        "aspects": detected,
        "confidences": {str(a): float(p) for a, p in zip(binarizer.classes_, probs)},
    }


def predict_aspect_polarity(
    text: str,
    aspect: str,
    polarity_models: Dict[str, Dict[str, Any]],
    min_confidence: float = 0.60,
) -> Dict[str, Any]:
    """Binary polarity prediction for a specific aspect."""
    default = {
        "aspect": aspect,
        "polarity": "Positive",
        "confidence": 0.50,
        "is_low_confidence": True,
        "probabilities": {"Positive": 0.5, "Negative": 0.5},
    }

    cleaned = clean_text(text)
    if not cleaned or aspect not in polarity_models:
        return default

    entry = polarity_models[aspect]
    vec = entry.get("vectorizer")
    model = entry["model"]
    if vec is None:
        return default

    feat = vec.transform([cleaned])
    pred = str(model.predict(feat)[0])
    probs = model.predict_proba(feat)[0]
    conf = float(max(probs))

    return {
        "aspect": aspect,
        "polarity": pred,
        "confidence": conf,
        "is_low_confidence": conf < min_confidence,
        "probabilities": {str(c): float(p) for c, p in zip(model.classes_, probs)},
    }


def predict_hierarchical(
    text: str,
    aspect_model: Any,
    polarity_models: Dict[str, Dict[str, Any]],
    vectorizer: Optional[Any] = None,
    binarizer: Optional[MultiLabelBinarizer] = None,
    vocab: Optional[Any] = None,
    min_confidence: float = 0.60,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hierarchical ABSA: Detect aspects → binary polarity per aspect."""
    aspect_res = predict_aspects(
        text, aspect_model, vectorizer=vectorizer, binarizer=binarizer, vocab=vocab
    )
    aspects = aspect_res["aspects"]
    details: List[Dict[str, Any]] = []

    for asp in aspects:
        info = predict_aspect_polarity(text, asp, polarity_models, min_confidence=min_confidence)
        pol, conf, low = info["polarity"], info["confidence"], info["is_low_confidence"]
        if low:
            status, color, bg = "low_confidence", "#D97706", "#FFFBEB"
        elif pol == "Positive":
            status, color, bg = "positive", "#10B981", "#ECFDF5"
        else:
            status, color, bg = "negative", "#EF4444", "#FEF2F2"

        details.append({
            "aspect": asp,
            "polarity": pol,
            "confidence": conf,
            "is_low_confidence": low,
            "status": status,
            "color": color,
            "bg_color": bg,
        })

    return {
        "aspects": aspects,
        "aspect_details": details,
        "confidences": aspect_res["confidences"],
    }

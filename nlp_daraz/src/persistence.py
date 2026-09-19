from typing import Any, Dict, Optional, Tuple
import joblib
import torch

from src.config import ALL_ASPECTS, MODELS_LSTM, MODELS_TFIDF
from src.features import AspectLSTM, SentimentLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DIRS = {"tfidf": MODELS_TFIDF, "lstm": MODELS_LSTM}


def save_artifacts(
    task: str,
    model: Any,
    vectorizer: Optional[Any] = None,
    binarizer: Optional[Any] = None,
    model_type: str = "tfidf",
    vocab: Optional[Any] = None,
) -> None:
    """Save one trained task artifact. task ∈ {'sentiment', 'issue'}."""
    d = _DIRS[model_type]
    d.mkdir(parents=True, exist_ok=True)

    if model_type == "lstm":
        torch.save(model.state_dict(), d / f"{task}_model.pt")
        if vocab is not None:
            joblib.dump(vocab, d / "vocab.pkl")
    else:
        joblib.dump(model, d / f"{task}_model.pkl")
        if vectorizer is not None:
            joblib.dump(vectorizer, d / f"{task}_vectorizer.pkl")

    if binarizer is not None:
        joblib.dump(binarizer, d / f"{task}_binarizer.pkl")


def load_artifacts(
    task: str,
    model_type: str = "tfidf",
) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Any]]:
    """Load one task artifact. Returns (model, vectorizer, binarizer, vocab)."""
    d = _DIRS[model_type]
    bin_path = d / f"{task}_binarizer.pkl"
    binarizer = joblib.load(bin_path) if bin_path.exists() else None

    if model_type == "lstm":
        vocab_path = d / "vocab.pkl"
        model_path = d / f"{task}_model.pt"
        if not vocab_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Missing LSTM artifacts in {d}. Run pipeline.ipynb first.")

        vocab = joblib.load(vocab_path)
        model = (
            SentimentLSTM(vocab.vocab_size)
            if task == "sentiment"
            else AspectLSTM(vocab.vocab_size, len(ALL_ASPECTS))
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model, None, binarizer, vocab

    # TF-IDF
    model_path = d / f"{task}_model.pkl"
    vec_path = d / f"{task}_vectorizer.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing TF-IDF model at {model_path}. Run pipeline.ipynb first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path) if vec_path.exists() else None
    return model, vectorizer, binarizer, None


def save_polarity_artifacts(
    polarity_models: Dict[str, Dict[str, Any]],
    model_type: str = "tfidf",
) -> None:
    """Save aspect-specific polarity models and vectorizers."""
    d = _DIRS[model_type]
    d.mkdir(parents=True, exist_ok=True)
    for aspect, entry in polarity_models.items():
        key = aspect.lower().replace(" ", "_")
        joblib.dump(entry["model"], d / f"polarity_{key}_model.pkl")
        if "vectorizer" in entry and entry["vectorizer"] is not None:
            joblib.dump(entry["vectorizer"], d / f"polarity_{key}_vectorizer.pkl")


def load_polarity_artifacts(
    model_type: str = "tfidf",
) -> Dict[str, Dict[str, Any]]:
    """Load all saved aspect-specific polarity models."""
    d = _DIRS[model_type]
    out: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        key = aspect.lower().replace(" ", "_")
        mp, vp = d / f"polarity_{key}_model.pkl", d / f"polarity_{key}_vectorizer.pkl"
        if mp.exists():
            entry: Dict[str, Any] = {"model": joblib.load(mp)}
            if vp.exists():
                entry["vectorizer"] = joblib.load(vp)
            out[aspect] = entry
    return out

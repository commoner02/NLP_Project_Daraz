"""Model training routines for TF-IDF and PyTorch BiLSTM classifiers."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    ALL_ASPECTS,
    LSTM_EPOCHS,
    LSTM_LR,
    RANDOM_STATE,
)
from src.data_processing import get_polarity_split
from src.evaluate import _evaluate
from src.features import AspectLSTM, SentimentLSTM, build_tfidf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- TF-IDF trainers ----------------------------------------------------------
def _train_logreg_classifier(
    X_tr: Any,
    y_tr: Any,
    X_te: Any,
    y_te: Any,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Helper to train a balanced Logistic Regression classifier on TF-IDF features."""
    vec = build_tfidf()
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=max_iter, random_state=RANDOM_STATE, solver="lbfgs"
    )
    model.fit(vec.fit_transform(X_tr), y_tr)
    return model, vec, _evaluate(y_te, model.predict(vec.transform(X_te)))


def train_sentiment(
    X_tr: Any,
    y_tr: Any,
    X_te: Any,
    y_te: Any,
) -> Tuple[LogisticRegression, FeatureUnion, Dict[str, Any]]:
    """Train 3-class sentiment classifier using TF-IDF."""
    return _train_logreg_classifier(X_tr, y_tr, X_te, y_te)


def train_aspects(
    X_tr: Any,
    y_tr: List[List[str]],
    X_te: Any,
    y_te: List[List[str]],
) -> Tuple[OneVsRestClassifier, FeatureUnion, MultiLabelBinarizer, Dict[str, Any]]:
    """Train multi-label aspect classifier using TF-IDF."""
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    y_tr_bin = mlb.fit_transform(y_tr)
    y_te_bin = mlb.transform(y_te)

    vec = build_tfidf()
    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs"
        )
    )
    model.fit(vec.fit_transform(X_tr), y_tr_bin)
    return model, vec, mlb, _evaluate(y_te_bin, model.predict(vec.transform(X_te)), multilabel=True)


def train_polarities(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Train one binary classifier per aspect. Skips aspects with < MIN samples."""
    out: Dict[str, Dict[str, Any]] = {}
    for aspect in ALL_ASPECTS:
        split = get_polarity_split(df, aspect)
        if split is None:
            print(f"  [Skipped] {aspect}: too few samples (< MIN_POLARITY_SAMPLES)")
            continue
        X_tr, X_te, y_tr, y_te = split
        model, vec, m = _train_logreg_classifier(X_tr, y_tr, X_te, y_te)
        out[aspect] = {
            "model": model,
            "vectorizer": vec,
            "metrics": m,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }
    return out


# -- LSTM trainers ------------------------------------------------------------
def _train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    name: str = "Model",
) -> nn.Module:
    """Shared PyTorch BiLSTM training loop."""
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LSTM_LR)
    model.train()
    print(f"    Training BiLSTM {name}...")
    for epoch in range(1, LSTM_EPOCHS + 1):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y.float() if isinstance(criterion, nn.BCEWithLogitsLoss) else y)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"      Epoch [{epoch:02d}/{LSTM_EPOCHS:02d}] - Loss: {avg_loss:.4f}", flush=True)
    return model


def train_lstm_sentiment(
    train_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    y_te: List[str],
    class_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[SentimentLSTM, Dict[str, Any]]:
    """Train 3-class sentiment classifier using BiLSTM."""
    model = SentimentLSTM(vocab_size)
    _train_lstm(model, train_loader, nn.CrossEntropyLoss(), name="Sentiment Model")

    model.eval()
    idx2class = {i: c for i, c in enumerate(SentimentLSTM.CLASSES)}
    preds: List[str] = []
    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x.to(DEVICE))
            preds.extend([idx2class[i] for i in logits.argmax(1).cpu().numpy().tolist()])
    return model, _evaluate(y_te, preds)


def train_lstm_aspects(
    train_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    y_te_bin: np.ndarray,
    mlb: MultiLabelBinarizer,
) -> Tuple[AspectLSTM, MultiLabelBinarizer, Dict[str, Any]]:
    """Train multi-label aspect classifier using BiLSTM."""
    model = AspectLSTM(vocab_size, len(ALL_ASPECTS))
    _train_lstm(model, train_loader, nn.BCEWithLogitsLoss(), name="Multi-Label Aspect Model")

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            probs = torch.sigmoid(model(x.to(DEVICE)))
            preds.extend((probs > 0.5).int().cpu().numpy())
    return model, mlb, _evaluate(y_te_bin, np.array(preds), multilabel=True)

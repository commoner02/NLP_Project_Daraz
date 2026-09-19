"""Metrics, confusion matrices, and benchmark CSV/JSON export."""
import json
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
)

from src.config import ALL_ASPECTS, RESULTS_DIR


def _evaluate(y_true: Any, y_pred: Any, multilabel: bool = False) -> Dict[str, Any]:
    """Standardized evaluation metrics for single-label and multi-label tasks."""
    if multilabel:
        y_pred_arr = np.array(y_pred)
        return {
            "hamming_loss": float(hamming_loss(y_true, y_pred_arr)),
            "micro_f1": float(f1_score(y_true, y_pred_arr, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred_arr, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred_arr, average="weighted", zero_division=0)),
            "predictions": y_pred_arr,
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "predictions": y_pred,
    }


def save_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: List[Any],
    title: str,
    filename: str,
) -> None:
    """Generate and save Seaborn confusion matrix heatmap."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics_summary_json(
    summary_dict: Dict[str, Any],
    filename: str = "metrics_summary.json",
) -> None:
    """Save summary metrics to JSON in results/ with safe type conversion."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False, default=_convert)


def save_model_comparison_csv(
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    filename: str = "model_comparison.csv",
) -> pd.DataFrame:
    """Save benchmark summary rows to CSV in results/."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df

"""
Evaluation and Metrics Reporting Module for Bangla Review Analytics.
Plots confusion matrices, writes classification reports to CSV, and saves JSON summaries.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import RESULTS_DIR


def plot_and_save_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: List[Any],
    title: str,
    filename: str
) -> None:
    """
    Generate and save a high-resolution Seaborn confusion matrix heatmap directly to results/.
    """
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
        linewidths=0.5
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report_csv(
    report_dict: Dict[str, Any],
    filename: str
) -> None:
    """
    Save a Scikit-learn classification report dictionary as a CSV directly to results/.
    """
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(output_path, index=True)


def save_metrics_summary_json(
    summary_dict: Dict[str, Any],
    filename: str = "metrics_summary.json"
) -> None:
    """Save summary metrics JSON directly to results/."""
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
    data_or_s_tf: Any,
    *args: Any,
    filename: str = "model_comparison.csv"
) -> pd.DataFrame:
    """
    Save model comparison table to results/ directory and return DataFrame.
    Supports:
      1. save_model_comparison_csv(comparison_rows_or_df, filename="model_comparison.csv")
      2. save_model_comparison_csv(s_metrics_tfidf, s_metrics_bert, a_metrics_tfidf, a_metrics_bert)
    """
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data_or_s_tf, pd.DataFrame):
        df = data_or_s_tf
    elif isinstance(data_or_s_tf, list):
        df = pd.DataFrame(data_or_s_tf)
    elif len(args) >= 3:
        s_tf = data_or_s_tf
        s_bt, a_tf, a_bt = args[0], args[1], args[2]
        rows = [
            {
                "Task": "Sentiment Analysis",
                "Model": "TF-IDF + Logistic Regression",
                "Accuracy": round(s_tf.get("accuracy", 0.0), 4),
                "Macro F1": round(s_tf.get("macro_f1", 0.0), 4),
                "Weighted F1": round(s_tf.get("weighted_f1", 0.0), 4),
                "Additional Metric": "N/A"
            },
            {
                "Task": "Sentiment Analysis",
                "Model": "BanglaBERT + Logistic Regression",
                "Accuracy": round(s_bt.get("accuracy", 0.0), 4),
                "Macro F1": round(s_bt.get("macro_f1", 0.0), 4),
                "Weighted F1": round(s_bt.get("weighted_f1", 0.0), 4),
                "Additional Metric": "N/A"
            },
            {
                "Task": "Aspect Detection",
                "Model": "TF-IDF + OneVsRest LogReg",
                "Accuracy": round(a_tf.get("accuracy", 0.0), 4),
                "Macro F1": round(a_tf.get("macro_f1", 0.0), 4),
                "Weighted F1": round(a_tf.get("weighted_f1", 0.0), 4),
                "Additional Metric": f"Hamming Loss: {a_tf.get('hamming_loss', 0.0):.4f}"
            },
            {
                "Task": "Aspect Detection",
                "Model": "BanglaBERT + OneVsRest LogReg",
                "Accuracy": round(a_bt.get("accuracy", 0.0), 4),
                "Macro F1": round(a_bt.get("macro_f1", 0.0), 4),
                "Weighted F1": round(a_bt.get("weighted_f1", 0.0), 4),
                "Additional Metric": f"Hamming Loss: {a_bt.get('hamming_loss', 0.0):.4f}"
            }
        ]
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()

    df.to_csv(output_path, index=False)
    return df



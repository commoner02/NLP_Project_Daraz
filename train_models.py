import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.config import MODELS_CACHE, SENTIMENT_LABELS, ensure_dirs
from src.data_processing import get_aspect_split, get_sentiment_split, load_annotated_data
from src.embeddings import get_or_cache_bert_features
from src.evaluation import (
    plot_and_save_confusion_matrix,
    save_classification_report_csv,
    save_metrics_summary_json,
    save_model_comparison_csv
)
from src.models import (
    save_artifacts,
    save_polarity_artifacts,
    train_aspect_bert,
    train_aspect_polarity_tfidf,
    train_aspect_tfidf,
    train_sentiment_bert,
    train_sentiment_tfidf
)


def main():
    start_time = time.time()
    print("=" * 60)
    print("   BANGLA DARAZ ABSA - TRAINING & BENCHMARKING PIPELINE   ")
    print("=" * 60)

    ensure_dirs()

    # 1. Load Data & Splits
    print("\n[Step 1/4] Loading dataset and creating 80/20 splits...")
    df = load_annotated_data()
    X_train_s, X_test_s, y_train_s, y_test_s = get_sentiment_split(df)
    X_train_a, X_test_a, y_train_a, y_test_a = get_aspect_split(df)
    print(f"  ✓ Loaded {len(df):,} reviews | Sentiment Split: {len(X_train_s)} train, {len(X_test_s)} test")

    # 2. TF-IDF Models
    print("\n[Step 2/4] Training TF-IDF models (Sentiment, Aspects, Polarities)...")
    s_model_tf, s_vec_tf, s_metrics_tf = train_sentiment_tfidf(X_train_s, y_train_s, X_test_s, y_test_s)
    save_artifacts("sentiment", s_model_tf, s_vec_tf, model_type="tfidf")
    plot_and_save_confusion_matrix(
        y_test_s, s_metrics_tf["predictions"],
        labels=SENTIMENT_LABELS,
        title="Sentiment Confusion Matrix (TF-IDF)",
        filename="sentiment_tfidf.png"
    )
    save_classification_report_csv(s_metrics_tf["report"], "sentiment_tfidf.csv")
    print(f"  ✓ [TF-IDF] Sentiment -> Acc: {s_metrics_tf['accuracy']:.4f} | Macro-F1: {s_metrics_tf['macro_f1']:.4f}")

    a_model_tf, a_vec_tf, a_mlb_tf, a_metrics_tf = train_aspect_tfidf(X_train_a, y_train_a, X_test_a, y_test_a)
    save_artifacts("issue", a_model_tf, a_vec_tf, a_mlb_tf, model_type="tfidf")
    save_classification_report_csv(a_metrics_tf["report"], "issue_tfidf.csv")
    print(f"  ✓ [TF-IDF] Aspects   -> Micro-F1: {a_metrics_tf['micro_f1']:.4f} | Hamming Loss: {a_metrics_tf['hamming_loss']:.4f}")

    polarity_models_tf = train_aspect_polarity_tfidf(df)
    save_polarity_artifacts(polarity_models_tf, model_type="tfidf")
    for asp, p_data in polarity_models_tf.items():
        m = p_data["metrics"]
        safe_name = asp.lower().replace(" ", "_")
        save_classification_report_csv(m["report"], f"polarity_{safe_name}_tfidf.csv")
        print(f"  ✓ [TF-IDF] {asp:15s} Polarity -> Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

    # 3. BanglaBERT Models
    print("\n[Step 3/4] Extracting BanglaBERT embeddings & training classifiers...")
    s_train_bert = get_or_cache_bert_features(X_train_s, MODELS_CACHE / "sentiment_train.npy")
    s_test_bert = get_or_cache_bert_features(X_test_s, MODELS_CACHE / "sentiment_test.npy")
    a_train_bert = get_or_cache_bert_features(X_train_a, MODELS_CACHE / "issue_train.npy")
    a_test_bert = get_or_cache_bert_features(X_test_a, MODELS_CACHE / "issue_test.npy")

    s_model_bt, s_metrics_bt = train_sentiment_bert(s_train_bert, y_train_s, s_test_bert, y_test_s)
    save_artifacts("sentiment", s_model_bt, model_type="bert")
    plot_and_save_confusion_matrix(
        y_test_s, s_metrics_bt["predictions"],
        labels=SENTIMENT_LABELS,
        title="Sentiment Confusion Matrix (BanglaBERT)",
        filename="sentiment_bert.png"
    )
    save_classification_report_csv(s_metrics_bt["report"], "sentiment_bert.csv")
    print(f"  ✓ [BanglaBERT] Sentiment -> Acc: {s_metrics_bt['accuracy']:.4f} | Macro-F1: {s_metrics_bt['macro_f1']:.4f}")

    a_model_bt, a_mlb_bt, a_metrics_bt = train_aspect_bert(a_train_bert, y_train_a, a_test_bert, y_test_a)
    save_artifacts("issue", a_model_bt, binarizer=a_mlb_bt, model_type="bert")
    save_classification_report_csv(a_metrics_bt["report"], "issue_bert.csv")
    print(f"  ✓ [BanglaBERT] Aspects   -> Micro-F1: {a_metrics_bt['micro_f1']:.4f} | Hamming Loss: {a_metrics_bt['hamming_loss']:.4f}")

    # 4. Save Benchmark Summaries
    print("\n[Step 4/4] Exporting benchmark tables and metrics...")
    comparison_rows = [
        {
            "Task": "Sentiment Analysis",
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": round(s_metrics_tf["accuracy"], 4),
            "Macro F1": round(s_metrics_tf["macro_f1"], 4),
            "Weighted F1": round(s_metrics_tf["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Sentiment Analysis",
            "Model": "BanglaBERT + Logistic Regression",
            "Accuracy": round(s_metrics_bt["accuracy"], 4),
            "Macro F1": round(s_metrics_bt["macro_f1"], 4),
            "Weighted F1": round(s_metrics_bt["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Aspect Detection",
            "Model": "TF-IDF + OneVsRest LogReg",
            "Accuracy": round(a_metrics_tf["accuracy"], 4),
            "Macro F1": round(a_metrics_tf["macro_f1"], 4),
            "Weighted F1": round(a_metrics_tf["weighted_f1"], 4),
            "Additional Metric": f"Hamming Loss: {a_metrics_tf['hamming_loss']:.4f}"
        },
        {
            "Task": "Aspect Detection",
            "Model": "BanglaBERT + OneVsRest LogReg",
            "Accuracy": round(a_metrics_bt["accuracy"], 4),
            "Macro F1": round(a_metrics_bt["macro_f1"], 4),
            "Weighted F1": round(a_metrics_bt["weighted_f1"], 4),
            "Additional Metric": f"Hamming Loss: {a_metrics_bt['hamming_loss']:.4f}"
        }
    ]

    for asp, p_data in polarity_models_tf.items():
        m = p_data["metrics"]
        comparison_rows.append({
            "Task": f"Polarity: {asp}",
            "Model": "TF-IDF + Balanced Binary LogReg",
            "Accuracy": round(m["accuracy"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "Weighted F1": round(m["weighted_f1"], 4),
            "Additional Metric": f"Samples: {p_data['n_train']+p_data['n_test']} (Train:{p_data['n_train']})"
        })

    save_model_comparison_csv(comparison_rows, "model_comparison.csv")

    metrics_summary = {
        "dataset_statistics": {
            "total_reviews": len(df),
            "language": "bn",
            "sentiment_distribution": df["sentiment"].value_counts().to_dict()
        },
        "sentiment_analysis": {
            "tfidf": s_metrics_tf,
            "banglabert": s_metrics_bt
        },
        "aspect_detection": {
            "tfidf": a_metrics_tf,
            "banglabert": a_metrics_bt
        },
        "aspect_polarities": {
            asp: {
                "accuracy": p_data["metrics"]["accuracy"],
                "macro_f1": p_data["metrics"]["macro_f1"],
                "n_train": p_data["n_train"],
                "n_test": p_data["n_test"]
            }
            for asp, p_data in polarity_models_tf.items()
        }
    }
    save_metrics_summary_json(metrics_summary, "metrics_summary.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"       TRAINING PIPELINE COMPLETED IN {elapsed:.1f}s!      ")
    print("=" * 60)


if __name__ == "__main__":
    main()

import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))

from src.config import (
    MODELS_CACHE,
    SENTIMENT_LABELS,
    ensure_dirs,
)
from src.data_processing import (
    get_aspect_split,
    get_sentiment_split,
    load_data,
)
from src.embeddings import get_or_cache_bert_features
from src.evaluation import (
    save_confusion_matrix,
    save_metrics_summary_json,
    save_model_comparison_csv,
)
from src.models import (
    save_artifacts,
    save_polarity_artifacts,
    train_aspects,
    train_polarities,
    train_sentiment,
)


def main() -> None:
    start_time = time.time()
    print("=" * 65)
    print("   BANGLA DARAZ ABSA - UNIFIED TRAINING & BENCHMARK PIPELINE   ")
    print("=" * 65)

    ensure_dirs()

    # -------------------------------------------------------------------------
    # 1. LOAD DATASET & SPLITS
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Loading dataset & generating stratified splits...")
    df = load_data()
    Xs_tr, Xs_te, ys_tr, ys_te = get_sentiment_split(df)
    Xa_tr, Xa_te, ya_tr, ya_te = get_aspect_split(df)
    print(f"  ✓ Loaded {len(df):,} reviews | Train: {len(Xs_tr):,} | Test: {len(Xs_te):,}")

    # -------------------------------------------------------------------------
    # 2. TF-IDF MODELS
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Training TF-IDF models (Sentiment, Aspects, Polarities)...")
    s_m_tf, s_v_tf, s_mt_tf = train_sentiment(Xs_tr, ys_tr, Xs_te, ys_te, use_bert=False)
    save_artifacts("sentiment", s_m_tf, s_v_tf, model_type="tfidf")
    save_confusion_matrix(
        ys_te, s_mt_tf["predictions"], SENTIMENT_LABELS,
        "Sentiment Confusion Matrix (TF-IDF)", "sentiment_tfidf.png"
    )
    print(f"  ✓ [TF-IDF] Sentiment -> Acc: {s_mt_tf['accuracy']:.4f} | Macro-F1: {s_mt_tf['macro_f1']:.4f} | Weighted-F1: {s_mt_tf['weighted_f1']:.4f}")

    a_m_tf, a_v_tf, a_b_tf, a_mt_tf = train_aspects(Xa_tr, ya_tr, Xa_te, ya_te, use_bert=False)
    save_artifacts("issue", a_m_tf, a_v_tf, a_b_tf, model_type="tfidf")
    print(f"  ✓ [TF-IDF] Aspects   -> Micro-F1: {a_mt_tf['micro_f1']:.4f} | Macro-F1: {a_mt_tf['macro_f1']:.4f} | Hamming Loss: {a_mt_tf['hamming_loss']:.4f}")

    pol_models_tf = train_polarities(df)
    save_polarity_artifacts(pol_models_tf, model_type="tfidf")
    for asp, p_entry in pol_models_tf.items():
        m = p_entry["metrics"]
        print(f"  ✓ [TF-IDF] {asp:15s} Polarity -> Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 3. BANGLABERT MODELS
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Extracting BanglaBERT embeddings & training classifiers...")
    s_tr_bert = get_or_cache_bert_features(Xs_tr, MODELS_CACHE / "sentiment_train.npy")
    s_te_bert = get_or_cache_bert_features(Xs_te, MODELS_CACHE / "sentiment_test.npy")
    a_tr_bert = get_or_cache_bert_features(Xa_tr, MODELS_CACHE / "issue_train.npy")
    a_te_bert = get_or_cache_bert_features(Xa_te, MODELS_CACHE / "issue_test.npy")

    s_m_bt, _, s_mt_bt = train_sentiment(s_tr_bert, ys_tr, s_te_bert, ys_te, use_bert=True)
    save_artifacts("sentiment", s_m_bt, model_type="bert")
    save_confusion_matrix(
        ys_te, s_mt_bt["predictions"], SENTIMENT_LABELS,
        "Sentiment Confusion Matrix (BanglaBERT)", "sentiment_bert.png"
    )
    print(f"  ✓ [BanglaBERT] Sentiment -> Acc: {s_mt_bt['accuracy']:.4f} | Macro-F1: {s_mt_bt['macro_f1']:.4f} | Weighted-F1: {s_mt_bt['weighted_f1']:.4f}")

    a_m_bt, _, a_b_bt, a_mt_bt = train_aspects(a_tr_bert, ya_tr, a_te_bert, ya_te, use_bert=True)
    save_artifacts("issue", a_m_bt, binarizer=a_b_bt, model_type="bert")
    print(f"  ✓ [BanglaBERT] Aspects   -> Micro-F1: {a_mt_bt['micro_f1']:.4f} | Macro-F1: {a_mt_bt['macro_f1']:.4f} | Hamming Loss: {a_mt_bt['hamming_loss']:.4f}")

    # -------------------------------------------------------------------------
    # 4. EXPORT BENCHMARKS & METRICS
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Exporting model_comparison.csv & metrics_summary.json...")
    comparison_rows: List[Dict[str, Any]] = [
        {
            "Task": "Sentiment Analysis",
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": round(s_mt_tf["accuracy"], 4),
            "Macro F1": round(s_mt_tf["macro_f1"], 4),
            "Weighted F1": round(s_mt_tf["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Sentiment Analysis",
            "Model": "BanglaBERT + Logistic Regression",
            "Accuracy": round(s_mt_bt["accuracy"], 4),
            "Macro F1": round(s_mt_bt["macro_f1"], 4),
            "Weighted F1": round(s_mt_bt["weighted_f1"], 4),
            "Additional Metric": "N/A"
        },
        {
            "Task": "Aspect Detection",
            "Model": "TF-IDF + OneVsRest LogReg",
            "Accuracy": round(1.0 - a_mt_tf["hamming_loss"], 4),
            "Macro F1": round(a_mt_tf["macro_f1"], 4),
            "Weighted F1": round(a_mt_tf["weighted_f1"], 4),
            "Additional Metric": f"Micro-F1: {a_mt_tf['micro_f1']:.4f} | Hamming Loss: {a_mt_tf['hamming_loss']:.4f}"
        },
        {
            "Task": "Aspect Detection",
            "Model": "BanglaBERT + OneVsRest LogReg",
            "Accuracy": round(1.0 - a_mt_bt["hamming_loss"], 4),
            "Macro F1": round(a_mt_bt["macro_f1"], 4),
            "Weighted F1": round(a_mt_bt["weighted_f1"], 4),
            "Additional Metric": f"Micro-F1: {a_mt_bt['micro_f1']:.4f} | Hamming Loss: {a_mt_bt['hamming_loss']:.4f}"
        }
    ]

    for asp, p_entry in pol_models_tf.items():
        m = p_entry["metrics"]
        comparison_rows.append({
            "Task": f"Polarity: {asp}",
            "Model": "TF-IDF + Balanced Binary LogReg",
            "Accuracy": round(m["accuracy"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "Weighted F1": round(m["weighted_f1"], 4),
            "Additional Metric": f"Samples: {p_entry['n_train']+p_entry['n_test']} (Train:{p_entry['n_train']})"
        })

    save_model_comparison_csv(comparison_rows, "model_comparison.csv")

    metrics_summary = {
        "dataset_statistics": {
            "total_reviews": len(df),
            "language": "bn",
            "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        },
        "sentiment_analysis": {
            "tfidf": {k: v for k, v in s_mt_tf.items() if k != "predictions"},
            "banglabert": {k: v for k, v in s_mt_bt.items() if k != "predictions"},
        },
        "aspect_detection": {
            "tfidf": {k: v for k, v in a_mt_tf.items() if k != "predictions"},
            "banglabert": {k: v for k, v in a_mt_bt.items() if k != "predictions"},
        },
        "aspect_polarities": {
            asp: {
                "accuracy": p_entry["metrics"]["accuracy"],
                "macro_f1": p_entry["metrics"]["macro_f1"],
                "n_train": p_entry["n_train"],
                "n_test": p_entry["n_test"],
            }
            for asp, p_entry in pol_models_tf.items()
        }
    }
    save_metrics_summary_json(metrics_summary, "metrics_summary.json")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"       ✅ TRAINING PIPELINE COMPLETED IN {elapsed:.1f}s!      ")
    print("=" * 65)


if __name__ == "__main__":
    main()

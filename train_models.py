import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.config import (
    MODELS_CACHE,
    SENTIMENT_LABELS,
    ALL_ASPECTS,
    LSTM_BATCH_SIZE,
    LSTM_MAX_LENGTH,
    ensure_dirs,
)
from src.data_processing import (
    get_aspect_split,
    get_sentiment_split,
    load_data,
)
from src.lstm_utils import BanglaVocab, create_dataloader
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
    train_lstm_sentiment,
    train_lstm_aspects
)
from sklearn.preprocessing import MultiLabelBinarizer

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
    s_m_tf, s_v_tf, s_mt_tf = train_sentiment(Xs_tr, ys_tr, Xs_te, ys_te)
    save_artifacts("sentiment", s_m_tf, s_v_tf, model_type="tfidf")
    save_confusion_matrix(
        ys_te, s_mt_tf["predictions"], SENTIMENT_LABELS,
        "Sentiment Confusion Matrix (TF-IDF)", "sentiment_tfidf.png"
    )
    print(f"  ✓ [TF-IDF] Sentiment -> Acc: {s_mt_tf['accuracy']:.4f} | Macro-F1: {s_mt_tf['macro_f1']:.4f} | Weighted-F1: {s_mt_tf['weighted_f1']:.4f}")

    a_m_tf, a_v_tf, a_b_tf, a_mt_tf = train_aspects(Xa_tr, ya_tr, Xa_te, ya_te)
    save_artifacts("issue", a_m_tf, a_v_tf, a_b_tf, model_type="tfidf")
    print(f"  ✓ [TF-IDF] Aspects   -> Micro-F1: {a_mt_tf['micro_f1']:.4f} | Macro-F1: {a_mt_tf['macro_f1']:.4f} | Hamming Loss: {a_mt_tf['hamming_loss']:.4f}")

    pol_models_tf = train_polarities(df)
    save_polarity_artifacts(pol_models_tf, model_type="tfidf")
    for asp, p_entry in pol_models_tf.items():
        m = p_entry["metrics"]
        print(f"  ✓ [TF-IDF] {asp:15s} Polarity -> Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 3. PYTORCH LSTM MODELS
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Preparing Vocab & Training LSTM classifiers...")
    vocab = BanglaVocab()
    vocab.fit(df["cleaned_text"].tolist())
    print(f"  ✓ Built Vocab with {vocab.vocab_size} tokens")

    # Map sentiment string labels to integers
    s_class_mapping = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
    ys_tr_int = [s_class_mapping[lbl] for lbl in ys_tr]
    ys_te_int = [s_class_mapping[lbl] for lbl in ys_te]

    s_train_loader = create_dataloader(Xs_tr, ys_tr_int, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE)
    s_test_loader = create_dataloader(Xs_te, ys_te_int, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE, shuffle=False)

    s_m_lstm, s_mt_lstm = train_lstm_sentiment(s_train_loader, s_test_loader, vocab.vocab_size, ys_te.tolist(), s_class_mapping)
    save_artifacts("sentiment", s_m_lstm, model_type="lstm", vocab=vocab)
    save_confusion_matrix(
        ys_te, s_mt_lstm["predictions"], SENTIMENT_LABELS,
        "Sentiment Confusion Matrix (LSTM)", "sentiment_lstm.png"
    )
    print(f"  ✓ [LSTM] Sentiment -> Acc: {s_mt_lstm['accuracy']:.4f} | Macro-F1: {s_mt_lstm['macro_f1']:.4f} | Weighted-F1: {s_mt_lstm['weighted_f1']:.4f}")

    # Map aspect lists to binary multi-labels
    mlb = MultiLabelBinarizer(classes=ALL_ASPECTS)
    ya_tr_bin = mlb.fit_transform(ya_tr)
    ya_te_bin = mlb.transform(ya_te)

    a_train_loader = create_dataloader(Xa_tr, ya_tr_bin, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE)
    a_test_loader = create_dataloader(Xa_te, ya_te_bin, vocab, LSTM_MAX_LENGTH, LSTM_BATCH_SIZE, shuffle=False)

    a_m_lstm, a_b_lstm, a_mt_lstm = train_lstm_aspects(a_train_loader, a_test_loader, vocab.vocab_size, ya_te_bin, mlb)
    save_artifacts("issue", a_m_lstm, binarizer=a_b_lstm, model_type="lstm", vocab=vocab)
    print(f"  ✓ [LSTM] Aspects   -> Micro-F1: {a_mt_lstm['micro_f1']:.4f} | Macro-F1: {a_mt_lstm['macro_f1']:.4f} | Hamming Loss: {a_mt_lstm['hamming_loss']:.4f}")

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
            "Model": "LSTM (PyTorch)",
            "Accuracy": round(s_mt_lstm["accuracy"], 4),
            "Macro F1": round(s_mt_lstm["macro_f1"], 4),
            "Weighted F1": round(s_mt_lstm["weighted_f1"], 4),
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
            "Model": "LSTM (PyTorch)",
            "Accuracy": round(1.0 - a_mt_lstm["hamming_loss"], 4),
            "Macro F1": round(a_mt_lstm["macro_f1"], 4),
            "Weighted F1": round(a_mt_lstm["weighted_f1"], 4),
            "Additional Metric": f"Micro-F1: {a_mt_lstm['micro_f1']:.4f} | Hamming Loss: {a_mt_lstm['hamming_loss']:.4f}"
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
            "lstm": {k: v for k, v in s_mt_lstm.items() if k != "predictions"},
        },
        "aspect_detection": {
            "tfidf": {k: v for k, v in a_mt_tf.items() if k != "predictions"},
            "lstm": {k: v for k, v in a_mt_lstm.items() if k != "predictions"},
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

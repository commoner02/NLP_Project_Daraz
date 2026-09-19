# Bangla Daraz Review Analytics & ABSA Platform
## Hierarchical Aspect-Based Sentiment Analysis on Bangla E-Commerce Reviews

An end-to-end Natural Language Processing (NLP) pipeline and interactive web dashboard designed for **Hierarchical Aspect-Based Sentiment Analysis (ABSA)** on Bangla customer reviews from Daraz Bangladesh.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Dataset & Preprocessing](#4-dataset--preprocessing)
5. [Model Architectures](#5-model-architectures)
6. [Installation & Setup](#6-installation--setup)
7. [How to Run](#7-how-to-run)
8. [Benchmark Results](#8-benchmark-results)

---

## 1. Project Overview

Online customer reviews often contain multi-faceted opinions where different aspects of a purchase have contrasting polarities (e.g., *"Product is great, but delivery was delayed"*). This platform addresses this challenge for Bangla through a three-level hierarchical formulation:

1. **Overall Sentiment Analysis**: 3-class classification (`Positive`, `Neutral`, `Negative`).
2. **Multi-Label Aspect Detection**: Identifies mentioned product and service categories across 5 core dimensions:
   - Product Quality
   - Price
   - Delivery
   - Packaging
   - Seller Service
3. **Aspect-Level Binary Polarity**: Dedicated binary classifiers (`Positive` vs. `Negative`) for each detected aspect.
4. **Dual Representation Benchmarking**:
   - **TF-IDF Pipeline**: Word n-grams (1, 2) + Character n-grams (3, 5) FeatureUnion with balanced Logistic Regression.
   - **PyTorch BiLSTM Pipeline**: Bidirectional LSTM with learned embeddings and mean/max pooling.

---

## 2. System Architecture

```
                                  +-------------------------------+
                                  |   Raw Bangla Daraz Reviews    |
                                  +---------------+---------------+
                                                  |
                                                  v
                                  +-------------------------------+
                                  |   Bangla Text Normalizer      |
                                  | (Unicode NFC, Noise Cleaning, |
                                  |  Negation Preservation)       |
                                  +---------------+---------------+
                                                  |
                                                  v
                        +-------------------------+-------------------------+
                        |                                                   |
                        v                                                   v
        +-------------------------------+                   +-------------------------------+
        |        TF-IDF Pipeline        |                   |      PyTorch BiLSTM Pipeline  |
        |  (Word (1,2) + Char (3,5))    |                   |  (Embedding + 2-layer BiLSTM) |
        +---------------+---------------+                   +---------------+---------------+
                        |                                                   |
                        +-------------------------+-------------------------+
                                                  |
                                                  v
                                  +-------------------------------+
                                  |   Hierarchical Predictions    |
                                  |   1. Overall Sentiment (3-Cls)|
                                  |   2. Aspect Detection (Multi) |
                                  |   3. Aspect Polarity (Binary) |
                                  +---------------+---------------+
                                                  |
                                                  v
                                  +-------------------------------+
                                  |   Streamlit Web Dashboard     |
                                  +-------------------------------+
```

---

## 3. Project Directory Structure

```
nlp_daraz/
├── .streamlit/
│   └── config.toml                  # UI theme settings
├── data/
│   ├── original_data/               # Mendeley raw source files
│   └── processed_data/
│       ├── annotated_bangla.csv     # Raw input corpus (review_id, original_text, label)
│       ├── clean_annotated_bangla.csv # Clean exported dataset with parsed targets
│       └── dataset_summary.json     # Class and aspect distribution stats
├── models/
│   ├── lstm/                        # PyTorch BiLSTM model binaries
│   └── tfidf/                       # Scikit-learn TF-IDF model binaries
├── results/
│   ├── metrics_summary.json         # Evaluation metrics JSON
│   ├── model_comparison.csv         # Comparative benchmark table
│   ├── sentiment_lstm.png           # BiLSTM confusion matrix
│   └── sentiment_tfidf.png          # TF-IDF confusion matrix
├── src/
│   ├── __init__.py
│   ├── config.py                    # Constants, paths, and hyperparameters
│   ├── preprocessing.py             # Unicode normalization and text cleaning
│   ├── data_processing.py           # Parsing labels and train/test splitting
│   ├── features.py                  # Vocab, PyTorch Datasets & LSTM models
│   ├── train.py                     # TF-IDF & BiLSTM training routines
│   ├── predict.py                   # Clean inference routines
│   ├── evaluate.py                  # Metric computation and plot savers
│   └── persistence.py               # Model save/load helpers
├── app.py                           # Streamlit web dashboard
├── pipeline.ipynb                   # Single end-to-end training notebook
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Project documentation
```

---

## 4. Dataset & Preprocessing

The primary input corpus is `data/processed_data/annotated_bangla.csv`, containing:
- `review_id`: Unique identifier for deduplication.
- `original_text`: Raw Bangla text.
- `label`: Compound aspect-sentiment annotation (e.g., `packaging_negative#product_quality_negative`).

Running `load_data()` cleans the Bangla text (preserving negations like *না*, *নাই*, *নয়*, *নেই*), parses the `label` into structured targets, and exports `clean_annotated_bangla.csv`.

---

## 5. Model Architectures

### 1. TF-IDF + Logistic Regression
- **Feature Union**: Word n-grams `(1, 2)` + Character n-grams `(3, 5)` with sublinear TF scaling.
- **Sentiment Model**: Balanced multiclass Logistic Regression (`lbfgs`).
- **Aspect Model**: Multi-label `OneVsRestClassifier(LogisticRegression)`.
- **Polarity Models**: Dedicated binary Logistic Regression per aspect.

### 2. PyTorch BiLSTM
- **Vocabulary**: Custom frequency-filtered Bangla tokenizer (`BanglaVocab`).
- **Architecture**:
  - `nn.Embedding(vocab_size, 300, padding_idx=0)`
  - 2-layer Bidirectional LSTM (`hidden_dim=128`, `dropout=0.3`)
  - Concatenated Mean-Pooling + Max-Pooling over sequence length
  - Dense linear projection to output classes.

---

## 6. Installation & Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .nlp_venv
source .nlp_venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 7. How to Run

### Option A: Training Pipeline (Jupyter Notebook)
Open and run all cells in `pipeline.ipynb`:
```bash
jupyter notebook pipeline.ipynb
```
The notebook executes end-to-end:
1. Data loading, cleaning, and target extraction.
2. TF-IDF models training & evaluation.
3. BiLSTM models training & evaluation.
4. Exporting comparison tables and confusion matrices to `results/`.
5. Live sample inference.

### Option B: Interactive Web Dashboard
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

---

## 8. Benchmark Results

Evaluation metrics on the test split ($n=404$):

| Task | Model | Accuracy | Macro F1 | Weighted F1 | Additional Metric |
|---|---|---|---|---|---|
| Sentiment Analysis | TF-IDF + LogReg | 0.8168 | 0.6974 | 0.8130 | N/A |
| Sentiment Analysis | BiLSTM | 0.8243 | 0.7042 | 0.8190 | N/A |
| Aspect Detection | TF-IDF + OvR | 0.8876 | 0.6558 | 0.8931 | Micro-F1: 0.8998 |
| Aspect Detection | BiLSTM | 0.8941 | 0.6841 | 0.9012 | Micro-F1: 0.9085 |

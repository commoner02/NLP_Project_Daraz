# 🛒 Bangla Daraz Review Analytics & ABSA Platform
## Hierarchical Aspect-Based Sentiment Analysis on Bangla E-Commerce Reviews

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Web%20App-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pretrained Models](https://img.shields.io/badge/HuggingFace-BanglaBERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/sagorsarker/bangla-bert-base)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

An end-to-end Natural Language Processing (NLP) analytics platform and interactive web dashboard designed for **Hierarchical Aspect-Based Sentiment Analysis (ABSA)** on Bangla customer reviews from Daraz Bangladesh.

---

## 📑 Table of Contents
1. [Project Overview & Key Objectives](#-1-project-overview--key-objectives)
2. [System Architecture](#-2-system-architecture)
3. [Project Directory Structure](#-3-project-directory-structure)
4. [NLP Methodology & Preprocessing](#-4-nlp-methodology--preprocessing)
5. [Installation & Setup](#-5-installation--setup)
6. [How to Run](#-6-how-to-run)
   - [A. Master CLI Training Script (`train_models.py`)](#a-master-cli-training-script-train_modelspy)
   - [B. Interactive Web Dashboard (`app.py`)](#b-interactive-web-dashboard-apppy)
   - [C. Exploratory Jupyter Notebook (`pipeline.ipynb`)](#c-exploratory-jupyter-notebook-pipelineipynb)
7. [Empirical Benchmark Results](#-7-empirical-benchmark-results)
8. [Live Inference Case Studies](#-8-live-inference-case-studies)
9. [Citation & License](#-9-citation--license)

---

## 🌟 1. Project Overview & Key Objectives

Online customer reviews often contain complex, multi-faceted opinions where different aspects of a purchase have contrasting polarities (e.g., *"Product is great, but delivery was delayed"*). This platform solves this challenge for the Bangla language through a three-level hierarchical formulation:

1. **Overall Sentiment Analysis**: 3-class classification (`Positive`, `Neutral`, `Negative`) with confidence metrics.
2. **Multi-Label Aspect Detection**: Identifies mentioned product and service categories across 5 core dimensions:
   - 🏷️ **Product Quality**
   - 🏷️ **Price**
   - 🏷️ **Delivery**
   - 🏷️ **Packaging**
   - 🏷️ **Seller Service**
3. **Aspect-Level Binary Polarity**: Trains dedicated binary classifiers (`Positive` vs. `Negative`) for each detected aspect dimension.
4. **Dual Feature Representation Benchmarking**:
   - **TF-IDF Pipeline**: Word n-grams `(1, 2)` + Character n-grams `(3, 5)` FeatureUnion with balanced Logistic Regression.
   - **BanglaBERT Pipeline**: 768-dimensional contextual sentence embeddings extracted from `sagorsarker/bangla-bert-base` with mean pooling.

---

## 🔍 2. System Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Raw Bangla Daraz Reviews    │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │   Bangla Text Normalizer      │
                                  │ (Unicode NFC, Zero-Width,     │
                                  │  Noise, Negation-Preserving)  │
                                  └──────────────┬────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
         ┌─────────────────────────────┐                   ┌─────────────────────────────┐
         │     TF-IDF Vectorizer       │                   │    BanglaBERT Embeddings    │
         │  (Word (1,2) + Char (3,5))  │                   │  (sagorsarker/bangla-bert)  │
         └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                        │                                                 │
         ┌──────────────┼──────────────┐                   ┌──────────────┴──────────────┐
         ▼              ▼              ▼                   ▼                             ▼
  ┌──────────────┐┌──────────────┐┌──────────────┐  ┌──────────────┐              ┌──────────────┐
  │  Sentiment   ││    Aspect    ││Aspect-Level  │  │  Sentiment   │              │    Aspect    │
  │  Classifier  ││ (Multi-Label)││Polarity (x5) │  │  Classifier  │              │ (Multi-Label)│
  │  (3-Class)   ││ (5 Aspects)  ││(Pos vs Neg)  │  │  (3-Class)   │              │ (5 Aspects)  │
  └──────────────┘└──────────────┘└──────────────┘  └──────────────┘              └──────────────┘
```

---

## 📁 3. Project Directory Structure

```
NLP_Project/
└── nlp_daraz/                         # Primary project root
    │
    ├── .streamlit/                    # Streamlit UI configuration
    │   └── config.toml                # Theme, port, and server settings
    │
    ├── data/                          # Dataset repository
    │   ├── original_data/             # Mendeley raw source benchmark files
    │   │   ├── annotated_dataset.csv  # 3,587 composite labeled reviews
    │   │   ├── preprocessed_dataset.csv# 10,657 reviews (bn, banglish, mix)
    │   │   └── original_dataset.csv   # 19,638 raw Daraz reviews
    │   │
    │   └── processed_data/            # Standardized, cleaned Bangla dataset
    │       └── annotated_bangla.csv   # 2,016 annotated Bangla reviews with ABSA ground truth
    │
    ├── models/                        # Serialized model binaries & caches
    │   ├── tfidf/                     # TF-IDF model weights & vectorizers (*.pkl)
    │   │   ├── sentiment_model.pkl
    │   │   ├── sentiment_vectorizer.pkl
    │   │   ├── issue_model.pkl
    │   │   ├── issue_vectorizer.pkl
    │   │   ├── issue_binarizer.pkl
    │   │   └── polarity_*_model.pkl   # Dedicated binary polarity models per aspect
    │   │
    │   ├── bert/                      # BanglaBERT classification heads (*.pkl)
    │   │   ├── sentiment_model.pkl
    │   │   ├── issue_model.pkl
    │   │   └── issue_binarizer.pkl
    │   │
    │   └── cache/                     # Precomputed 768-dim BERT embeddings (*.npy)
    │       ├── sentiment_train.npy
    │       ├── sentiment_test.npy
    │       ├── issue_train.npy
    │       └── issue_test.npy
    │
    ├── results/                       # Evaluation figures & benchmark tables
    │   ├── metrics_summary.json       # Complete precision/recall/F1 metrics JSON
    │   ├── model_comparison.csv       # Unified benchmark comparison table
    │   ├── sentiment_tfidf.png        # TF-IDF confusion matrix heatmap
    │   └── sentiment_bert.png         # BanglaBERT confusion matrix heatmap
    │
    ├── src/                           # Modular NLP Python package
    │   ├── __init__.py                # Package versioning
    │   ├── config.py                  # Paths, aspects, and model hyperparameters
    │   ├── preprocessing.py           # Unicode NFC, zero-width & text cleaning
    │   ├── data_processing.py         # Universal delimiter parser & stratified splitters
    │   ├── embeddings.py              # BanglaBERT extraction & .npy caching
    │   ├── models.py                  # Training, hierarchical ABSA & persistence
    │   └── evaluation.py              # Heatmap plots & JSON/CSV exporters
    │
    ├── app.py                         # Interactive Streamlit analytics dashboard
    ├── train_models.py                # End-to-end training & benchmarking CLI script
    ├── pipeline.ipynb                 # Interactive Jupyter notebook
    ├── requirements.txt               # Dependencies with CPU PyTorch index
    ├── .gitignore                     # Git tracking exclusions
    ├── PROJECT_CODEBASE.md            # Complete Codebase & Architecture Reference
    └── README.md                      # Project documentation
```

---

## 🔬 4. NLP Methodology & Preprocessing

1. **Unicode NFC Normalization & Zero-Width Stripping**:
   - Converts decomposed Unicode vowel markers into canonical composed forms (NFC).
   - Removes invisible zero-width characters (`\u200c`, `\u200d`, `\ufeff`) injected by mobile Bangla keyboard layouts (Avro, Gboard, Ridmik).
2. **Noise & Elongation Handling**:
   - Strips HTML tags, URLs, and email addresses.
   - Collapses character elongations (e.g., *"খুউউব"* $\rightarrow$ *"খুব"*).
3. **Negation & Contrast Retention**:
   - Preserves all sentiment-bearing negation particles (*"না"*, *"নাই"*, *"নয়"*, *"নেই"*) and contrastive conjunctions (*"কিন্তু"*). Traditional stopword removal strips `"না"`, flipping negative reviews like *"ব্যাটারি ভালো না"* into positive ones.
4. **Code-Switching & Loanword Preservation**:
   - Daraz reviews frequently mix English terms with Bangla script. The cleaner retains alphanumeric tokens (*"battery"*, *"delivery"*, *"sound quality"*).
5. **Universal Delimiter Parsing**:
   - Robustly parses composite ABSA tags separated by either `#` (e.g., `delivery_negative#product_quality_positive`) or `;` (e.g., `delivery_negative;product_quality_positive`).

---

## 🚀 5. Installation & Setup

```bash
# 1. Clone or navigate to the repository
cd /path/to/NLP_Project/nlp_daraz

# 2. Create a virtual environment
python3 -m venv .nlp_venv

# 3. Activate the virtual environment
source .nlp_venv/bin/activate       # Linux / macOS
# .nlp_venv\Scripts\activate        # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🏃 6. How to Run

### A. Master CLI Training Script (`train_models.py`)
To train all models, extract BanglaBERT embeddings, evaluate on test sets, and generate benchmark tables in ~50 seconds:
```bash
python train_models.py
```

### B. Interactive Web Dashboard (`app.py`)
To launch the interactive multi-tab Streamlit web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

#### Dashboard Capabilities:
- **🔍 Review Analyzer**: Real-time multi-task inference with instant toggle between **TF-IDF** and **BanglaBERT**, sentiment confidence meters, quick presets, and detected aspect cards with polarity badges (✅ Positive / 😡 Negative / ⚠️ Low Confidence).
- **📈 Benchmarks & Data Insights**: Empirical benchmark table, side-by-side confusion matrices, sentiment distribution pie charts, and aspect frequency bar charts.

### C. Exploratory Jupyter Notebook (`pipeline.ipynb`)
Open `pipeline.ipynb` in VS Code or Jupyter Notebook to step through the data exploration, feature extraction, and modeling cells interactively.

---

## 📊 7. Empirical Benchmark Results

Evaluated on held-out 20% stratified test sets (`annotated_bangla.csv`, 2,016 reviews):

| Task | Model Architecture | Accuracy | Macro F1 | Weighted F1 | Additional Metrics & Sample Sizes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Sentiment Analysis** | **TF-IDF + Logistic Regression** | **91.58%** | **0.8441** | **0.9182** | 404 test samples (Stratified) |
| **Sentiment Analysis** | BanglaBERT + Logistic Regression | 82.92% | 0.6966 | 0.8322 | Frozen mean-pooled 768-dim embeddings |
| **Aspect Detection** | **TF-IDF + OneVsRest LogReg** | **96.68%** | **0.7999** | **0.9412** | **Micro-F1: 0.9410** \| **Hamming Loss: 0.0332** |
| **Aspect Detection** | BanglaBERT + OneVsRest LogReg | 92.97% | 0.6501 | 0.8937 | **Micro-F1: 0.8797** \| **Hamming Loss: 0.0703** |
| **Polarity: Product Quality** | TF-IDF + Balanced Binary LogReg | **97.53%** | **0.9560** | **0.9755** | 1,816 total samples (Train: 1,452) |
| **Polarity: Price** | TF-IDF + Balanced Binary LogReg | **97.89%** | **0.7446** | **0.9738** | 471 total samples (Train: 376) |
| **Polarity: Delivery** | TF-IDF + Balanced Binary LogReg | **89.47%** | **0.8021** | **0.8900** | 282 total samples (Train: 225) |
| **Polarity: Packaging** | TF-IDF + Balanced Binary LogReg | **80.00%** | **0.7205** | **0.7702** | 72 total samples (Train: 57) |
| **Polarity: Seller Service** | TF-IDF + Balanced Binary LogReg | **95.45%** | **0.4884** | **0.9323** | 108 total samples (Train: 86) |

---

## 🧪 8. Live Inference Case Studies

| Input Bangla Review | Predicted Sentiment | Detected Aspects & Specific Polarities | Analysis |
| :--- | :--- | :--- | :--- |
| `"সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।"` | **Negative** (47.8%) | 🏷️ Product Quality: **Negative** (62.6% 😡) | ✅ Negation *"ভালো না"* is properly captured. |
| `"সেলার খুব হেল্পফুল ছিল কিন্তু ডেলিভারি দেরি হয়েছে।"` | **Neutral** (81.0%) | 🏷️ Delivery: **Negative** (68.8% 😡)<br>🏷️ Seller Service: **Positive** (74.9% ✅) | ✅ Contrasting aspect polarities are correctly separated. |
| `"দাম অনেক বেশি কিন্তু কোয়ালিটি একদম বাজে।"` | **Neutral** (50.5%) | 🏷️ Product Quality: **Negative** (64.4% 😡)<br>🏷️ Price: **Negative** (87.0% 😡) | ✅ Multi-aspect negative feedback identified. |
| `""` *(Empty review)* | **Neutral** (0.0%) | *No aspects detected* | ✅ Safe fallback without injecting false positives. |

---

## 📄 9. Citation & License

- **Dataset**: Mendeley Data (*Daraz E-Commerce Reviews ABSA Dataset*)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))

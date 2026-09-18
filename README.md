# 🛒 Bangla Daraz Review Analytics & ABSA Platform

A streamlined natural language processing (NLP) analytics platform and interactive Streamlit application designed for **Aspect-Based Sentiment Analysis (ABSA)** on Daraz Bangladesh reviews.

The system benchmarks two distinct feature representation families across two foundational tasks:
1. **Sentiment Analysis**: 3-class classification (`Positive`, `Neutral`, `Negative`)
2. **Aspect Detection**: Multi-label classification across 5 key aspects (`Product Quality`, `Price`, `Delivery`, `Packaging`, `Seller Service`).

---

## 📁 1. Simplified Project Structure

```
nlp_daraz/
│
├── data/                             # Data directory
│   ├── original_data/                # Original Mendeley source corpus
│   │   ├── original_dataset.csv      # 19,638 raw Daraz reviews
│   │   ├── preprocessed_dataset.csv  # 10,657 reviews (bn, banglish, mix)
│   │   └── annotated_dataset.csv     # 3,587 composite labeled reviews
│   │
│   ├── processed_data/               # Cleaned Bangla datasets (Active)
│   │   └── annotated_bangla.csv      # 2,016 cleaned Bangla reviews with ABSA ground truth
│   │
│   └── bangla_stopwords.txt          # Curated Bangla stopwords
│
├── models/                           # Model binaries & caches
│   ├── tfidf/                        # TF-IDF model weights & vectorizers (*.pkl)
│   ├── bert/                         # BanglaBERT classifier weights (*.pkl)
│   └── cache/                        # Cached 768-dim BERT embeddings (*.npy)
│
├── results/                          # Output evaluation plots and reports
│   ├── *.png                         # Confusion matrix heatmaps
│   ├── *.csv                         # Classification reports & 4-row model comparison
│   └── metrics_summary.json          # Overall metrics summary
│
├── src/                              # Clean, modular Python helpers
│   ├── __init__.py
│   ├── config.py                     # Centralized paths and hyperparameters
│   ├── preprocessing.py              # Pure Bangla text cleaning pipeline
│   ├── data_processing.py            # Annotated data loading and stratified train/test splits
│   ├── embeddings.py                 # BanglaBERT feature extraction and caching
│   ├── models.py                     # Unified train, predict, save, and load routines
│   └── evaluation.py                 # Metrics computation, plots, and CSV/JSON export
│
├── pipeline.ipynb                    # 📓 Master Interactive Jupyter Notebook
├── train_models.py                   # 🚀 1-Click CLI training script
├── app.py                            # 🌐 Interactive Streamlit Web UI
├── requirements.txt                  # Python dependencies (defaults to CPU PyTorch)
├── .gitignore                        # Git exclusion rules
├── PROJECT_CODEBASE.md               # 📄 Complete Codebase & Architecture Reference
└── README.md                         # Documentation
```

---

## 🚀 2. Quickstart & Installation

### Step 1: Create Virtual Environment
```bash
# 1. Create a clean virtual environment:
python3 -m venv .nlp_venv

# 2. Activate the virtual environment:
source .nlp_venv/bin/activate

# 3. Install dependencies:
pip install -r nlp_daraz/requirements.txt
```

---

## 🏃 3. How to Run the Project

### Option A: Interactive Jupyter Notebook (`pipeline.ipynb`)
Open `pipeline.ipynb` in VS Code or Jupyter Notebook:
```bash
# Select .nlp_venv as your Jupyter Kernel and run through cells:
# 1. Text Preprocessing demo
# 2. Data Loading & Distribution plots
# 3. TF-IDF Models Training (Sentiment & Aspects)
# 4. BanglaBERT Feature Extraction & Caching
# 5. BanglaBERT Classifier Training
# 6. Benchmark Comparison Table
# 7. Live Custom Review Tester
```

### Option B: Command-Line Training Script (`train_models.py`)
Run the master training script from your terminal:
```bash
python train_models.py
```

---

## 🌐 4. Launching the Web Dashboard

To launch the interactive multi-tab Streamlit dashboard:

```bash
streamlit run app.py
```

### Dashboard Features:
- **🔍 Review Analyzer**: Real-time multi-task inference with instant toggle between **TF-IDF** and **BanglaBERT** architectures + Bangla review presets.
- **📈 Benchmarks & Data Insights**: 4-row empirical benchmark table, test accuracy, macro F1, Hamming loss, confusion matrix heatmaps, and aspect mention distributions.

---

## 📊 5. Empirical Benchmark Comparison

Performance on held-out 20% stratified test sets:

| Task | Architecture / Model | Accuracy | Macro F1 | Weighted F1 | Additional Metric |
|---|---|---|---|---|---|
| **Sentiment Analysis** | TF-IDF + Logistic Regression | **87.62%** | **0.7747** | **0.8777** | — |
| **Sentiment Analysis** | BanglaBERT + Logistic Regression | 82.92% | 0.6966 | 0.8322 | — |
| **Aspect Detection** | TF-IDF + OneVsRest LogReg | **96.34%** | **0.7804** | **0.9353** | Hamming Loss: **0.0366** |
| **Aspect Detection** | BanglaBERT + OneVsRest LogReg | 92.97% | 0.6501 | 0.8937 | Hamming Loss: 0.0703 |

---

## 📄 6. Citation & License
- **Dataset**: Mendeley Data (Daraz E-Commerce Reviews ABSA Dataset)
- **License**: CC BY-NC 4.0

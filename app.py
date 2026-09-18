"""
Streamlit Web Application: Bangla Review Sentiment & Aspect Analytics Platform
Aspect-Based Sentiment Analysis (ABSA) for Daraz Bangladesh.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.config import ALL_ASPECTS, RESULTS_DIR
try:
    from src.models import load_artifacts, predict_aspects, predict_sentiment
except ImportError:
    from src.models import load_artifacts, predict_sentiment, predict_issue as predict_aspects
from src.preprocessing import clean_text

# Page Config
st.set_page_config(
    page_title="Bangla Daraz ABSA Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #4B5563; margin-bottom: 1.2rem; }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px 18px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-title { font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .sentiment-pos { color: #10B981; }
    .sentiment-neg { color: #EF4444; }
    .sentiment-neu { color: #F59E0B; }
    .aspect-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 6px 14px;
        border-radius: 16px;
        font-size: 0.92rem;
        font-weight: 600;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load both TF-IDF and BanglaBERT model artifacts."""
    loaded = {"status": "ready", "tfidf": {}, "bert": {}}
    try:
        s_m_tf, s_v_tf, _ = load_artifacts("sentiment", "tfidf")
        a_m_tf, a_v_tf, a_b_tf = load_artifacts("issue", "tfidf")
        loaded["tfidf"] = {
            "sentiment_model": s_m_tf,
            "sentiment_vectorizer": s_v_tf,
            "aspect_model": a_m_tf,
            "aspect_vectorizer": a_v_tf,
            "aspect_binarizer": a_b_tf
        }

        s_m_bt, _, _ = load_artifacts("sentiment", "bert")
        a_m_bt, _, a_b_bt = load_artifacts("issue", "bert")
        loaded["bert"] = {
            "sentiment_model": s_m_bt,
            "aspect_model": a_m_bt,
            "aspect_binarizer": a_b_bt or a_b_tf
        }
    except Exception as e:
        loaded["status"] = "error"
        loaded["message"] = str(e)
    return loaded


@st.cache_data
def load_dataset():
    """Load annotated ABSA dataset."""
    from src.data_processing import load_annotated_data
    try:
        return load_annotated_data()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_benchmarks():
    """Load model comparison benchmarks from results/."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path)
    return pd.DataFrame()


models_data = load_all_models()
dataset_df = load_dataset()
comparison_df = load_benchmarks()

# Sidebar
st.sidebar.title("🛒 Daraz ABSA Analytics")
st.sidebar.markdown("**Sentiment & Aspect NLP Platform**")

view_mode = st.sidebar.radio(
    "Navigation",
    ["🔍 Review Analyzer", "📈 Benchmarks & Data Insights"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Architecture")
selected_model = st.sidebar.radio(
    "Select Model:",
    ["TF-IDF", "BanglaBERT"],
    help="Toggle between TF-IDF (N-gram Union) and BanglaBERT feature representations."
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Corpus**: Mendeley ABSA Dataset\n\n"
    f"**Total Annotated Reviews**: {len(dataset_df):,} rows\n\n"
    f"**Aspects**: 5 Product Dimensions"
)


# -----------------------------------------------------------------------------
# 1. REVIEW ANALYZER
# -----------------------------------------------------------------------------
if view_mode == "🔍 Review Analyzer":
    st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Live NLP inference using <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

    if models_data["status"] != "ready":
        st.error(f"Models not loaded: {models_data.get('message')}. Please run `python train_models.py` first.")
        st.stop()

    use_bert = (selected_model == "BanglaBERT")
    active = models_data["bert"] if use_bert else models_data["tfidf"]

    presets = {
        "Select a sample review preset...": "",
        "Preset 1 (Positive Quality & Fast Delivery)": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
        "Preset 2 (Negative Delay & Poor Quality)": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
        "Preset 3 (Damaged Packaging & Good Product)": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
        "Preset 4 (Excellent Seller Service & Accurate)": "সেলার খুব ভালো ব্যবহার করেছে, প্রোডাক্ট যেমন দেখেছি তেমনই পেয়েছি।",
        "Preset 5 (Price Concern & Quality Failure)": "দাম অনেক বেশি কিন্তু কোয়ালিটি একদম বাজে।"
    }

    selected_preset = st.selectbox("💡 Quick Test Presets:", list(presets.keys()))
    default_text = presets[selected_preset]

    user_text = st.text_area(
        "Enter Bangla Review Text:",
        value=default_text,
        height=95,
        placeholder="দারাজ রিভিউ এখানে লিখুন... (e.g. প্রোডাক্ট ভালো ছিলো কিন্তু ডেলিভারি দেরি হয়েছে)"
    )

    if st.button("🚀 Analyze Review", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner(f"Analyzing with {selected_model}..."):
                cleaned = clean_text(user_text)

                if use_bert:
                    s_res = predict_sentiment(user_text, active["sentiment_model"], use_bert=True)
                    a_res = predict_aspects(
                        user_text,
                        active["aspect_model"],
                        binarizer=active["aspect_binarizer"],
                        use_bert=True
                    )
                else:
                    s_res = predict_sentiment(
                        user_text,
                        active["sentiment_model"],
                        vectorizer=active["sentiment_vectorizer"],
                        use_bert=False
                    )
                    a_res = predict_aspects(
                        user_text,
                        active["aspect_model"],
                        vectorizer=active["aspect_vectorizer"],
                        binarizer=active["aspect_binarizer"],
                        use_bert=False
                    )

            st.markdown("---")
            st.subheader("NLP Prediction Results")

            c1, c2 = st.columns([1, 1.2])

            # Sentiment Box
            with c1:
                s_lbl = s_res["sentiment"]
                s_cls = "sentiment-pos" if s_lbl == "Positive" else ("sentiment-neg" if s_lbl == "Negative" else "sentiment-neu")
                s_ico = "😊" if s_lbl == "Positive" else ("😡" if s_lbl == "Negative" else "😐")

                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Predicted Sentiment</div>
                    <div class="kpi-value {s_cls}">{s_ico} {s_lbl}</div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Confidence: {s_res['confidence']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                if s_res["probabilities"]:
                    prob_df = pd.DataFrame(list(s_res["probabilities"].items()), columns=["Sentiment", "Probability"])
                    prob_df["Percentage"] = prob_df["Probability"] * 100
                    fig_s = px.bar(
                        prob_df, x="Percentage", y="Sentiment", orientation="h",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
                        text=prob_df["Percentage"].apply(lambda p: f"{p:.1f}%")
                    )
                    fig_s.update_layout(height=160, margin=dict(t=8, b=8, l=8, r=8), showlegend=False, xaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_s, width="stretch")

            # Aspect Box
            with c2:
                st.markdown("""
                <div class="kpi-card">
                    <div class="kpi-title">Detected Aspects</div>
                    <div style="margin-top: 10px;">
                """, unsafe_allow_html=True)

                asp_html = ""
                for asp in a_res["aspects"]:
                    asp_html += f'<span class="aspect-badge">🏷️ {asp}</span>'
                st.markdown(asp_html, unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            with st.expander("🔍 Cleaned Bangla Tokens"):
                st.code(cleaned, language="text")


# -----------------------------------------------------------------------------
# 2. BENCHMARKS & DATA INSIGHTS
# -----------------------------------------------------------------------------
elif view_mode == "📈 Benchmarks & Data Insights":
    st.markdown('<div class="main-header">Model Performance & Empirical Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Direct evaluation on held-out 20% test sets (404 reviews).</div>', unsafe_allow_html=True)

    if not comparison_df.empty:
        st.subheader("1. Dual-Model Benchmark Summary (4 Configurations)")
        st.dataframe(comparison_df, width="stretch")
    else:
        st.info("Run `python train_models.py` to generate the benchmark table.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrices
    st.subheader("2. Sentiment Confusion Matrices")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("**TF-IDF Confusion Matrix**")
        p_tf = RESULTS_DIR / "sentiment_tfidf.png"
        if p_tf.exists():
            st.image(str(p_tf), width="stretch")
    with c_m2:
        st.markdown("**BanglaBERT Confusion Matrix**")
        p_bt = RESULTS_DIR / "sentiment_bert.png"
        if p_bt.exists():
            st.image(str(p_bt), width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Explorer
    st.subheader("3. Dataset Distribution & Explorer")
    d1, d2 = st.columns([1, 1.3])

    if not dataset_df.empty and "sentiment" in dataset_df.columns:
        with d1:
            sent_counts = dataset_df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig_p = px.pie(
                sent_counts, names="Sentiment", values="Count", hole=0.4,
                color="Sentiment",
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}
            )
            fig_p.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_p, width="stretch")

        with d2:
            aspect_items = []
            for item in dataset_df["aspects_str"].dropna():
                for a in str(item).split(";"):
                    if a.strip():
                        aspect_items.append(a.strip())
            asp_s = pd.Series(aspect_items).value_counts().reset_index()
            asp_s.columns = ["Aspect", "Mentions"]
            fig_b = px.bar(asp_s, x="Mentions", y="Aspect", orientation="h", color="Mentions", color_continuous_scale="Blues")
            fig_b.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_b, width="stretch")

    st.dataframe(dataset_df[["cleaned_text", "sentiment", "aspects_str"]].head(10).rename(columns={
        "cleaned_text": "Review (Bangla)",
        "sentiment": "Sentiment",
        "aspects_str": "Aspects"
    }), width="stretch")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
    "Bangla Daraz Review Analytics • Sentiment & Aspect NLP Platform"
    "</div>",
    unsafe_allow_html=True
)

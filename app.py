import os
import sys
from pathlib import Path
import pandas as pd

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    px = None
    HAS_PLOTLY = False

try:
    import streamlit as st
except ImportError:
    st = None

sys.path.insert(0, os.path.abspath("."))

from src.config import RESULTS_DIR
from src.models import (
    load_artifacts,
    load_polarity_artifacts,
    predict_hierarchical,
    predict_sentiment,
)
from src.preprocessing import clean_text


def load_all_models():
    """Load model artifacts for TF-IDF and BanglaBERT pipelines."""
    loaded = {"status": "ready", "tfidf": {}, "bert": {}}
    try:
        s_m_tf, s_v_tf, _ = load_artifacts("sentiment", "tfidf")
        a_m_tf, a_v_tf, a_b_tf = load_artifacts("issue", "tfidf")
        p_models_tf = load_polarity_artifacts("tfidf")

        loaded["tfidf"] = {
            "sentiment_model": s_m_tf,
            "sentiment_vectorizer": s_v_tf,
            "aspect_model": a_m_tf,
            "aspect_vectorizer": a_v_tf,
            "aspect_binarizer": a_b_tf,
            "polarity_models": p_models_tf,
        }

        s_m_bt, _, _ = load_artifacts("sentiment", "bert")
        a_m_bt, _, a_b_bt = load_artifacts("issue", "bert")
        loaded["bert"] = {
            "sentiment_model": s_m_bt,
            "aspect_model": a_m_bt,
            "aspect_binarizer": a_b_bt or a_b_tf,
            "polarity_models": p_models_tf,
        }
    except Exception as e:
        loaded["status"] = "error"
        loaded["message"] = str(e)
    return loaded


def load_dataset():
    """Load preprocessed ABSA dataset."""
    from src.data_processing import load_data
    try:
        return load_data()
    except Exception:
        return pd.DataFrame()


def load_benchmarks():
    """Load model comparison benchmarks."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        return pd.read_csv(comp_path)
    return pd.DataFrame()


# Only execute Streamlit UI flow when running within Streamlit
if st is not None:
    # Page Configuration
    st.set_page_config(
        page_title="Bangla Daraz ABSA Analytics",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
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
        .aspect-row {
            margin-bottom: 8px;
            padding: 10px 14px;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .aspect-name { font-weight: 600; color: #1E293B; font-size: 0.95rem; }
        .aspect-pill { font-weight: 600; font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    load_models_cached = st.cache_resource(load_all_models)
    load_data_cached = st.cache_data(load_dataset)
    load_benchmarks_cached = st.cache_data(load_benchmarks)

    models_data = load_models_cached()
    dataset_df = load_data_cached()
    comparison_df = load_benchmarks_cached()

    # Sidebar
    st.sidebar.title("🛒 Daraz ABSA")
    st.sidebar.markdown("**Sentiment & Aspect NLP Platform**")

    view_mode = st.sidebar.radio(
        "Navigation",
        ["🔍 Review Analyzer", "📈 Benchmarks & Data Insights"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Pipeline")
    selected_model = st.sidebar.radio(
        "Select Model Architecture:",
        ["TF-IDF", "BanglaBERT"],
        help="Toggle between TF-IDF (N-gram Union) and BanglaBERT representations."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Corpus**: Mendeley Bangla Daraz ABSA\n\n"
        f"**Total Reviews**: {len(dataset_df):,} rows\n\n"
        f"**Aspects**: 5 Dimensions (Quality, Price, Delivery, Packaging, Seller)"
    )

    # 1. REVIEW ANALYZER
    if view_mode == "🔍 Review Analyzer":
        st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Hierarchical ABSA live inference via <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

        if models_data["status"] != "ready":
            st.error(f"Models missing: {models_data.get('message')}. Please run `python train_models.py` first.")
            st.stop()

        use_bert = (selected_model == "BanglaBERT")
        active = models_data["bert"] if use_bert else models_data["tfidf"]

        presets = {
            "— choose sample preset —": "",
            "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
            "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
            "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
            "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
            "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল。"
        }

        selected_preset = st.selectbox("💡 Quick Test Presets:", list(presets.keys()))
        default_text = presets.get(selected_preset or "", "")

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
                try:
                    with st.spinner(f"Analyzing with {selected_model}..."):
                        cleaned = clean_text(user_text)

                        if use_bert:
                            s_res = predict_sentiment(user_text, active["sentiment_model"], use_bert=True)
                            a_res = predict_hierarchical(
                                user_text,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
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
                            a_res = predict_hierarchical(
                                user_text,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
                                vectorizer=active["aspect_vectorizer"],
                                binarizer=active["aspect_binarizer"],
                                use_bert=False
                            )
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
                    st.stop()

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
                        <div class="kpi-title">Overall Predicted Sentiment</div>
                        <div class="kpi-value {s_cls}">{s_ico} {s_lbl}</div>
                        <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Confidence: {s_res['confidence']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if s_res["probabilities"]:
                        prob_df = pd.DataFrame(list(s_res["probabilities"].items()), columns=["Sentiment", "Probability"])
                        prob_df["Percentage"] = prob_df["Probability"] * 100
                        if px is not None:
                            fig_s = px.bar(
                                prob_df, x="Percentage", y="Sentiment", orientation="h",
                                color="Sentiment",
                                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
                                text=prob_df["Percentage"].apply(lambda p: f"{p:.1f}%")
                            )
                            fig_s.update_layout(height=160, margin=dict(t=8, b=8, l=8, r=8), showlegend=False, xaxis=dict(range=[0, 100]))
                            st.plotly_chart(fig_s, width="stretch")
                        else:
                            st.bar_chart(prob_df.set_index("Sentiment")["Percentage"])

                # Aspect Box
                with c2:
                    st.markdown("""
                    <div class="kpi-card">
                        <div class="kpi-title">Detected Aspects & Polarities</div>
                        <div style="margin-top: 10px;">
                    """, unsafe_allow_html=True)

                    aspect_details = a_res.get("aspect_details", [])
                    if aspect_details:
                        asp_html = ""
                        for item in aspect_details:
                            asp = item["aspect"]
                            pol = item["polarity"]
                            icon = item["icon"]
                            conf = item["confidence"] * 100
                            color = item["color"]
                            bg_color = item.get("bg_color", "#ECFDF5" if pol == "Positive" else "#FEF2F2")
                            asp_html += (
                                f'<div class="aspect-row">'
                                f'  <span class="aspect-name">🏷️ {asp}</span>'
                                f'  <span class="aspect-pill" style="color: {color}; background-color: {bg_color}; border: 1px solid {color}44;">'
                                f'    {icon} {pol} <span style="font-size: 0.78rem; opacity: 0.85;">({conf:.1f}%)</span>'
                                f'  </span>'
                                f'</div>'
                            )
                        st.markdown(asp_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<em>No specific aspect detected.</em>", unsafe_allow_html=True)

                    st.markdown("</div></div>", unsafe_allow_html=True)

                with st.expander("🔍 Cleaned Tokens"):
                    st.code(cleaned, language="text")

    # 2. BENCHMARKS & DATA INSIGHTS
    elif view_mode == "📈 Benchmarks & Data Insights":
        st.markdown('<div class="main-header">Model Performance & Empirical Benchmarks</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluation on held-out 20% test sets (Stratified ABSA splits).</div>', unsafe_allow_html=True)

        if not comparison_df.empty:
            st.subheader("1. Comprehensive Model Benchmark Summary")
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

        # Dataset Distribution
        st.subheader("3. Dataset Distribution & Explorer")
        d1, d2 = st.columns([1, 1.3])

        if not dataset_df.empty and "sentiment" in dataset_df.columns:
            with d1:
                sent_counts = dataset_df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                if px is not None:
                    fig_p = px.pie(
                        sent_counts, names="Sentiment", values="Count", hole=0.4,
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}
                    )
                    fig_p.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_p, width="stretch")
                else:
                    st.write(sent_counts)

            with d2:
                aspect_items = []
                for item in dataset_df["aspects_str"].dropna():
                    for a in str(item).split(";"):
                        if a.strip():
                            aspect_items.append(a.strip())
                asp_s = pd.Series(aspect_items).value_counts().reset_index()
                asp_s.columns = ["Aspect", "Mentions"]
                if px is not None:
                    fig_b = px.bar(asp_s, x="Mentions", y="Aspect", orientation="h", color="Mentions", color_continuous_scale="Blues")
                    fig_b.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_b, width="stretch")
                else:
                    st.write(asp_s)

        if not dataset_df.empty:
            st.dataframe(dataset_df[["cleaned_text", "sentiment", "aspects_str"]].head(10).rename(columns={
                "cleaned_text": "Review (Bangla)",
                "sentiment": "Sentiment",
                "aspects_str": "Aspects"
            }), width="stretch")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
        "Bangla Daraz Review Analytics • Hierarchical Aspect-Based Sentiment Analysis (ABSA)"
        "</div>",
        unsafe_allow_html=True
    )

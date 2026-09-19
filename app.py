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
    """Load model artifacts for TF-IDF and LSTM pipelines."""
    loaded = {"status": "ready", "tfidf": {}, "lstm": {}}
    try:
        s_m_tf, s_v_tf, _, _ = load_artifacts("sentiment", "tfidf")
        a_m_tf, a_v_tf, a_b_tf, _ = load_artifacts("issue", "tfidf")
        p_models_tf = load_polarity_artifacts("tfidf")

        loaded["tfidf"] = {
            "sentiment_model": s_m_tf,
            "sentiment_vectorizer": s_v_tf,
            "aspect_model": a_m_tf,
            "aspect_vectorizer": a_v_tf,
            "aspect_binarizer": a_b_tf,
            "polarity_models": p_models_tf,
        }

        s_m_lstm, _, _, s_v_lstm = load_artifacts("sentiment", "lstm")
        a_m_lstm, _, a_b_lstm, _ = load_artifacts("issue", "lstm")
        loaded["lstm"] = {
            "sentiment_model": s_m_lstm,
            "aspect_model": a_m_lstm,
            "aspect_binarizer": a_b_lstm or a_b_tf,
            "vocab": s_v_lstm,
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header { 
            font-size: 2.4rem; 
            font-weight: 800; 
            background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem; 
        }
        .sub-header { 
            font-size: 1.1rem; 
            color: #64748B; 
            margin-bottom: 1.8rem; 
            font-weight: 500;
        }
        .kpi-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px 24px;
            border: 1px solid #E2E8F0;
            border-top: 5px solid #3B82F6;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: all 0.3s ease;
        }
        .kpi-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        .kpi-title { font-size: 0.85rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
        .kpi-value { font-size: 1.8rem; font-weight: 700; margin-top: 6px; display: flex; align-items: center; gap: 8px; }
        
        .sentiment-pos { color: #059669; }
        .sentiment-neg { color: #DC2626; }
        .sentiment-neu { color: #D97706; }
        
        .aspect-row {
            margin-bottom: 10px;
            padding: 12px 16px;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
        }
        .aspect-row:hover {
            background-color: #FFFFFF;
            border-color: #CBD5E1;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transform: translateX(4px);
        }
        .aspect-name { font-weight: 600; color: #334155; font-size: 0.95rem; }
        .aspect-pill { 
            font-weight: 600; 
            font-size: 0.85rem; 
            padding: 6px 12px; 
            border-radius: 999px;
            display: flex;
            align-items: center;
            gap: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    load_models_cached = st.cache_resource(load_all_models)
    load_data_cached = st.cache_data(load_dataset)
    load_benchmarks_cached = st.cache_data(load_benchmarks)

    models_data = load_models_cached()
    dataset_df = load_data_cached()
    comparison_df = load_benchmarks_cached()

    # Sidebar
    st.sidebar.markdown("<h2>✨ Daraz ABSA</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='color: #64748B; font-weight: 500; margin-top: -10px; margin-bottom: 20px;'>Sentiment & Aspect NLP Platform</div>", unsafe_allow_html=True)

    view_mode = st.sidebar.radio(
        "Navigation",
        ["🔍 Review Analyzer", "📈 Benchmarks & Data Insights"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Pipeline")
    selected_model = st.sidebar.radio(
        "Select Model Architecture:",
        ["TF-IDF", "LSTM"],
        help="Toggle between TF-IDF (N-gram Union) and LSTM representations."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Corpus**: Mendeley Bangla Daraz ABSA\n\n"
        f"**Total Reviews**: {len(dataset_df):,} rows\n\n"
        f"**Aspects**: 5 Dimensions\n(Quality, Price, Delivery, Packaging, Seller)"
    )

    # 1. REVIEW ANALYZER
    if view_mode == "🔍 Review Analyzer":
        st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Hierarchical ABSA live inference via <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

        if models_data["status"] != "ready":
            st.error(f"Models missing: {models_data.get('message')}. Please run `python train_models.py` first.")
            st.stop()

        use_lstm = (selected_model == "LSTM")
        active = models_data["lstm"] if use_lstm else models_data["tfidf"]

        presets = {
            "— choose sample preset —": "",
            "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
            "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
            "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
            "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
            "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল।"
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

                        if use_lstm:
                            s_res = predict_sentiment(
                                user_text, 
                                active["sentiment_model"], 
                                use_lstm=True, 
                                vocab=active["vocab"]
                            )
                            a_res = predict_hierarchical(
                                user_text,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
                                binarizer=active["aspect_binarizer"],
                                use_lstm=True,
                                vocab=active["vocab"]
                            )
                        else:
                            s_res = predict_sentiment(
                                user_text,
                                active["sentiment_model"],
                                vectorizer=active["sentiment_vectorizer"],
                                use_lstm=False
                            )
                            a_res = predict_hierarchical(
                                user_text,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
                                vectorizer=active["aspect_vectorizer"],
                                binarizer=active["aspect_binarizer"],
                                use_lstm=False
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
                                f'<div class="aspect-name">{asp}</div>'
                                f'<div class="aspect-pill" style="color: {color}; background-color: {bg_color};">'
                                f'{icon} {pol} ({conf:.0f}%)</div>'
                                f'</div>'
                            )
                        st.markdown(asp_html, unsafe_allow_html=True)
                    else:
                        st.info("No specific product aspects detected in this review.")

                    st.markdown("</div></div>", unsafe_allow_html=True)

    # 2. BENCHMARKS
    elif view_mode == "📈 Benchmarks & Data Insights":
        st.markdown('<div class="main-header">Model Benchmarks & Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluation metrics across TF-IDF and LSTM pipelines</div>', unsafe_allow_html=True)

        if not comparison_df.empty:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                    "Macro F1": st.column_config.NumberColumn(format="%.4f"),
                    "Weighted F1": st.column_config.NumberColumn(format="%.4f"),
                }
            )

            if px is not None:
                st.markdown("---")
                st.subheader("Performance Comparison (F1 Score)")
                
                chart_df = comparison_df[comparison_df["Task"].isin(["Sentiment Analysis", "Aspect Detection"])]
                if not chart_df.empty:
                    fig = px.bar(
                        chart_df, x="Task", y="Macro F1", color="Model", barmode="group",
                        color_discrete_sequence=["#3B82F6", "#10B981"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Benchmark results not found. Please run the training pipeline first.")

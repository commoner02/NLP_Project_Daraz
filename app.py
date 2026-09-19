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


if st is not None:
    # Page Configuration
    st.set_page_config(
        page_title="Daraz ABSA Analytics",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # State init
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "TF-IDF"
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # Custom Styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header { 
            font-size: 2.2rem; 
            font-weight: 800; 
            color: #1E293B;
            margin-bottom: 0.2rem; 
        }
        
        .kpi-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px 24px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
        }
        .kpi-title { font-size: 0.95rem; font-weight: 600; color: #475569; margin-bottom: 12px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;}
        .kpi-value { font-size: 1.6rem; font-weight: 700; color: #1E293B; display: flex; align-items: center; gap: 8px; }
        
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
        }
        .aspect-name { font-weight: 600; color: #334155; font-size: 0.95rem; }
        .aspect-pill { 
            font-weight: 600; 
            font-size: 0.85rem; 
            padding: 6px 12px; 
            border-radius: 6px;
        }
    </style>
    """, unsafe_allow_html=True)

    load_models_cached = st.cache_resource(load_all_models)
    load_benchmarks_cached = st.cache_data(load_benchmarks)

    models_data = load_models_cached()
    comparison_df = load_benchmarks_cached()

    st.markdown('<div class="main-header">Daraz Review Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab_dash, tab_settings = st.tabs(["Dashboard", "Settings"])

    with tab_settings:
        st.subheader("Model Configuration")
        st.radio(
            "Select Model Architecture:",
            ["TF-IDF", "LSTM"],
            key="selected_model",
            help="Toggle between TF-IDF (N-gram Union) and LSTM representations."
        )

        st.markdown("---")
        st.subheader("Model Benchmarks & Insights")
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
                st.write("**Performance Comparison (F1 Score)**")
                chart_df = comparison_df[comparison_df["Task"].isin(["Sentiment Analysis", "Aspect Detection"])]
                if not chart_df.empty:
                    fig = px.bar(
                        chart_df, x="Task", y="Macro F1", color="Model", barmode="group",
                        color_discrete_sequence=["#3B82F6", "#10B981"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Benchmark results not found. Please run the training pipeline first.")

    with tab_dash:
        if models_data["status"] != "ready":
            st.error(f"Models missing: {models_data.get('message')}. Please run `python train_models.py` first.")
            st.stop()

        presets = {
            "— choose sample preset —": "",
            "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
            "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
            "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
            "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
            "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল।"
        }

        # Preset selection logic
        def apply_preset():
            if st.session_state.preset_select != "— choose sample preset —":
                st.session_state.user_text = presets[st.session_state.preset_select]
                
        st.selectbox(
            "Presets", 
            list(presets.keys()), 
            key="preset_select", 
            on_change=apply_preset,
            label_visibility="collapsed"
        )

        user_input = st.text_area(
            "Input",
            value=st.session_state.user_text,
            height=120,
            key="user_text",
            placeholder="Enter review text here..."
        )

        c_analyze, c_reset = st.columns([1, 1])
        with c_analyze:
            analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
        with c_reset:
            if st.button("Reset", use_container_width=True):
                st.session_state.user_text = ""
                st.session_state.preset_select = "— choose sample preset —"
                st.session_state.analysis_results = None
                st.rerun()

        if analyze_clicked:
            if not user_input.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                use_lstm = (st.session_state.selected_model == "LSTM")
                active = models_data["lstm"] if use_lstm else models_data["tfidf"]
                try:
                    with st.spinner(f"Analyzing with {st.session_state.selected_model}..."):
                        cleaned = clean_text(user_input)

                        if use_lstm:
                            s_res = predict_sentiment(
                                user_input, 
                                active["sentiment_model"], 
                                use_lstm=True, 
                                vocab=active["vocab"]
                            )
                            a_res = predict_hierarchical(
                                user_input,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
                                binarizer=active["aspect_binarizer"],
                                use_lstm=True,
                                vocab=active["vocab"]
                            )
                        else:
                            s_res = predict_sentiment(
                                user_input,
                                active["sentiment_model"],
                                vectorizer=active["sentiment_vectorizer"],
                                use_lstm=False
                            )
                            a_res = predict_hierarchical(
                                user_input,
                                active["aspect_model"],
                                polarity_models=active.get("polarity_models", {}),
                                vectorizer=active["aspect_vectorizer"],
                                binarizer=active["aspect_binarizer"],
                                use_lstm=False
                            )
                        st.session_state.analysis_results = (s_res, a_res)
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")

        if st.session_state.analysis_results is not None:
            s_res, a_res = st.session_state.analysis_results
            
            st.markdown("<br/>", unsafe_allow_html=True)
            
            # Overall Sentiment Section
            s_lbl = s_res["sentiment"]
            s_cls = "sentiment-pos" if s_lbl == "Positive" else ("sentiment-neg" if s_lbl == "Negative" else "sentiment-neu")
            
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Overall Sentiment & Confidence</div>
                <div class="kpi-value {s_cls}">{s_lbl} <span style="font-size: 0.9rem; color: #64748B; font-weight: 500;">({s_res['confidence']*100:.1f}%)</span></div>
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
                    fig_s.update_layout(height=120, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, xaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.bar_chart(prob_df.set_index("Sentiment")["Percentage"])
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Aspect Breakdown Section
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-title">Aspect Breakdown</div>
            """, unsafe_allow_html=True)

            aspect_details = a_res.get("aspect_details", [])
            if aspect_details:
                asp_html = ""
                for item in aspect_details:
                    asp = item["aspect"]
                    pol = item["polarity"]
                    conf = item["confidence"] * 100
                    color = item["color"]
                    bg_color = item.get("bg_color", "#ECFDF5" if pol == "Positive" else "#FEF2F2")
                    asp_html += (
                        f'<div class="aspect-row">'
                        f'<div class="aspect-name">{asp}</div>'
                        f'<div class="aspect-pill" style="color: {color}; background-color: {bg_color};">'
                        f'{pol} ({conf:.0f}%)</div>'
                        f'</div>'
                    )
                st.markdown(asp_html, unsafe_allow_html=True)
            else:
                st.write("No specific product aspects detected in this review.")

            st.markdown("</div>", unsafe_allow_html=True)

import os
import sys
from typing import Any, Dict, Optional
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

sys.path.insert(0, os.path.abspath("."))

from src.config import RESULTS_DIR
from src.persistence import (
    load_artifacts,
    load_polarity_artifacts,
)
from src.predict import (
    predict_hierarchical,
    predict_sentiment,
)
from src.preprocessing import clean_text


@st.cache_resource
def get_model_bundle(family: str) -> Dict[str, Any]:
    """Load only the selected model family artifacts."""
    polarity_models = load_polarity_artifacts("tfidf")
    if family == "TF-IDF":
        s_m, s_v, _, _ = load_artifacts("sentiment", "tfidf")
        a_m, a_v, a_b, _ = load_artifacts("issue", "tfidf")
        return {
            "sentiment_model": s_m,
            "sentiment_vectorizer": s_v,
            "aspect_model": a_m,
            "aspect_vectorizer": a_v,
            "aspect_binarizer": a_b,
            "polarity_models": polarity_models,
            "vocab": None,
        }

    s_m, _, _, s_v = load_artifacts("sentiment", "lstm")
    a_m, _, a_b, _ = load_artifacts("issue", "lstm")
    return {
        "sentiment_model": s_m,
        "sentiment_vectorizer": None,
        "aspect_model": a_m,
        "aspect_vectorizer": None,
        "aspect_binarizer": a_b,
        "polarity_models": polarity_models,
        "vocab": s_v,
    }


def load_dataset() -> pd.DataFrame:
    """Load preprocessed ABSA dataset."""
    from src.data_processing import load_data
    try:
        return load_data()
    except Exception:
        return pd.DataFrame()


def load_benchmarks() -> pd.DataFrame:
    """Load model comparison benchmarks."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        try:
            return pd.read_csv(comp_path)
        except Exception:
            pass
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(
        page_title="Bangla Daraz ABSA Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styling
    st.markdown("""
    <style>
        .main-header { font-size: 2.1rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
        .kpi-card { 
            background: #FFFFFF; 
            border: 1px solid #E2E8F0; 
            border-radius: 12px; 
            padding: 1.25rem; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .kpi-title { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
        .kpi-value { font-size: 1.6rem; font-weight: 800; margin-top: 0.25rem; }
        .sentiment-pos { color: #10B981; }
        .sentiment-neg { color: #EF4444; }
        .sentiment-neu { color: #F59E0B; }
        .aspect-row { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 8px 12px; 
            background: #F8FAFC; 
            border: 1px solid #E2E8F0; 
            border-radius: 8px; 
            margin-bottom: 6px; 
        }
        .aspect-name { font-weight: 700; color: #1E293B; font-size: 0.95rem; }
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

    load_data_cached = st.cache_data(load_dataset)
    load_benchmarks_cached = st.cache_data(load_benchmarks)

    dataset_df = load_data_cached()
    comparison_df = load_benchmarks_cached()

    # Sidebar
    st.sidebar.markdown("<h2>Daraz ABSA</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='color: #64748B; font-weight: 500; margin-top: -10px; margin-bottom: 20px;'>Sentiment & Aspect NLP Platform</div>", unsafe_allow_html=True)

    view_mode = st.sidebar.radio(
        "Navigation",
        ["Review Analyzer", "Benchmarks & Data Insights"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Pipeline")
    selected_model = str(st.sidebar.radio(
        "Select Model Architecture:",
        ["TF-IDF", "LSTM"],
        help="Toggle between TF-IDF (N-gram Union) and LSTM representations."
    ) or "TF-IDF")

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Corpus**: Mendeley Bangla Daraz ABSA\n\n"
        f"**Total Reviews**: {len(dataset_df):,} rows\n\n"
        f"**Aspects**: 5 Dimensions\n(Quality, Price, Delivery, Packaging, Seller)"
    )

    # 1. REVIEW ANALYZER
    if view_mode == "Review Analyzer":
        st.markdown('<div class="main-header">Bangla Review Sentiment & Aspect Analyzer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">Hierarchical ABSA live inference via <b>{selected_model}</b> pipeline.</div>', unsafe_allow_html=True)

        try:
            active = get_model_bundle(selected_model)
        except Exception as e:
            st.error(f"Models missing: {str(e)}. Please run `pipeline.ipynb` first.")
            st.stop()

        presets = {
            "— Choose Sample Preset —": "",
            "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
            "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
            "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
            "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
            "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল।"
        }

        selected_preset = st.selectbox("Quick Test Presets:", list(presets.keys()))
        default_text = presets.get(selected_preset or "", "")

        user_text = st.text_area(
            "Enter Bangla Review Text:",
            value=default_text,
            height=95,
            placeholder="দারাজ রিভিউ এখানে লিখুন... (e.g. প্রোডাক্ট ভালো ছিলো কিন্তু ডেলিভারি দেরি হয়েছে)"
        )

        if st.button("Analyze Review", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text before analyzing.")
            else:
                try:
                    with st.spinner(f"Analyzing with {selected_model}..."):
                        s_res = predict_sentiment(
                            user_text,
                            active["sentiment_model"],
                            vectorizer=active.get("sentiment_vectorizer"),
                            vocab=active.get("vocab"),
                        )
                        pol_map: Dict[str, Dict[str, Any]] = active.get("polarity_models") or {}
                        asp_bin: Optional[MultiLabelBinarizer] = active.get("aspect_binarizer")
                        a_res = predict_hierarchical(
                            user_text,
                            active["aspect_model"],
                            polarity_models=pol_map,
                            vectorizer=active.get("aspect_vectorizer"),
                            binarizer=asp_bin,
                            vocab=active.get("vocab"),
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

                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Overall Predicted Sentiment</div>
                        <div class="kpi-value {s_cls}">{s_lbl}</div>
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
                        st.plotly_chart(fig_s, use_container_width=True)

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
                        st.info("No specific product aspects detected in this review.")

                    st.markdown("</div></div>", unsafe_allow_html=True)

    # 2. BENCHMARKS
    elif view_mode == "Benchmarks & Data Insights":
        st.markdown('<div class="main-header">Model Benchmarks & Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluation metrics across TF-IDF and LSTM pipelines</div>', unsafe_allow_html=True)

        if not comparison_df.empty:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Benchmark comparison file `model_comparison.csv` not found. Please run the training pipeline first.")

        st.markdown("---")
        st.subheader("Dataset Distribution & Insights")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### Sentiment Class Distribution")
            if "sentiment" in dataset_df.columns:
                sent_counts = dataset_df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Count"]
                fig_dist = px.pie(
                    sent_counts, values="Count", names="Sentiment", hole=0.4,
                    color="Sentiment",
                    color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}
                )
                fig_dist.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_dist, use_container_width=True)

        with col_d2:
            st.markdown("#### Top Detected Review Aspects")
            if "aspects" in dataset_df.columns:
                all_asps = [asp for sublist in dataset_df["aspects"] for asp in sublist]
                asp_counts = pd.Series(all_asps).value_counts().reset_index()
                asp_counts.columns = ["Aspect", "Count"]
                fig_asp = px.bar(
                    asp_counts, x="Count", y="Aspect", orientation="h",
                    color="Count", color_continuous_scale="Teal"
                )
                fig_asp.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                st.plotly_chart(fig_asp, use_container_width=True)


if __name__ == "__main__":
    main()

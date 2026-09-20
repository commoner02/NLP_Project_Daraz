import os
import sys
from typing import Any, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
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


def load_benchmarks() -> pd.DataFrame:
    """Load model comparison benchmarks."""
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        try:
            df = pd.read_csv(comp_path)
            df.columns = ["Task", "Model", "Accuracy", "Macro F1", "Weighted F1", "Additional"]
            return df
        except Exception:
            pass
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(
        page_title="Daraz Review Analyzer",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom Clean CSS Styling
    st.markdown("""
    <style>
        /* Center container max-width */
        .main .block-container {
            max-width: 780px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        
        /* App Title */
        .app-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: #1E293B;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        /* Active model badge */
        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            color: #64748B;
            margin-bottom: 1.2rem;
        }
        .model-pill {
            font-weight: 600;
            color: #2563EB;
            background: #EFF6FF;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #DBEAFE;
        }

        /* Section Headings */
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1E293B;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        /* Card Container */
        .result-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 14px;
            margin-bottom: 14px;
        }
        .card-header {
            font-size: 0.92rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 8px;
        }
        .card-divider {
            border: 0;
            border-top: 1px solid #E2E8F0;
            margin: 0 0 14px 0;
        }

        /* Aspect item row */
        .aspect-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: #FAFAFA;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .aspect-label {
            font-size: 0.92rem;
            font-weight: 600;
            color: #1E293B;
        }
        .aspect-pill {
            font-size: 0.82rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
        }

        /* Streamlit Button & Tabs Customization */
        div[data-testid="stTabs"] button {
            font-size: 0.95rem;
            font-weight: 500;
            color: #64748B;
            padding: 8px 16px;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #2563EB;
            font-weight: 600;
            border-bottom: 2px solid #2563EB;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="app-title">Daraz Review Analyzer</div>', unsafe_allow_html=True)

    PRESETS = {
        "Damaged Packaging & Good Product": "প্রোডাক্ট ভালো কিন্তু প্যাকেজিং নষ্ট ছিল।",
        "Positive Quality & Fast Delivery": "প্রোডাক্ট খুব ভালো ছিল, ডেলিভারিও দ্রুত পেয়েছি। ধন্যবাদ।",
        "Negative Delay & Poor Quality": "ডেলিভারি অনেক দেরি হয়েছে, প্রোডাক্টও বাজে কোয়ালিটি।",
        "Negation Test (Poor Battery, Good Sound)": "সাউন্ড কোয়ালিটি ভালো কিন্তু ব্যাটারি ভালো না একদমই।",
        "Price Concern & Seller Service": "দাম অনেক বেশি কিন্তু সেলার খুব হেল্পফুল ছিল।",
        "Custom / Clear": "",
    }

    # Initialize Session State
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "TF-IDF"
    if "user_review_text" not in st.session_state:
        st.session_state["user_review_text"] = PRESETS["Damaged Packaging & Good Product"]
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None

    # Callbacks to manage widget state safely before rerun
    def on_preset_change():
        chosen = st.session_state.get("preset_selector")
        if chosen in PRESETS:
            st.session_state["user_review_text"] = PRESETS[chosen]
            st.session_state["analysis_results"] = None

    def on_reset():
        st.session_state["user_review_text"] = ""
        st.session_state["analysis_results"] = None
        st.session_state["preset_selector"] = "Custom / Clear"

    def on_model_change():
        st.session_state["analysis_results"] = None

    tab_dash, tab_settings = st.tabs(["Dashboard", "Settings"])

    # =========================================================================
    # TAB 1: DASHBOARD
    # =========================================================================
    with tab_dash:
        current_model = st.session_state.get("selected_model", "TF-IDF")
        st.markdown(
            f'<div class="model-badge">Active Model: <span class="model-pill">{current_model}</span></div>',
            unsafe_allow_html=True
        )

        st.selectbox(
            "Quick Test Presets",
            options=list(PRESETS.keys()),
            key="preset_selector",
            on_change=on_preset_change,
            label_visibility="collapsed",
        )

        st.text_area(
            "Review Input",
            key="user_review_text",
            height=100,
            placeholder="দারাজ রিভিউ এখানে লিখুন...",
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
        with col_btn2:
            st.button("Reset", on_click=on_reset, use_container_width=True)

        if analyze_clicked:
            raw_text = st.session_state.get("user_review_text", "").strip()
            if not raw_text:
                st.warning("Please enter review text before analyzing.")
            else:
                try:
                    active = get_model_bundle(st.session_state["selected_model"])
                    with st.spinner("Analyzing review..."):
                        s_res = predict_sentiment(
                            raw_text,
                            active["sentiment_model"],
                            vectorizer=active.get("sentiment_vectorizer"),
                            vocab=active.get("vocab"),
                        )
                        pol_map: Dict[str, Dict[str, Any]] = active.get("polarity_models") or {}
                        asp_bin: Optional[MultiLabelBinarizer] = active.get("aspect_binarizer")
                        a_res = predict_hierarchical(
                            raw_text,
                            active["aspect_model"],
                            polarity_models=pol_map,
                            vectorizer=active.get("aspect_vectorizer"),
                            binarizer=asp_bin,
                            vocab=active.get("vocab"),
                        )
                        st.session_state["analysis_results"] = {
                            "sentiment": s_res,
                            "aspects": a_res,
                        }
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")

        # Display Results if available
        results = st.session_state.get("analysis_results")
        if results:
            s_res = results["sentiment"]
            a_res = results["aspects"]

            s_lbl = s_res["sentiment"]
            s_conf = s_res["confidence"] * 100
            s_color = "#10B981" if s_lbl == "Positive" else ("#EF4444" if s_lbl == "Negative" else "#F59E0B")

            # --- Card 1: Overall Sentiment & Confidence ---
            st.markdown(
                f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px 20px; margin-top: 14px; margin-bottom: 14px;">'
                f'<div style="font-size: 0.92rem; font-weight: 600; color: #475569; margin-bottom: 8px;">Overall Sentiment & Confidence</div>'
                f'<hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 0 0 14px 0;" />'
                f'<div style="font-size: 1.25rem; font-weight: 700; color: {s_color}; margin-bottom: 4px;">'
                f'{s_lbl} ({s_conf:.1f}%)'
                f'</div></div>',
                unsafe_allow_html=True
            )

            if s_res.get("probabilities"):
                # Bar Chart: Negative, Neutral, Positive
                probs = s_res["probabilities"]
                categories = ["Negative", "Neutral", "Positive"]
                values = [probs.get(cat, 0.0) * 100 for cat in categories]
                colors = ["#EF4444", "#F59E0B", "#10B981"]

                fig_sent = go.Figure(
                    go.Bar(
                        x=values,
                        y=categories,
                        orientation="h",
                        marker=dict(color=colors),
                        text=[f"{v:.1f}%" for v in values],
                        textposition=["inside" if v > 15 else "outside" for v in values],
                        insidetextanchor="middle",
                        textfont=dict(color=["white" if v > 15 else "#475569" for v in values], size=11, family="sans-serif"),
                    )
                )
                fig_sent.update_layout(
                    height=160,
                    margin=dict(t=5, b=25, l=60, r=20),
                    xaxis=dict(
                        title="Percentage",
                        range=[0, 100],
                        tickmode="linear",
                        tick0=0,
                        dtick=20,
                        showgrid=True,
                        gridcolor="#F1F5F9"
                    ),
                    yaxis=dict(
                        title="Sentiment",
                        autorange="reversed"
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_sent, use_container_width=True)

            # --- Card 2: Aspect Breakdown ---
            aspect_details = a_res.get("aspect_details", [])
            asp_items = []
            if aspect_details:
                for item in aspect_details:
                    asp_name = item["aspect"]
                    asp_pol = item["polarity"]
                    asp_conf = item["confidence"] * 100
                    p_col = "#10B981" if asp_pol == "Positive" else "#EF4444"
                    p_bg = "#ECFDF5" if asp_pol == "Positive" else "#FEF2F2"
                    asp_items.append(
                        f'<div class="aspect-item" style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #FAFAFA; border: 1px solid #E2E8F0; border-radius: 6px; margin-bottom: 8px;">'
                        f'<div class="aspect-label" style="font-size: 0.92rem; font-weight: 600; color: #1E293B;">{asp_name}</div>'
                        f'<div class="aspect-pill" style="font-size: 0.82rem; font-weight: 600; padding: 4px 10px; border-radius: 9999px; color: {p_col}; background-color: {p_bg};">'
                        f'{asp_pol} ({asp_conf:.0f}%)'
                        f'</div></div>'
                    )
                asp_content = "".join(asp_items)
            else:
                asp_content = '<div style="color: #64748B; font-size: 0.9rem; padding: 6px 0;">No specific product aspects detected in this review.</div>'

            st.markdown(
                f'<div class="result-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px 20px; margin-top: 14px; margin-bottom: 14px;">'
                f'<div class="card-header" style="font-size: 0.92rem; font-weight: 600; color: #475569; margin-bottom: 8px;">Aspect Breakdown</div>'
                f'<hr class="card-divider" style="border: 0; border-top: 1px solid #E2E8F0; margin: 0 0 14px 0;" />'
                f'{asp_content}'
                f'</div>',
                unsafe_allow_html=True
            )

    # =========================================================================
    # TAB 2: SETTINGS & MODEL CONFIGURATION
    # =========================================================================
    with tab_settings:
        st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

        st.radio(
            "Select Model Architecture:",
            options=["TF-IDF", "LSTM"],
            key="selected_model",
            on_change=on_model_change,
            help="Choose between TF-IDF (N-gram feature union) and LSTM (PyTorch sequential deep model)."
        )

        st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 24px 0;' />", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Model Benchmarks & Insights</div>', unsafe_allow_html=True)

        comparison_df = load_benchmarks()
        if not comparison_df.empty:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=False
            )
        else:
            st.info("No benchmark data available. Please run model training.")

        st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 24px 0;' />", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="font-size: 1.05rem;">Performance Comparison (F1 Score)</div>', unsafe_allow_html=True)

        # Dynamically extract Macro F1 values from comparison_df if available
        s_tfidf_f1 = 0.8490
        s_lstm_f1 = 0.7846
        a_tfidf_f1 = 0.8038
        a_lstm_f1 = 0.4986

        if not comparison_df.empty and "Task" in comparison_df.columns:
            try:
                s_t = comparison_df[(comparison_df["Task"] == "Sentiment Analysis") & (comparison_df["Model"].str.contains("TF-IDF"))]
                if not s_t.empty:
                    s_tfidf_f1 = float(s_t.iloc[0]["Macro F1"])
                s_l = comparison_df[(comparison_df["Task"] == "Sentiment Analysis") & (comparison_df["Model"].str.contains("LSTM"))]
                if not s_l.empty:
                    s_lstm_f1 = float(s_l.iloc[0]["Macro F1"])
                a_t = comparison_df[(comparison_df["Task"] == "Aspect Detection") & (comparison_df["Model"].str.contains("TF-IDF"))]
                if not a_t.empty:
                    a_tfidf_f1 = float(a_t.iloc[0]["Macro F1"])
                a_l = comparison_df[(comparison_df["Task"] == "Aspect Detection") & (comparison_df["Model"].str.contains("LSTM"))]
                if not a_l.empty:
                    a_lstm_f1 = float(a_l.iloc[0]["Macro F1"])
            except Exception:
                pass

        tasks = ["Sentiment Analysis", "Aspect Detection"]
        fig_comp = go.Figure(data=[
            go.Bar(
                name="TF-IDF",
                x=tasks,
                y=[s_tfidf_f1, a_tfidf_f1],
                marker_color="#3B82F6",
                text=[f"{s_tfidf_f1:.4f}", f"{a_tfidf_f1:.4f}"],
                textposition="auto"
            ),
            go.Bar(
                name="PyTorch BiLSTM",
                x=tasks,
                y=[s_lstm_f1, a_lstm_f1],
                marker_color="#10B981",
                text=[f"{s_lstm_f1:.4f}", f"{a_lstm_f1:.4f}"],
                textposition="auto"
            )
        ])

        fig_comp.update_layout(
            barmode="group",
            height=340,
            margin=dict(t=15, b=30, l=40, r=20),
            xaxis=dict(title="Task"),
            yaxis=dict(title="Macro F1", range=[0, 1.0]),
            legend=dict(
                title=dict(text="Model Family"),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_comp, use_container_width=True)


if __name__ == "__main__":
    main()

"""
Cocoa Disease Detection Web Application
Production UX/UI Dashboard — Agricultural Diagnostic System
"""

import base64
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import pandas as pd
import altair as alt
import streamlit as st

from src.predictor import CocoaDiseasePredictor


# Page Configuration
LOGO_PATH = Path("assets/cocoa_logo.png")
SVG_LOGO_PATH = Path("assets/cocoa_logo.svg")

st.set_page_config(
    page_title="Cocoa Disease Detection",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_svg_as_base64(svg_path: Path) -> str:
    """Read SVG file and return base64 encoded data URI."""
    if svg_path.exists():
        with open(svg_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/svg+xml;base64,{encoded}"
    return ""


# Tailored Design System: Matching Production Dashboard Mockup
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --color-brand-primary: #1E4620;
        --color-brand-dark: #143516;
        --color-brand-light: #2D6A4F;
        --color-cocoa: #4A2B18;
        --color-bg-canvas: #EFF2F5;
        --color-card-bg: #FFFFFF;
        --color-sidebar-bg: #F4F6F8;
        --color-border: #E2E6EA;
        --color-border-dashed: #2D6A4F;
        --color-text-title: #1A1F24;
        --color-text-body: #495057;
        --color-text-muted: #6C757D;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--color-text-body);
    }

    .stApp {
        background-color: var(--color-bg-canvas);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--color-sidebar-bg);
        border-right: 1px solid var(--color-border);
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.8rem 0.2rem 1.4rem 0.2rem;
        border-bottom: 1px solid var(--color-border);
        margin-bottom: 1.2rem;
    }

    .sidebar-logo {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: #FFFFFF;
        border: 1px solid var(--color-border);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 4px;
    }

    .sidebar-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    .sidebar-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--color-text-title);
        line-height: 1.2;
    }

    .sidebar-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-top: 1.2rem;
        margin-top: 2rem;
        border-top: 1px solid var(--color-border);
    }

    .user-avatar-badge {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #2D6A4F;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .version-tag {
        font-size: 0.78rem;
        color: var(--color-text-muted);
        font-weight: 600;
    }

    /* Main Dashboard Header Card */
    .dashboard-header-card {
        background: var(--color-card-bg);
        border: 1px solid var(--color-border);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }

    .dashboard-header-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: var(--color-text-title);
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .dashboard-header-subtitle {
        font-size: 0.98rem;
        color: var(--color-text-muted);
        margin: 0;
        line-height: 1.5;
        max-width: 920px;
    }

    /* Section Container Card */
    .content-panel-card {
        background: var(--color-card-bg);
        border: 1px solid var(--color-border);
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 480px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        display: flex;
        flex-direction: column;
    }

    .section-numbered-header {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--color-text-title);
        margin-bottom: 1.2rem;
    }

    .number-badge {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background-color: #E9ECEF;
        color: #212529;
        font-size: 0.85rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Custom Dashed Uploader Accent */
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    div[data-testid="stFileUploader"] section {
        border: 2px dashed #3B7A57 !important;
        background-color: #F6FBF7 !important;
        border-radius: 12px !important;
        padding: 1.4rem !important;
        transition: all 0.2s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
        background-color: #EEF8F0 !important;
        border-color: #1E4620 !important;
    }

    /* Placeholder Box matching Mockup */
    .placeholder-panel {
        background-color: #F8F9FA;
        border: 1px solid var(--color-border);
        border-radius: 12px;
        padding: 3rem 1.5rem;
        text-align: center;
        color: var(--color-text-muted);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex-grow: 1;
        margin-top: 1rem;
    }

    .placeholder-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        opacity: 0.75;
    }

    .placeholder-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #343A40;
        margin-bottom: 0.35rem;
    }

    .placeholder-subtitle {
        font-size: 0.86rem;
        color: #6C757D;
        line-height: 1.45;
        max-width: 380px;
    }

    /* Diagnostic Result Presentation */
    .result-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .pill-healthy {
        background-color: #E8F5E9;
        color: #1B5E20;
        border: 1px solid #A5D6A7;
    }

    .pill-disease {
        background-color: #FFEBEE;
        color: #B71C1C;
        border: 1px solid #FFCDD2;
    }

    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1A1F24;
        margin: 0.6rem 0 0.2rem 0;
    }

    .result-score-row {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .result-score-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1E4620;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #1E4620 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #153417 !important;
    }

    div.stButton > button[kind="secondary"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading diagnostic model...")
def get_predictor() -> CocoaDiseasePredictor:
    """Load model and metadata once on startup."""
    return CocoaDiseasePredictor(device="cpu")


def render_sidebar(svg_data_uri: str) -> str:
    """Render the left navigation sidebar matching the UI mockup."""
    with st.sidebar:
        logo_tag = f'<img src="{svg_data_uri}" alt="Logo">' if svg_data_uri else "🍫"
        st.markdown(
            f"""
            <div class="sidebar-header">
                <div class="sidebar-logo">
                    {logo_tag}
                </div>
                <div class="sidebar-title">
                    Cocoa Disease<br>Detection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_choice = st.radio(
            "Navigation",
            ["Disease Analysis", "Data Explorer", "About System", "Settings"],
            index=0,
            label_visibility="collapsed",
            format_func=lambda x: {
                "Disease Analysis": "🌿  Disease Analysis",
                "Data Explorer": "📊  Data Explorer",
                "About System": "ℹ️  About System",
                "Settings": "⚙️  Settings",
            }[x],
        )

        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-footer">
                <div class="user-avatar-badge">U</div>
                <div class="version-tag">v1.2.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return nav_choice


def build_horizontal_bar_chart(data: list, predicted_class: str, is_healthy: bool) -> alt.Chart:
    """Render a clean horizontal bar chart for all 9 classes."""
    df = pd.DataFrame(data)
    highlight_color = "#1E4620" if is_healthy else "#A82A2A"
    neutral_color = "#D1D5DB"

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
        .encode(
            x=alt.X(
                "percentage:Q",
                title="Probability (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(grid=True, tickCount=5, format=".0f"),
            ),
            y=alt.Y(
                "display_name:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelLimit=240),
            ),
            color=alt.condition(
                alt.datum.class_name == predicted_class,
                alt.value(highlight_color),
                alt.value(neutral_color),
            ),
            tooltip=[
                alt.Tooltip("display_name:N", title="Category"),
                alt.Tooltip("percentage:Q", title="Probability", format=".2f"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
    )
    return chart


def main() -> None:
    predictor = get_predictor()
    svg_uri = load_svg_as_base64(SVG_LOGO_PATH)

    active_page = render_sidebar(svg_uri)

    # Top Header Card (Present across all views)
    logo_header_tag = (
        f'<img src="{svg_uri}" style="width: 28px; height: 28px; vertical-align: middle;">'
        if svg_uri
        else "🍫"
    )

    st.markdown(
        f"""
        <div class="dashboard-header-card">
            <h1 class="dashboard-header-title">
                Cocoa Disease Detection {logo_header_tag}
            </h1>
            <p class="dashboard-header-subtitle">
                Agricultural plant pathology diagnostic system. Upload a photograph of a cocoa pod or leaf 
                to evaluate symptoms against 9 recognized disease and health categories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if active_page == "Disease Analysis":
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(
                """
                <div class="section-numbered-header">
                    <span class="number-badge">1</span>
                    <span>Upload Cocoa Specimen</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Drag and drop file here",
                type=["jpg", "jpeg", "png", "webp"],
                help="Limit 200MB per file • JPEG, PNG, WEBP (Production Recommended: <15MB)",
                label_visibility="collapsed",
                key="file_uploader_specimen",
            )

            # Invalidate cached prediction when new file is chosen
            file_sig = f"{uploaded_file.name}_{uploaded_file.size}" if uploaded_file else None
            if st.session_state.get("active_file_sig") != file_sig:
                st.session_state["active_file_sig"] = file_sig
                st.session_state["prediction"] = None

            loaded_image: Optional[Image.Image] = None

            if uploaded_file is not None:
                if uploaded_file.size > 200 * 1024 * 1024:
                    st.error("Uploaded file exceeds 200 MB maximum threshold.")
                else:
                    try:
                        loaded_image = predictor.preprocessor.load_image(uploaded_file)
                    except Exception as err:
                        st.error(f"Image error: {err}")

            if loaded_image is not None and uploaded_file is not None:
                st.image(
                    loaded_image,
                    caption=f"{uploaded_file.name} ({loaded_image.width}×{loaded_image.height} px)",
                    use_container_width=True,
                )

                btn_c1, btn_c2 = st.columns([1, 1])
                with btn_c1:
                    run_analyse = st.button(
                        "Analyse Image",
                        type="primary",
                        use_container_width=True,
                        key="btn_run_prediction",
                    )
                with btn_c2:
                    run_clear = st.button(
                        "Clear Image",
                        type="secondary",
                        use_container_width=True,
                        key="btn_clear_specimen",
                    )

                if run_clear:
                    st.session_state["active_file_sig"] = None
                    st.session_state["prediction"] = None
                    st.rerun()

                if run_analyse:
                    with st.spinner("Analyzing image features with EfficientNet-B0..."):
                        try:
                            st.session_state["prediction"] = predictor.predict(loaded_image)
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
            else:
                st.markdown(
                    """
                    <div class="placeholder-panel">
                        <div class="placeholder-icon">📷</div>
                        <div class="placeholder-title">No image selected</div>
                        <div class="placeholder-subtitle">
                            Upload a high-resolution, well-lit photo of a cocoa pod or leaf.<br>
                            (Supported: JPG, JPEG, PNG, WEBP, Max 15MB)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown(
                """
                <div class="section-numbered-header">
                    <span class="number-badge">2</span>
                    <span>Diagnostic Results</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prediction = st.session_state.get("prediction")

            if prediction is not None:
                pred_class = prediction["predicted_class"]
                display_name = prediction["display_name"]
                conf_pct = prediction["confidence_percentage"]
                is_healthy = prediction["is_healthy"]
                is_uncertain = prediction["is_uncertain"]

                pill_class = "pill-healthy" if is_healthy else "pill-disease"
                pill_label = "HEALTHY SPECIMEN" if is_healthy else "PATHOLOGY DETECTED"

                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E6EA; border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem; border-left: 4px solid {'#1E4620' if is_healthy else '#B71C1C'};">
                        <span class="result-status-pill {pill_class}">{pill_label}</span>
                        <div class="result-title">{display_name}</div>
                        <div class="result-score-row">
                            <span style="font-size: 0.9rem; color: #6C757D;">Model Probability:</span>
                            <span class="result-score-val">{conf_pct:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if is_uncertain:
                    st.warning(
                        f"⚠️ Ambiguous Result ({conf_pct:.1f}%): Highest classification is below {prediction['uncertainty_threshold']*100:.0f}%. "
                        "Symptoms may be atypical or overlapping. Verification by an agricultural specialist is recommended."
                    )

                if prediction.get("description"):
                    with st.expander("Clinical Notes & Symptoms", expanded=True):
                        st.write(prediction["description"])

                st.markdown("<div style='font-size: 0.9rem; font-weight: 700; margin: 0.8rem 0 0.3rem 0; color: #1A1F24;'>Class Probability Breakdown</div>", unsafe_allow_html=True)
                chart_data = [
                    {
                        "class_name": it["class"],
                        "display_name": it["display_name"],
                        "percentage": it["percentage"],
                    }
                    for it in prediction["all_classes_sorted"]
                ]
                chart = build_horizontal_bar_chart(chart_data, pred_class, is_healthy)
                st.altair_chart(chart, use_container_width=True)

            elif loaded_image is not None:
                st.markdown(
                    """
                    <div class="placeholder-panel">
                        <div class="placeholder-icon">🔬</div>
                        <div class="placeholder-title">Ready for analysis</div>
                        <div class="placeholder-subtitle">
                            Specimen loaded. Click <strong>Analyse Image</strong> to begin diagnostic classification.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="placeholder-panel">
                        <div class="placeholder-icon">📋</div>
                        <div class="placeholder-title">No analysis available</div>
                        <div class="placeholder-subtitle">
                            Upload an image in the panel on the left to begin diagnostic classification.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif active_page == "Data Explorer":
        st.subheader("📊 Dataset Explorer & Pathology Distribution")
        st.markdown(
            """
            The model was trained on the **Unified Cocoa Disease Dataset** ([Kaggle](https://www.kaggle.com/datasets/danielohachor/cocoa-disease-unified)).
            Below is the pathology reference table and performance breakdown:
            """
        )

        specs = predictor.get_model_specs()
        metrics = specs.get("evaluation_metrics", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.0)*100:.1f}%")
        c2.metric("Precision (Macro)", f"{metrics.get('test_precision_macro', 0.0)*100:.1f}%")
        c3.metric("Recall (Macro)", f"{metrics.get('test_recall_macro', 0.0)*100:.1f}%")
        c4.metric("F1 Score (Macro)", f"{metrics.get('test_f1_macro', 0.0)*100:.1f}%")

        st.markdown("---")
        st.markdown("#### 9 Categories in the Training Distribution")
        cols = st.columns(3)
        for i, c_key in enumerate(predictor.classes):
            with cols[i % 3]:
                st.markdown(f"**{predictor.CLASS_DISPLAY_NAMES.get(c_key, c_key)}**")
                st.caption(predictor.DISEASE_DESCRIPTIONS.get(c_key, ""))

    elif active_page == "About System":
        st.subheader("ℹ️ About the Cocoa Diagnostic System")
        st.markdown(
            """
            This web application provides computer vision decision support for cacao producers and extension officers.

            - **Deep Learning Model:** Fine-tuned `EfficientNet-B0` architecture
            - **Input Resolution:** `224 × 224` pixels RGB
            - **Normalization:** ImageNet distribution (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
            - **Hosting Platform:** Streamlit Community Cloud
            """
        )

        st.markdown("---")
        st.markdown(
            """
            **⚠️ Agricultural Decision Support Notice:**  
            Automated predictions are intended for preliminary screening. Consult qualified agricultural officers 
            before applying chemical treatments or initiating crop eradication.
            """
        )

    elif active_page == "Settings":
        st.subheader("⚙️ System Preferences")
        thresh = st.slider(
            "Uncertainty Alert Threshold",
            min_value=0.30,
            max_value=0.80,
            value=float(predictor.uncertainty_threshold),
            step=0.05,
            help="Flag predictions with top probabilities below this threshold as ambiguous.",
        )
        predictor.uncertainty_threshold = thresh
        st.success(f"Uncertainty threshold set to {thresh*100:.0f}%.")

        st.markdown("---")
        st.markdown(f"**Device Mode:** `{predictor.device.type.upper()}`")
        st.markdown(f"**Model Path:** `{predictor.model_path}`")


if __name__ == "__main__":
    main()

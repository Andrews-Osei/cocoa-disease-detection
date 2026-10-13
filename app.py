"""
Cocoa Disease Detection Web Application
Built with Streamlit and PyTorch EfficientNet-B0
"""

import io
from pathlib import Path
from typing import Optional
from PIL import Image
import pandas as pd
import altair as alt
import streamlit as st

from src.predictor import CocoaDiseasePredictor


# Page Configuration
st.set_page_config(
    page_title="Cocoa Disease Detection",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Main container and typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #52b788 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(27, 67, 50, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #d8f3dc;
        font-size: 1.05rem;
        margin-bottom: 0;
        max-width: 800px;
        line-height: 1.5;
    }
    
    .metric-card {
        background-color: var(--secondary-background-color, #f8f9fa);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    
    .metric-card h4 {
        margin-top: 0;
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #555;
    }
    
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1b4332;
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    
    .badge-healthy {
        background-color: #d8f3dc;
        color: #1b4332;
        border: 1px solid #74c69d;
    }
    
    .badge-disease {
        background-color: #ffe3e3;
        color: #c92a2a;
        border: 1px solid #ffa8a8;
    }
    
    .badge-warning {
        background-color: #fff3bf;
        color: #e67700;
        border: 1px solid #ffd43b;
    }
    
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .disclaimer-box {
        background-color: #f1f3f5;
        border-left: 4px solid #495057;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #495057;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained PyTorch model...")
def get_predictor() -> CocoaDiseasePredictor:
    """Load model and metadata once on startup."""
    return CocoaDiseasePredictor(device="cpu")


def render_sidebar(predictor: CocoaDiseasePredictor) -> None:
    """Render informative sidebar with model provenance and verified metrics."""
    with st.sidebar:
        st.subheader("🌾 About the Project")
        st.markdown(
            """
            This application uses deep learning (**EfficientNet-B0**) to classify cocoa pod and leaf images 
            into **9 health and disease categories** commonly encountered in cacao farms.
            """
        )

        st.markdown("---")
        st.subheader("📊 Verified Model Specs")
        specs = predictor.get_model_specs()
        metrics = specs.get("evaluation_metrics", {})

        st.markdown(f"**Architecture:** `{specs.get('model_name', 'EfficientNet-B0')}`")
        st.markdown(f"**Inference Device:** `{specs.get('device', 'cpu').upper()}`")
        input_size = specs.get("input_size", (224, 224))
        st.markdown(f"**Input Resolution:** `{input_size[0]} × {input_size[1]} RGB`")
        st.markdown(f"**Categories Recognized:** `{specs.get('num_classes', 9)}`")

        if metrics:
            st.markdown("##### Test Set Evaluation")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics.get('test_accuracy', 0.0) * 100:.1f}%")
                st.metric("Recall", f"{metrics.get('test_recall_macro', 0.0) * 100:.1f}%")
            with col2:
                st.metric("Precision", f"{metrics.get('test_precision_macro', 0.0) * 100:.1f}%")
                st.metric("F1 Score", f"{metrics.get('test_f1_macro', 0.0) * 100:.1f}%")

        st.markdown("---")
        st.subheader("📚 Dataset Source")
        st.markdown(
            """
            Trained on the **Unified Cocoa Disease Dataset**:  
            [Kaggle Dataset Link](https://www.kaggle.com/datasets/danielohachor/cocoa-disease-unified)
            """
        )

        st.markdown("---")
        st.caption(
            "Decision Support System v1.0 • Built with PyTorch & Streamlit Cloud"
        )


def main() -> None:
    predictor = get_predictor()
    render_sidebar(predictor)

    # Application Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🍫 Cocoa Disease Detection</h1>
            <p>
                Upload an image of a cocoa pod or leaf to detect signs of infectious fungal diseases, 
                viral pathogens, pest infestations, or verify plant health in real-time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_predict, tab_library, tab_instructions = st.tabs(
        ["🔍 Disease Diagnosis", "📖 Disease Library (9 Classes)", "🚀 Deployment & Help"]
    )

    with tab_predict:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("1. Provide Cocoa Image")
            
            # Option to choose between file upload or built-in sample images
            input_mode = st.radio(
                "Select input method:",
                ["Upload from device", "Try sample demo images"],
                horizontal=True,
            )

            selected_image: Optional[Image.Image] = None
            image_name = "Image"

            if input_mode == "Upload from device":
                uploaded_file = st.file_uploader(
                    "Choose a JPG, JPEG, or PNG image",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Upload a clear close-up photograph of a cocoa pod or leaf (max 15MB).",
                )
                if uploaded_file is not None:
                    # Validate size (< 15MB)
                    if uploaded_file.size > 15 * 1024 * 1024:
                        st.error("Uploaded file is too large (>15MB). Please upload a smaller image.")
                    else:
                        try:
                            selected_image = predictor.preprocessor.load_image(uploaded_file)
                            image_name = uploaded_file.name
                        except Exception as e:
                            st.error(f"Could not read image: {e}")
            else:
                samples = {
                    "Healthy Cocoa Pod": Path("assets/samples/healthy_pod_sample.jpg"),
                    "Diseased Cocoa Pod": Path("assets/samples/diseased_pod_sample.jpg"),
                }
                sample_choice = st.selectbox("Select a demonstration sample:", list(samples.keys()))
                sample_path = samples[sample_choice]
                if sample_path.exists():
                    selected_image = Image.open(sample_path)
                    image_name = sample_choice

            if selected_image is not None:
                st.image(
                    selected_image,
                    caption=f"Selected: {image_name} ({selected_image.width} × {selected_image.height} px)",
                    use_container_width=True,
                )
                predict_button = st.button("🔬 Predict Disease", type="primary", use_container_width=True)
            else:
                st.info("👆 Please upload an image or choose a demo sample to begin.")
                predict_button = False

        with col_output:
            st.subheader("2. Diagnosis & Prediction Results")

            if selected_image is not None and predict_button:
                with st.spinner("Analyzing image features with EfficientNet-B0..."):
                    try:
                        result = predictor.predict(selected_image)
                        st.session_state["last_result"] = result
                    except Exception as exc:
                        st.error(f"Inference error: {exc}")
                        st.session_state["last_result"] = None

            result = st.session_state.get("last_result")

            if result is not None:
                pred_class = result["predicted_class"]
                display_name = result["display_name"]
                conf_pct = result["confidence_percentage"]
                is_healthy = result["is_healthy"]
                is_uncertain = result["is_uncertain"]

                # Prominent Result Card
                badge_class = "badge-healthy" if is_healthy else "badge-disease"
                badge_text = "HEALTHY POD / LEAF" if is_healthy else "DISEASE / PEST DETECTED"

                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 5px solid {'#2d6a4f' if is_healthy else '#c92a2a'};">
                        <span class="badge {badge_class}">{badge_text}</span>
                        <h2 style="margin-top: 0.5rem; margin-bottom: 0.2rem; color: #1b4332;">{display_name}</h2>
                        <p style="font-size: 1.15rem; margin-bottom: 0;">
                            Estimated Confidence: <strong>{conf_pct:.2f}%</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Confidence progress bar
                st.progress(result["confidence"])

                # Uncertainty Warning if score is under threshold
                if is_uncertain:
                    st.warning(
                        f"⚠️ **Low Confidence Notice ({conf_pct:.1f}%):** The model's highest probability is below "
                        f"{result['uncertainty_threshold']*100:.0f}%. Visual symptoms may be ambiguous or unrepresented. "
                        "We strongly recommend manual agronomic verification."
                    )

                # Agronomic Description
                if result.get("description"):
                    with st.expander("🌾 Clinical Description & Symptoms", expanded=True):
                        st.write(result["description"])

                # Probability Breakdown Chart
                st.markdown("#### Class Probability Distribution")
                df_probs = pd.DataFrame(result["all_classes_sorted"])
                df_probs["Percentage"] = df_probs["probability"] * 100.0

                chart = (
                    alt.Chart(df_probs)
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                    .encode(
                        x=alt.X("Percentage:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("display_name:N", title=None, sort="-x"),
                        color=alt.condition(
                            alt.datum.display_name == display_name,
                            alt.value("#2d6a4f" if is_healthy else "#c92a2a"),
                            alt.value("#adb5bd"),
                        ),
                        tooltip=["display_name", alt.Tooltip("Percentage:Q", format=".2f")],
                    )
                    .properties(height=280)
                )

                st.altair_chart(chart, use_container_width=True)

                if st.button("🔄 Clear and Analyze Another Image"):
                    st.session_state["last_result"] = None
                    st.rerun()

            elif selected_image is not None and not predict_button:
                st.info("Click the **'🔬 Predict Disease'** button on the left to run classification.")
            else:
                st.write("Results and probability distribution will appear here once an image is analyzed.")

    with tab_library:
        st.subheader("📚 Cocoa Pathology Reference Library")
        st.markdown(
            "The model has been trained on the 9 distinct cocoa health conditions and diseases listed below:"
        )
        
        cols = st.columns(3)
        for i, cls_name in enumerate(predictor.classes):
            with cols[i % 3]:
                st.markdown(f"### {predictor.CLASS_DISPLAY_NAMES.get(cls_name, cls_name)}")
                badge = "badge-healthy" if cls_name == "healthy" else "badge-disease"
                st.markdown(
                    f"<span class='badge {badge}'>{'BENIGN' if cls_name == 'healthy' else 'PATHOGEN / PEST'}</span>",
                    unsafe_allow_html=True,
                )
                st.write(predictor.DISEASE_DESCRIPTIONS.get(cls_name, "No description available."))
                st.markdown("---")

    with tab_instructions:
        st.subheader("🚀 Deployment on Streamlit Community Cloud")
        st.markdown(
            """
            This application is ready to deploy directly on **Streamlit Community Cloud**:
            
            1. **Push to GitHub**:
               Ensure your repository contains `app.py`, `requirements.txt`, `src/`, `models/best_model.pth`, and `artifacts/`.
            2. **Log in to Streamlit Cloud**:
               Navigate to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
            3. **Create New App**:
               - Select repository: `<your-username>/cocoa-disease-detection`
               - Branch: `main`
               - Main file path: `app.py`
            4. **Deploy**:
               Click **Deploy!** Streamlit Cloud will install dependencies from `requirements.txt` and launch the app.
            """
        )

    # Persistent Agricultural Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Agricultural Decision Support Disclaimer:</strong><br>
            Predictions generated by this machine learning model are intended solely for preliminary screening 
            and agricultural decision support. Uncalibrated softmax probabilities do not guarantee definitive diagnoses. 
            Always confirm symptoms with physical on-farm inspections and consult qualified agricultural extension 
            specialists or certified plant pathologists prior to initiating pesticide or fungicide treatments.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

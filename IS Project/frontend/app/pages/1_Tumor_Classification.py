import streamlit as st
import requests
import time
import pandas as pd
from PIL import Image
import io
import os

API_URL = "http://backend:8000"

st.set_page_config(page_title="🧠 Brain Tumor Classification", layout="wide")

# =========================
# PAGE HEADER
# =========================
st.markdown(
    """
    <h1 style="text-align:center; color:#1B2631;">🧠 Brain Tumor Classification</h1>
    <p style="text-align:center; color:gray; font-size:17px;">
    Upload an MRI image to classify tumor presence and type using our trained CNN model.
    </p>
    <hr style="border: 1px solid #E0E0E0;">
    """,
    unsafe_allow_html=True
)

# =========================
# FILE UPLOAD / SAMPLE SELECTOR
# =========================
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

st.caption("Or choose a sample MRI for quick demo:")
sample = st.selectbox(
    "🎞️ Sample Image",
    ["None", "notumor_sample.jpg", "glioma_sample.jpg", "meningioma_sample.jpg", "pituitary_sample.jpg"]
)

if sample != "None" and not uploaded_file:
    sample_path = f"./samples/{sample}"
    if os.path.exists(sample_path):
        st.info(f"Using sample image: {sample}")
        uploaded_file = open(sample_path, "rb")
    else:
        st.warning(f"⚠️ Sample image '{sample}' not found. Please upload an MRI manually.")
        uploaded_file = None

# =========================
# MODEL INFERENCE
# =========================
if uploaded_file:
    with st.spinner("⏳ Running model inference... please wait"):
        # Compatible with Streamlit uploads and local sample files
        if hasattr(uploaded_file, "type"):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        else:
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            files = {"file": (sample, uploaded_file, "image/jpeg")}

        response = requests.post(f"{API_URL}/classify", files=files)
        time.sleep(0.5)

    if response.status_code == 200:
        result = response.json()
        probs = result["probabilities"]
        max_prob = max(probs.values())

        # Convert bytes to PIL image for display
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # =========================
        # LAYOUT: IMAGE + RESULTS
        # =========================
        col1, col2 = st.columns([1.1, 2], gap="large")

        # Left column: uploaded image and Grad-CAM visualization
        with col1:
            st.image(img, caption="Uploaded MRI Image", use_container_width=True)

            with st.expander("🔍 Model Attention (Grad-CAM visualization)"):
                st.info("Grad-CAM heatmap will be shown here (backend function placeholder).")
                gradcam_path = "classification_matrix/gradcam.png"
                if os.path.exists(gradcam_path):
                    st.image(gradcam_path, caption="Example Heatmap", use_container_width=True)
                else:
                    st.write("Upload `gradcam.png` under classification_matrix/ to display example heatmap.")

        # Right column: prediction results and probability visualization
        with col2:
            status_color = "#117A65" if "Detected" in result["tumor_status"] else "#1A5276"
            st.markdown(
                f"""
                <div style="background-color:#E8F6F3; padding:15px; border-radius:10px; border:1px solid #D1F2EB;">
                    <h3 style="color:{status_color}; margin:0;">🩺 {result['tumor_status']}</h3>
                    <p style="color:gray; margin-top:6px;">Model confidence: {max_prob*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Warning for low-confidence predictions
            if max_prob < 0.6:
                st.warning("⚠️ Low confidence prediction — further clinical review recommended.")

            # Probability table
            st.markdown("<h4 style='margin-top:20px;'>📊 Class Probabilities:</h4>", unsafe_allow_html=True)
            prob_table = pd.DataFrame({
                "Tumor Type": [cls.capitalize() for cls in probs.keys()],
                "Probability (%)": [round(prob * 100, 2) for prob in probs.values()]
            }).sort_values("Probability (%)", ascending=False)
            st.dataframe(prob_table, use_container_width=True, hide_index=True)

            # Probability bar chart
            st.markdown("<h4 style='margin-top:25px;'>📈 Probability Distribution</h4>", unsafe_allow_html=True)
            st.bar_chart(prob_table.set_index("Tumor Type"))

        st.markdown("---")

        # =========================
        # MODEL PERFORMANCE SECTION
        # =========================
        st.markdown("## 🧮 Model Evaluation Summary")

        perf_col1, perf_col2 = st.columns([1, 2])

        with perf_col1:
            st.metric("Accuracy", "95%")
            st.metric("Macro Avg F1", "95%")
            st.metric("Weighted Avg F1", "95%")
            st.metric("Validation Samples", "1311")

        with perf_col2:
            conf_matrix_path = "classification_matrix/confusion_matrix.png"
            if os.path.exists(conf_matrix_path):
                st.image(conf_matrix_path, caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("Upload 'confusion_matrix.png' under classification_matrix/ to display confusion matrix here.")

        with st.expander("📋 Detailed Classification Report"):
            metrics_data = {
                "Class": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
                "Precision": [0.92, 0.93, 0.98, 0.97],
                "Recall": [0.93, 0.87, 1.00, 0.99],
                "F1-Score": [0.93, 0.90, 0.99, 0.98],
                "Support": [300, 306, 405, 300]
            }
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, hide_index=True, use_container_width=True)

        # =========================
        # MODEL ARCHITECTURE
        # =========================
        with st.expander("🧩 Model Architecture Overview"):
            st.markdown(
                """
                **CNN Layers:** 4 convolutional blocks with ReLU activations, MaxPooling, and Batch Normalization.  
                **Dense Layers:** 256-unit fully connected layer with L2 regularization and Dropout(0.5).  
                **Output:** 4-class Softmax for [Glioma, Meningioma, Pituitary, No Tumor].
                """
            )
            model_diagram = "classification_matrix/cnn_architecture.png"
            if os.path.exists(model_diagram):
                st.image(model_diagram, caption="CNN Model Architecture Diagram", use_container_width=True)
            else:
                st.info("Upload 'cnn_architecture.png' under classification_matrix/ to display the CNN architecture diagram.")


        # =========================
        # DATASET & TRAINING DETAILS
        # =========================
        with st.expander("📘 Dataset & Training Details"):
            st.markdown(
                """
                - **Dataset:** Brain MRI Classification (Kaggle, 7023 images)  
                - **Model:** CNN (4 Conv2D + Dense layers, dropout=0.5)  
                - **Optimizer:** Adam (lr=0.0001), Loss: Categorical Crossentropy  
                - **Epochs:** 50 (with early stop), Batch size: 32  
                - **Hardware:** NVIDIA T4 GPU  
                - **Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix  
                """
            )

    else:
        st.error("❌ Error occurred while processing the request.")

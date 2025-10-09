import streamlit as st
import os

st.set_page_config(page_title="Brain Tumor MRI Project", layout="wide")

# ====== Header ======
st.markdown(
    """
    <h1 style="text-align:center; color:#1B2631;">🧠 Brain Tumor MRI Project</h1>
    <p style="text-align:center; color:gray; font-size:17px;">
    A multi-module deep learning system for brain tumor classification and MRI-based image generation.
    </p>
    <hr style="border: 1px solid #E0E0E0;">
    """,
    unsafe_allow_html=True
)

# ====== Overview ======
st.markdown("### 📘 Project Overview")
st.write(
    """
    This system combines **deep learning** and **interactive visualization** to assist in brain tumor
    diagnosis and progression prediction.  
    The platform is composed of:
    - **Tumor Classification:** A CNN model for detecting and classifying brain tumors from MRI images.  
    - **Predictive Image Generation:** A Pix2Pix GAN model for temporal MRI synthesis and treatment-progress prediction.
    """
)

# ====== Clinical Context & Workflow ======
st.markdown("---")
st.markdown("## 🩺 Clinical Context & System Workflow")

col1, col2 = st.columns([1.2, 2])

# left: diagram
with col1:
    diagram_path = "system_design.png"
    if os.path.exists(diagram_path):
        st.image(diagram_path, caption="System Architecture Overview", use_container_width=True)
    else:
        st.info("Upload 'system_design.png' under to show the architecture diagram.")

# right: explanation
with col2:
    st.markdown(
        """
        ### Clinical Motivation  
        Brain tumor detection and follow-up imaging are critical for early diagnosis and treatment planning.  
        Manual MRI interpretation is time-consuming and prone to inter-observer variation.  
        This project aims to:
        - Support **automated tumor classification** via CNNs.  
        - Enable **temporal MRI generation** for treatment prognosis using conditional GANs.  

        ### Workflow Summary  
        1. MRI image is uploaded from the user interface.  
        2. The **FastAPI backend** processes the image and calls the trained CNN model for classification.  
        3. Optionally, the **Pix2Pix GAN module** generates predicted MRI sequences for time-based analysis.  
        4. Results and visual explanations (Grad-CAM, confusion matrix, metrics) are displayed interactively.  
        """
    )

st.markdown("---")
st.markdown(
    """
    <p style="text-align:center; color:gray;">
    Use the left sidebar to navigate between modules:<br>
    👉 Tumor Classification | Predictive Image Generation
    </p>
    """,
    unsafe_allow_html=True
)

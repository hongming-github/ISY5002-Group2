import streamlit as st
import requests

API_URL = "http://backend:8000"

st.title("🧪 Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(f"{API_URL}/classify", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted class: {result['predicted_class']}")
        st.subheader("Class Probabilities:")
        for cls, prob in result["probabilities"].items():
            st.write(f"- {cls}: {prob*100:.2f}%")
    else:
        st.error("Error occurred while processing request.")
    
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)


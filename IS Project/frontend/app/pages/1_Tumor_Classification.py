import streamlit as st
import requests

API_URL = "http://backend:8000"

st.title("🧪 Tumor Classification")

uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/classify", files=files)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Tumor Present: {result['tumor_present']}")
    else:
        st.error("Error occurred while processing request.")

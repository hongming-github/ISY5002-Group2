import streamlit as st
import requests
from PIL import Image

API_URL = "http://backend:8000"

st.title("🔮 Predictive Post-Surgery MRI Generation")

file1 = st.file_uploader("Upload Pre-op MRI", type=["png", "jpg", "jpeg"], key="pre")
file2 = st.file_uploader("Upload 1st Post-op MRI", type=["png", "jpg", "jpeg"], key="post1")

if file1 and file2:
    files = {
        "file1": file1.getvalue(),
        "file2": file2.getvalue()
    }
    response = requests.post(f"{API_URL}/generate", files=files)
    if response.status_code == 200:
        result = response.json()
        st.image(result["generated_image_path"], caption="Predicted 2nd Post-op MRI")
    else:
        st.error("Error occurred while generating image.")

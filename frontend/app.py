# frontend/app.py
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="🧥 Fashion Classifier", layout="centered")
st.title("🧠 Fashion Product Classifier with Explainability")
st.markdown(
    "Upload a fashion image to predict **Color**, **Type**, **Season**, and **Gender**, along with interpretability heatmaps.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    print("📡 Sending image to backend:", uploaded_file.name)
    # Sending image to FastAPI backend
    with st.spinner("🔍 Classifying..."):
        try:
            # --- CHANGE 1: Send the image content, not the uploader object ---
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(
                "http://127.0.0.1:8080/predict/",
                files=files
            )

            if response.status_code == 200:
                result = response.json()

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("🎯 Predictions")
                    st.markdown(f"**Color:** `{result['color']}`")
                    st.markdown(f"**Type:** `{result['product_type']}`")
                    st.markdown(f"**Season:** `{result['season']}`")
                    st.markdown(f"**Gender:** `{result['gender']}`")

                    st.subheader("🔥 Grad-CAM Heatmap")
                    # --- CHANGE 2: Construct the full URL to fetch the heatmap ---
                    # result['heatmap_path'] will now be "/static/heatmap_123.png"
                    heatmap_url = f"http://127.0.0.1:8080{result['heatmap_path']}"

                    st.image(heatmap_url, caption="Model Attention Heatmap", use_column_width=True)

            else:
                st.error(f"❌ Backend returned error code: {response.status_code}")
                st.text(f"Error details: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"🚫 Could not connect to the backend. Is it running? Error: {e}")
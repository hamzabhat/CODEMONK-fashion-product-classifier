import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="🧥 Fashion Classifier", layout="centered")
st.title("🧠 Fashion Product Classifier with Explainability")
st.markdown("Upload a fashion image to predict **Color**, **Type**, **Season**, and **Gender**, along with interpretability heatmaps.")


# Image uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Sending image to FastAPI backend
    with st.spinner("🔍 Classifying..."):
        files = {'file': uploaded_file.getvalue()}
        try:
            response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file})
            if response.status_code == 200:
                result = response.json()

                st.subheader("🎯 Predictions")
                st.markdown(f"**Color:** `{result['color']}`")
                st.markdown(f"**Type:** `{result['type']}`")
                st.markdown(f"**Season:** `{result['season']}`")
                st.markdown(f"**Gender:** `{result['gender']}`")


                # Show heatmap
                st.subheader("🔥 Grad-CAM Heatmap")
                heatmap_url = result["heatmap_path"]
                heatmap_response = requests.get(f"http://127.0.0.1:8000/{heatmap_url}")
                if heatmap_response.status_code == 200:
                    heatmap_image = Image.open(io.BytesIO(heatmap_response.content))
                    st.image(heatmap_image, use_column_width=True)
                else:
                    st.warning("Heatmap could not be loaded.")

            else:
                st.error("❌ Prediction failed. Please try another image.")
        except Exception as e:
            st.error(f"🚫 Backend not reachable: {e}")

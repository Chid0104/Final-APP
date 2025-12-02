import streamlit as st
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="BasuraNet Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

# ---------------- CUSTOM CSS -------------------
st.markdown("""
<style>
.title {
    font-size: 45px !important;
    font-weight: 900 !important;
    text-align: center;
    color: #2c3e50;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: #34495e;
    margin-bottom: 30px;
}

.result-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.label {
    padding: 15px;
    background: #27ae60;
    color: white;
    font-size: 28px;
    font-weight: bold;
    border-radius: 12px;
}

.unknown {
    padding: 15px;
    background: #c0392b;
    color: white;
    font-size: 28px;
    font-weight: bold;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL + MAPPING -------------------
@st.cache_resource
def load_basuranet(model_path, mapping_path):
    model = load_model(model_path)

    # Load the JSON
    with open(mapping_path, "r") as f:
        raw_map = json.load(f)

    # Auto-detect mapping format
    # FORMAT A: {"0": "biodegradable"}  → correct
    # FORMAT B: {"biodegradable": 0}    → inverted
    first_key = list(raw_map.keys())[0]

    if first_key.isdigit():
        idx_to_class = raw_map  # already correct format
    else:
        # Flip mapping to match model outputs
        idx_to_class = {str(v): k for k, v in raw_map.items()}

    return model, idx_to_class


# ---------------- HEADER -------------------
st.markdown('<div class="title">♻️ BasuraNet Waste Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or use your webcam to classify waste</div>', unsafe_allow_html=True)


# ---------------- LOAD MODEL -------------------
model_path = os.environ.get("MODEL_PATH", "model/basuranet_final.h5")
mapping_path = "class_indices.json"
model, idx_to_class = load_basuranet(model_path, mapping_path)


# ---------------- LAYOUT -------------------
col1, col2 = st.columns([1, 1])

# ---------------- IMAGE INPUT SECTION -------------------
with col1:
    st.subheader("📤 Choose Input Method")

    method = st.radio(
        "Select input method:",
        ["Upload Image", "Use Webcam"]
    )

    image = None

    # Upload Mode
    if method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # Webcam Mode
    else:
        webcam = st.camera_input("Take a photo")
        if webcam:
            image = Image.open(webcam)
            st.image(image, caption="Captured from Webcam", use_container_width=True)

    # PROCESS IMAGE
    preds = None
    if image:
        img = image.resize((224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])


# ---------------- RESULT SECTION -------------------
with col2:
    st.subheader("🔍 Prediction Result")

    if image:

        # If low confidence → Unknown
        if confidence < 0.50:
            st.markdown("""
                <div class="result-card">
                    <div class="unknown">❓ Not a Waste Image</div>
                </div>
            """, unsafe_allow_html=True)

        else:
            # SAFE LOOKUP (works even if JSON is wrong format)
            result_label = idx_to_class.get(str(top_idx), "Unknown Waste Type")

            st.markdown(f"""
                <div class="result-card">
                    <div class="label">{result_label}</div>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Upload an image or use the webcam to get a prediction.")


# ---------------- FOOTER -------------------
st.markdown("---")
st.markdown("<center>Developed for PIT Machine Learning Project • BasuraNet 2025</center>", unsafe_allow_html=True)

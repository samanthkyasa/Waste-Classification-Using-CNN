 
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Constants
IMG_SIZE = 128

# Preprocess image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3))
    return image

# Predict function
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    class_labels = {0: "Organic Waste", 1: "Non-Recyclable Waste"}
    return class_labels[class_index], prediction

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Upload an image to classify it as **Organic** or **Non-Recyclable Waste**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing image...")

    # Predict
    label, confidence = predict_image(image)

    # Display results
    st.write(f"### 🏷️ Prediction: {label}")
    st.write(f"### 🔥 Confidence: {np.max(confidence) * 100:.2f}%")

    # Confidence bar
    st.progress(int(np.max(confidence) * 100))

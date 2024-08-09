import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load your model
model_path = r'C:/Users/risha\Desktop\DATA SCIENCE PROJECTS/Plant Diseases/models/4.keras'
model = load_model(model_path)

# Define your class labels
classes = ['Early blight', 'Late blight', 'Healthy']

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image as per your model requirement
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = np.argmax(prediction, axis=1)[0]
    return classes[result]

# Streamlit app interface
st.markdown("<h1 style='text-align: center; color: #2A9D8F;'>Potato Diseases Predictor</h1>", unsafe_allow_html=True)
st.write("### Upload an image of a potato plant to get a prediction:")

# Image upload handling
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)

    # Display the image on the Streamlit app
    st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

    if st.button("Predict"):
        # Make prediction
        result = predict_image(image)
        st.write(f"### **Prediction:** {result}")

# Additional styling
st.markdown("""
    <style>
        .stButton button {
            background-color: #2A9D8F;
            color: white;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: #21867A;
        }
        .stImage {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

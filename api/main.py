from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Path to your model
MODEL_PATH = r"C:\Users\risha\Desktop\DATA SCIENCE PROJECTS\Plant Diseases\models\model_v1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Define class labels for your model (example labels)
CLASS_LABELS = ["Early Blight", "Late Blight", "Healthy"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess the image to the required size and format
        img = img.resize((224, 224))  # Resize to match the model input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        predictions = MODEL.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]

        # Return a human-readable result
        return {"prediction": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Plant Disease Prediction</title>
        </head>
        <body>
            <h1>Plant Disease Prediction</h1>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit">
            </form>
        </body>
    </html>
    """


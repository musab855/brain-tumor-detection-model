import cv2
import numpy as np
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import os

# Settings
IMG_SIZE = 224  # must use the same size which was used during model training!!!
UPLOAD_FOLDER = "static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# model = load_model("model/brain_tumor_multiclass.keras")
model = load_model("model/brain_tumor_multiclass_finetuned.h5")


# Class labels (must match training order)!!!
CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def predict_image(img_path):
    # Read images
    img = cv2.imread(img_path)  # color by default
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=0)  

    # Prediction
    prediction = model.predict(normalized)[0]  
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = prediction[predicted_index]

    # Building result message
    if predicted_class != "no tumor":
        result = f"\u26A0\ufe0f {predicted_class.capitalize()} tumor detected"
    else:
        result = "\u2705 No signs of brain tumor detected"

    # Returing all class probabilties
    all_probs = dict(sorted(
        {CLASS_NAMES[i]: float(prediction[i]) * 100 for i in range(len(CLASS_NAMES))}.items(),
        key=lambda item: item[1], 
        reverse=True
    ))

    return result, round(confidence * 100, 2), all_probs

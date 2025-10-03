from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_classification_model.keras")
model = load_model(MODEL_PATH)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def predict_tumor(file_bytes):

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((128, 128))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]

    return {
        "predicted_class": class_names[pred_class],
        "probabilities": {cls: float(pred[0][i]) for i, cls in enumerate(class_names)}
    }

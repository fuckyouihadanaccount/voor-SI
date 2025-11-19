from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
import numpy as np
import pickle
import io
import base64

app = FastAPI()

MODEL_DIR = "model"
autoencoder = load_model(f"{MODEL_DIR}/model.h5", compile=False)
with open(f"{MODEL_DIR}/config.pkl", "rb") as f:
    config = pickle.load(f)

image_size = (28, 28)
threshold = 0.009

def prepare_from_file(file):
    img = Image.open(file).convert("L").resize(image_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)

def predict_anomaly(file):
    arr = prepare_from_file(file)
    recon = autoencoder.predict(arr)
    error = float(np.mean((arr - recon) ** 2))
    is_normal = error > threshold
    return error, is_normal

def blur_center(img: Image.Image, blur_factor=15):
    w, h = img.size

    # simpel middenvak
    cx1 = int(w * 0.25)
    cy1 = int(h * 0.25)
    cx2 = int(w * 0.75)
    cy2 = int(h * 0.75)

    center_region = img.crop((cx1, cy1, cx2, cy2))
    center_region = center_region.filter(ImageFilter.GaussianBlur(blur_factor))

    img.paste(center_region, (cx1, cy1))
    return img

@app.post("/predict")
def predict(file: UploadFile = File(...)):

    error, is_normal = predict_anomaly(file.file)

    file.file.seek(0)

    img = Image.open(file.file).convert("RGB")

    is_mugshot = not is_normal
    if is_mugshot:
        img = blur_center(img)
        
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return {
        "reconstruction_error": error,
        "is_mugshot": is_mugshot,
        "image_base64": img_base64
    }

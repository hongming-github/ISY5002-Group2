from fastapi import FastAPI, UploadFile, File
from .routes import router
from tensorflow.keras.models import load_model

app = FastAPI(title="Brain Tumor Project API")

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Brain Tumor Project API is running"}
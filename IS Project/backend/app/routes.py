from fastapi import APIRouter, UploadFile, File
from .models.classifier import predict_tumor
from .models.generator import generate_followup

router = APIRouter()

@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Receive MRI image and return classification result
    result = predict_tumor(file.file)
    return {"tumor_present": result}

@router.post("/generate")
async def generate(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Input pre-op + 1st follow-up, generate predicted 2nd follow-up image
    output_path = generate_followup(file1.file, file2.file)
    return {"generated_image_path": output_path}

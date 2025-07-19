from fastapi import APIRouter, UploadFile, File
from backend.services.model_service import get_predictions_and_heatmap
import shutil
import os
import uuid

router = APIRouter()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print("✅ Received file:", file.filename)
    # Save uploaded image temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Get predictions + heatmap
    results = get_predictions_and_heatmap(temp_filename)
    # Remove temp image
    os.remove(temp_filename)

    return results


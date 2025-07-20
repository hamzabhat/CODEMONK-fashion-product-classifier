import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.model_service import get_predictions_and_heatmap
import traceback  # <--- Add this import

router = APIRouter()


@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, saves it temporarily, gets predictions,
    and returns the results. Includes detailed error logging.
    """
    # Create a unique temporary filename
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"

    # Save the uploaded file temporarily
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📥 Received image for prediction: {file.filename}")
    print(f"✅ Saved to temp file: {temp_filename}")

    try:
        # ⭐ THIS IS THE CRITICAL PART ⭐
        # We call the prediction function inside a 'try' block.
        # If any error occurs inside get_predictions_and_heatmap,
        # the 'except' block below will be executed.

        results = get_predictions_and_heatmap(temp_filename)
        return results

    except Exception as e:
        # --- This block runs ONLY if an error occurs ---
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!         AN ERROR OCCURRED - DETAILS         !")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Print the type and message of the error
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")

        print("\n--- Full Traceback ---")
        # This is the most important line: it prints the exact file,
        # line number, and function call that caused the crash.
        traceback.print_exc()
        print("-----------------------------------------------\n")

        # Also, send a proper error back to the frontend
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred in the backend. Check backend terminal for details. Error: {e}"
        )

    finally:
        # This block always runs, ensuring the temp file is deleted
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"🗑️ Deleted temp file: {temp_filename}")
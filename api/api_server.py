from fastapi import FastAPI, HTTPException
import os

from client.supabase_client import supabaseClient

from full_pipeline import run_full_screening_from_files
from clinical_agent.clinical_agent import generate_clinical_explanation_from_file

app = FastAPI()
@app.get("/")
def root():
    return {
        "service": "ASD Multimodal Clinical Decision Support API",
        "status": "running",
        "endpoints": ["/health", "/eeg"]
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/eeg")
def run_eeg_from_supabase(payload: dict):
    # user_id = payload.get("user_id")
    file_key = payload.get("file_key")      # EEG CSV
    image_key = payload.get("image_key")    # Face image
    video_key = payload.get("video_key")    # 10s video

    if  not file_key or not image_key or not video_key:
        raise HTTPException(
            status_code=400,
            detail="user_id, file_key, image_key, and video_key are required"
        )

    try:
        eeg_bucket = "eeg"
        image_bucket = "images"
        video_bucket = "videos"

        # -----------------------------
        # Download files from Supabase
        # -----------------------------
        eeg_result = supabaseClient.storage.from_(eeg_bucket).download(file_key)
        image_result = supabaseClient.storage.from_(image_bucket).download(image_key)
        video_result = supabaseClient.storage.from_(video_bucket).download(video_key)

        # -----------------------------
        # Normalize to bytes
        # -----------------------------
        def get_bytes(result):
            if hasattr(result, "read"):
                return result.read()
            elif isinstance(result, bytes):
                return result
            elif isinstance(result, dict) and "data" in result:
                return result["data"]
            else:
                raise RuntimeError(f"Unexpected download result type: {type(result)}")

        eeg_bytes = get_bytes(eeg_result)
        image_bytes = get_bytes(image_result)
        video_bytes = get_bytes(video_result)

        # -----------------------------
        # Save locally
        # -----------------------------
        os.makedirs("tmp", exist_ok=True)

        eeg_path = os.path.join("tmp", os.path.basename(file_key))
        image_path = os.path.join("tmp", os.path.basename(image_key))
        video_path = os.path.join("tmp", os.path.basename(video_key))

        with open(eeg_path, "wb") as f:
            f.write(eeg_bytes)

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # -----------------------------
        # Run FULL Screening + Fusion
        # -----------------------------
        fusion_output = run_full_screening_from_files(
            eeg_csv_path=eeg_path,
            face_image_path=image_path,
            video_path=video_path
        )

        # -----------------------------
        # Run Clinical Support Agent
        # -----------------------------
        clinical_explanation = generate_clinical_explanation_from_file(
            fusion_output_path="outputs/fusion_output.json"
        )

        # -----------------------------
        # Final API Response
        # -----------------------------
        return {
            "status": "success",
            "screening_and_fusion_output": fusion_output,
            "clinical_decision_support": clinical_explanation
        }

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process multimodal screening + clinical support: {str(e)}"
        )

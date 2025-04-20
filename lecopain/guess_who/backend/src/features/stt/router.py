# src/features/stt/router.py
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

# Import the service function and response schema
from .services import transcribe_audio_file
from .schema import TranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
    description="Upload an audio file (e.g., wav, mp3, ogg, webm) for transcription.",
)
async def http_transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Endpoint to receive an audio file via HTTP POST request and return its transcription.
    """
    logger.info(f"Received file '{file.filename}' for transcription, content type: {file.content_type}")

    # Check content type if needed (basic check)
    # You might want more robust validation depending on ffmpeg capabilities
    # if not file.content_type or not file.content_type.startswith("audio/"):
    #     logger.warning(f"Invalid content type: {file.content_type}")
    #     raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Read the entire file content into memory.
        # For very large files, consider streaming to a temporary file first.
        audio_bytes = await file.read()

        if not audio_bytes:
             raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Call the service function to process and transcribe
        transcription_text = await transcribe_audio_file(audio_bytes)

        logger.info(f"Successfully processed file '{file.filename}'")
        return TranscriptionResponse(transcription=transcription_text)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (e.g., from service layer)
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error processing file '{file.filename}': {e}")
        # Return a structured error response using the schema
        return TranscriptionResponse(
            transcription="",
            error=f"An unexpected error occurred: {type(e).__name__}"
        )
    finally:
        # Ensure the file handle is closed (FastAPI usually handles this with UploadFile)
        await file.close()
        logger.debug(f"Closed file handle for '{file.filename}'")
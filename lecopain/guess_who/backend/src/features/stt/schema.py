# src/features/stt/schema.py
from pydantic import BaseModel

class TranscriptionResponse(BaseModel):
    """
    Pydantic model for the transcription response.
    """
    transcription: str
    error: str | None = None # Optional field for reporting errors
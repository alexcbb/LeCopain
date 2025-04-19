# src/features/stt/services.py
import asyncio
import logging
import subprocess
import numpy as np
import os
from fastapi import HTTPException

# Import the globally loaded model from main.py
# This assumes model is loaded in main.py and accessible
# Adjust the import path if your structure differs slightly
from src.main import model, FFMPEG_PATH # Import model and ffmpeg path

logger = logging.getLogger(__name__)

# --- Audio processing constants (same as before) ---
# INPUT_FORMAT is not strictly needed here as ffmpeg detects it,
# but it's good practice if you want to enforce certain input types later.
OUTPUT_SAMPLE_RATE = 16000
OUTPUT_CHANNELS = 1
OUTPUT_FORMAT = 'f32le' # PCM 32-bit float little-endian
BYTES_PER_SAMPLE = np.dtype(np.float32).itemsize

async def transcribe_audio_file(audio_bytes: bytes) -> str:
    """
    Processes an audio file (bytes), converts it using ffmpeg,
    and transcribes it using the loaded Faster Whisper model.
    """
    if not model:
        logger.error("Transcription failed: Faster Whisper model not loaded.")
        raise HTTPException(status_code=500, detail="Transcription model is not available.")

    if not audio_bytes:
        logger.warning("Transcription skipped: No audio data received.")
        return "" # Or raise HTTPException(status_code=400, detail="No audio data received.")

    ffmpeg_command = [
        FFMPEG_PATH,
        '-loglevel', 'error',
        '-i', 'pipe:0',         # Read input from stdin
        '-f', OUTPUT_FORMAT,    # Output format PCM float 32-bit little-endian
        '-ar', str(OUTPUT_SAMPLE_RATE), # Sample rate
        '-ac', str(OUTPUT_CHANNELS),    # Channel count (mono)
        '-'                     # Write output to stdout
    ]

    logger.debug(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Write audio data to ffmpeg's stdin and close it
        stdout_data, stderr_data = await process.communicate(input=audio_bytes)

        if process.returncode != 0:
            error_message = stderr_data.decode('utf-8', errors='ignore').strip()
            logger.error(f"ffmpeg error (code {process.returncode}): {error_message}")
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {error_message}")

        if not stdout_data:
            logger.warning("ffmpeg produced no output PCM data.")
            return "" # No data to transcribe

        # Convert PCM bytes to NumPy array
        try:
            audio_np = np.frombuffer(stdout_data, dtype=np.float32)
        except Exception as np_err:
             logger.error(f"Error converting PCM bytes to NumPy array: {np_err}")
             raise HTTPException(status_code=500, detail="Error processing audio data.")

        if audio_np.size == 0:
             logger.info("Transcription skipped: Empty audio array after conversion.")
             return ""

        # --- Transcription using faster-whisper ---
        logger.debug(f"Transcribing {audio_np.size} PCM samples using faster-whisper...")
        try:
            # Run synchronous model.transcribe in an executor thread
            segments, _ = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: list(model.transcribe(audio_np, beam_size=5))
            )
            transcription = " ".join(seg.text.strip() for seg in segments if seg.text).strip()

            if not transcription:
                transcription = "" # No speech detected or empty result
                logger.info("Transcription result is empty.")
            else:
                logger.info(f"Transcription successful.") # Don't log full transcription here usually
                # logger.debug(f"Transcription result: '{transcription}'") # Debug if needed

            return transcription

        except Exception as e:
            logger.exception(f"Error during faster-whisper transcription process: {e}")
            raise HTTPException(status_code=500, detail="Error during transcription process.")

    except FileNotFoundError:
         logger.error(f"CRITICAL ERROR: ffmpeg command '{FFMPEG_PATH}' not found.")
         raise HTTPException(status_code=500, detail="Server configuration error: ffmpeg not found.")
    except Exception as ffmpeg_err:
         logger.error(f"Error running ffmpeg process: {ffmpeg_err}", exc_info=True)
         raise HTTPException(status_code=500, detail="Server error during audio processing.")
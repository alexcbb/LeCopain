# src/main.py
import asyncio
import logging
import os
# Remove unused numpy/subprocess if not needed elsewhere
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Keep Whisper model loading if still needed for STT
from faster_whisper import WhisperModel

# --- Configuration (Keep existing logging setup) ---
try:
    import src.core.config
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# --- Locate ffmpeg (Keep if STT feature is used) ---
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
logger.info(f"Using ffmpeg command: '{FFMPEG_PATH}'") # Only relevant if STT is active

# --- Load the Faster Whisper model (Keep if STT feature is used) ---
MODEL_NAME = os.getenv("WHISPER_MODEL", "ctranslate2-4you/whisper-base.en-ct2-int8_bfloat16")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
# Translated log message
logger.info(f"Loading Faster Whisper model '{MODEL_NAME}' on device '{DEVICE}'...")
model = None # Initialize model as None
try:
    # Only load if STT feature might be used
    # Consider loading models conditionally based on enabled features if memory is a concern
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
    # Translated log message
    logger.info("Faster Whisper model loaded successfully.")
except Exception as model_load_err:
     # Translated log message
     logger.exception(f"CRITICAL ERROR: Failed to load Faster Whisper model '{MODEL_NAME}': {model_load_err}")
     # model remains None


# --- FastAPI and CORS configuration ---
app = FastAPI(title="AI Services API") # Already English

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---

# Include STT router (keep if needed)
from src.features.stt.router import router as stt_router
from src.features.guess_who.router import router as guess_who_router
app.include_router(stt_router, prefix="/api/stt", tags=["Speech-to-Text"]) # Tag already English
app.include_router(guess_who_router, prefix="/api/guess_who", tags=["Guess Who AI"])


# Include user router (keep if needed)
# from src.features.user.router import router as user_router
# app.include_router(user_router, prefix="/api/users", tags=["Users"]) # Example

# --- Health Check / Root Endpoint ---
@app.get("/", tags=["Health Check"]) # Tag already English
async def read_root():
    # Indicate which core services are available
    # Use English keys
    services_status = {
        "stt_model_loaded": model is not None,
        "guess_who_llm_available": 'mistral_client' in locals() and mistral_client is not None
    }
    # Translated message
    return {"message": "AI Services API is running.", "services": services_status}

# --- Uvicorn Startup (if running directly) ---
if __name__ == "__main__":
    # Check needs the variable in scope
    guess_who_client_available = False
    if 'src.features.guess_who.services' in sys.modules:
         from src.features.guess_who.services import mistral_client
         guess_who_client_available = mistral_client is not None

    if not guess_who_client_available:
         # Translated print statement
         print("\n!!! WARNING: Mistral client failed to initialize. Guess Who API will not work. Check logs and MISTRAL_API_KEY. !!!\n")
    if not model:
        # Translated print statement
        print("\n!!! WARNING: Faster Whisper model failed to load. STT API might not function correctly. Check logs. !!!\n")

    # Translated print statement
    print("\n--- AI Services API Ready ---")
    if guess_who_client_available:
        # This uses MODEL_NAME which is read from the environment variable
        # Import needed variable if check passed
        from src.features.guess_who.services import MODEL_NAME as GUESS_WHO_MODEL_NAME # Alias to avoid conflict if stt uses same env var
        # Translated print statements
        print(f"Guess Who LLM Model: {GUESS_WHO_MODEL_NAME}")
        print(f"Guess Who API Endpoints available under /api/guess_who")
    if model:
         # This uses MODEL_NAME which is read from the environment variable
        # Alias WHISPER_MODEL_NAME if needed, but it's defined earlier
        # Translated print statements
        print(f"Faster Whisper Model: {MODEL_NAME} (Device: {DEVICE})") # Assuming MODEL_NAME refers to whisper here
        print(f"STT API Endpoint available at /api/stt/transcribe (POST)")
    # Translated print statement
    print(f"Allowed Origins: {origins}")
    # Translated print statement (added 'src.' prefix based on common practice)
    print("\nRun with: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload\n")
else: # Added else block for clarity when main.py is imported
    # Perform client check when imported as well
    guess_who_client_available = False
    try:
        # Attempt import safely during module load
        import sys
        if 'src.features.guess_who.services' in sys.modules:
            from src.features.guess_who.services import mistral_client
            guess_who_client_available = mistral_client is not None
        elif 'mistral_client' in globals(): # Fallback check if imported differently
             guess_who_client_available = mistral_client is not None

    except ImportError:
         logger.warning("Could not check mistral_client status during import.")
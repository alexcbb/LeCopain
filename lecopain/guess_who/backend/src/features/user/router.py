from fastapi import APIRouter, Depends, HTTPException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/api/hello", tags=["test"])
def hello():
    return True
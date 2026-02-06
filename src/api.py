"""FastAPI Backend for Hallucination Detection"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time
from datetime import datetime

from src.model import HallucinationDetector
from utils.config import MODEL_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Hallucination Detection API",
    description="API for detecting hallucinations in AI-generated summaries",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
detector = None

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    global detector
    logger.info("Loading model...")
    detector = HallucinationDetector(MODEL_CONFIG['model_save_path'])
    logger.info("Model loaded successfully!")

# Pydantic models
class CheckRequest(BaseModel):
    original_text: str
    summary_text: str

class CheckResponse(BaseModel):
    result: str
    confidence: float
    prediction_idx: int
    all_scores: dict
    processing_time_ms: float
    timestamp: str

# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hallucination Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/check", response_model=CheckResponse)
async def check_hallucination(request: CheckRequest):
    """Check a single summary for hallucinations"""
    start_time = time.time()
    
    try:
        logger.info("Received check request")
        result = detector.predict(request.original_text, request.summary_text)
        processing_time = (time.time() - start_time) * 1000
        
        response = CheckResponse(
            **result,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Request processed in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples")
async def get_examples():
    """Get example inputs for testing"""
    return {
        "examples": [
            {
                "name": "Correct Summary",
                "original_text": "Apple released iPhone 15 in September 2023 with USB-C port.",
                "summary_text": "The iPhone 15 has a USB-C port.",
                "expected_result": "CORRECT"
            },
            {
                "name": "Hallucination",
                "original_text": "Apple released iPhone 15 in September 2023 with USB-C port.",
                "summary_text": "The iPhone 15 has a Lightning port.",
                "expected_result": "HALLUCINATION"
            }
        ]
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

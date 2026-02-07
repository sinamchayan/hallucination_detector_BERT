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

class BatchCheckRequest(BaseModel):
    text_pairs: List[dict]

# Metrics storage
metrics = {
    'total_requests': 0,
    'total_hallucinations_detected': 0,
    'average_confidence': 0.0,
    'hallucination_rate': 0.0
}

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

@app.get("/metrics")
async def get_metrics():
    """Get global usage metrics"""
    return metrics

@app.post("/check", response_model=CheckResponse)
async def check_hallucination(request: CheckRequest):
    """Check a single summary for hallucinations"""
    start_time = time.time()
    
    try:
        logger.info("Received check request")
        
        # Make prediction
        result = detector.predict(request.original_text, request.summary_text)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        global metrics
        metrics['total_requests'] += 1
        
        # Update running average confidence
        # New Avg = ((Old Avg * (N-1)) + New Value) / N
        if metrics['total_requests'] > 1:
            prev_total = metrics['total_requests'] - 1
            curr_conf = result['confidence']
            metrics['average_confidence'] = (
                (metrics['average_confidence'] * prev_total) + curr_conf
            ) / metrics['total_requests']
        else:
            metrics['average_confidence'] = result['confidence']
            
        if "HALLUCINATION" in result['result']:
            metrics['total_hallucinations_detected'] += 1
            
        if metrics['total_requests'] > 0:
            metrics['hallucination_rate'] = (
                metrics['total_hallucinations_detected'] / metrics['total_requests']
            )
        
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

@app.post("/batch")
async def batch_check(request: BatchCheckRequest):
    """Batch check multiple summaries"""
    start_time = time.time()
    
    try:
        logger.info(f"Received batch request with {len(request.text_pairs)} pairs")
        
        pairs = [(p['original_text'], p['summary_text']) for p in request.text_pairs]
        results = detector.batch_predict(pairs)
        
        processing_time = (time.time() - start_time) * 1000
        avg_time = processing_time / len(pairs) if pairs else 0
        
        # Update metrics for batch (simplified)
        global metrics
        metrics['total_requests'] += len(pairs)
        
        hallucination_count = sum(1 for r in results if "HALLUCINATION" in r['result'])
        metrics['total_hallucinations_detected'] += hallucination_count
        
        # Approximate average confidence update (simplified)
        if len(pairs) > 0:
            batch_avg_conf = sum(r['confidence'] for r in results) / len(pairs)
            
            # Simple weighted average update
            prev_total = metrics['total_requests'] - len(pairs)
            if prev_total > 0:
                metrics['average_confidence'] = (
                    (metrics['average_confidence'] * prev_total) + (batch_avg_conf * len(pairs))
                ) / metrics['total_requests']
            else:
                metrics['average_confidence'] = batch_avg_conf

        if metrics['total_requests'] > 0:
            metrics['hallucination_rate'] = (
                metrics['total_hallucinations_detected'] / metrics['total_requests']
            )
            
        return {
            "results": results,
            "total_pairs": len(pairs),
            "processing_time_ms": processing_time,
            "avg_processing_time_ms": avg_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

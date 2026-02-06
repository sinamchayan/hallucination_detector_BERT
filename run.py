"""Main entry point for the hallucination detector"""
import argparse
import subprocess
import sys
from utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model():
    """Train the hallucination detection model"""
    logger.info("Starting model training...")
    from src.model import HallucinationDetector
    detector = HallucinationDetector()
    detector.train()
    logger.info("Training complete!")

def start_api():
    """Start the FastAPI server"""
    logger.info("Starting API server...")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def start_ui():
    """Start the Streamlit UI"""
    logger.info("Starting Streamlit UI...")
    subprocess.run([
        sys.executable, "-m", "streamlit",
        "run", "src/ui.py",
        "--server.port", "8501"
    ])

def main():
    parser = argparse.ArgumentParser(description="Hallucination Detector")
    parser.add_argument(
        'command',
        choices=['train', 'api', 'ui', 'all'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command == 'api':
        start_api()
    elif args.command == 'ui':
        start_ui()
    elif args.command == 'all':
        logger.info("Run 'python run.py api' in one terminal")
        logger.info("Run 'python run.py ui' in another terminal")

if __name__ == '__main__':
    main()

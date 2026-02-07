"""Configuration for the hallucination detector"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model settings
MODEL_CONFIG = {
    'base_model': 'bert-base-uncased',
    'model_save_path': MODEL_DIR / 'saved_model',
    'num_labels': 3,
    'max_length': 128,
    'labels': {
        0: 'CORRECT ✅',
        1: 'UNCLEAR ⚠️',
        2: 'HALLUCINATION ❌'
    }
}

# Training settings
TRAINING_CONFIG = {
    'num_samples': 50000,
    'batch_size': 4,
    'epochs': 3,
    'learning_rate': 2e-5,
    'validation_split': 0.1,
    'early_stopping_patience': 2
}

# API settings
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info'
}

# UI settings
UI_CONFIG = {
    'title': '🔍 Hallucination Detector (BERT)',
    'page_icon': '🔍',
    'layout': 'wide',
    'api_url': os.getenv('API_URL', 'http://localhost:8000')
}

# MLflow settings
MLFLOW_CONFIG = {
    'tracking_uri': str(LOGS_DIR / 'mlruns'),
    'experiment_name': 'hallucination_detection'
}

# Logging
LOG_CONFIG = {
    'log_file': LOGS_DIR / 'app.log',
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

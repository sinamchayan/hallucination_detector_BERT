# Hallucination Detector (BERT - TensorFlow)

Detect false information (hallucinations) in AI-generated summaries using Natural Language Inference with TensorFlow BERT.

This project leverages a fine-tuned BERT model to classify the consistency between a source document and a generated summary. By treating the problem as a Natural Language Inference (NLI) task, the system can determine if the summary is entailed by the source (faithful) or contradicts it (hallucination). It serves as a critical quality assurance layer for LLM pipelines, ensuring the reliability of automated text generation in professional and academic contexts.

## Features

- ✅ Single & batch processing
- 📊 Interactive Streamlit dashboard
- 🚀 FastAPI REST API
- 📈 MLflow experiment tracking
- 🐳 Docker deployment ready
- 🧠 TensorFlow/Keras BERT model

## Quick Start

### 1. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Model (30-60 minutes)
```bash
python run.py train
```

### 3. Start API
```bash
python run.py api
```

### 4. Start UI (new terminal)
```bash
python run.py ui
```

Open browser at `http://localhost:8501`

## Project Directory

```
├── data/               # Data storage
├── docker/             # Docker configuration
├── logs/               # Application logs
├── mlruns/             # MLflow experiments
├── models/             # Saved models
├── src/                # Source code
│   ├── api.py          # FastAPI application
│   ├── model.py        # BERT model implementation
│   └── ui.py           # Streamlit dashboard
├── utils/              # Utility functions
│   ├── config.py       # Configuration settings
│   └── logger.py       # Logging setup
├── run.py              # Main entry point script
├── start_app.sh        # Startup script
└── requirements.txt    # Project dependencies
```

## Project Workflow

The project is designed to be modular. You can run components individually or use the helper script.

### Automated Startup
To start both the API and UI simultaneously:
```bash
chmod +x start_app.sh
./start_app.sh
```

### Manual Workflow
1. **Training Phase**: The model must be trained first to generate the necessary artifacts in `models/`.
   ```bash
   python run.py train
   ```
2. **Serving Phase**:
   - **API**: Handles inference requests.
   - **UI**: Provides a user interface for interaction.

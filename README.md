# 🔍 AI Hallucination Detector (TensorFlow)

Detect false information (hallucinations) in AI-generated summaries using Natural Language Inference with TensorFlow BERT.

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

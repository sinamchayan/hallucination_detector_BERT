import tensorflow as tf
import transformers
print(f"TensorFlow version: {tf.__version__}")
print(f"Transformers version: {transformers.__version__}")
try:
    from src.model import HallucinationDetector
    print("Code imported successfully")
except Exception as e:
    print(f"Import error: {e}")

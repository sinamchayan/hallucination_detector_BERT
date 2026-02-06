"""Hallucination Detection Model - TensorFlow Version"""
import os


from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from datasets import load_dataset
import numpy as np
import mlflow
import mlflow.tensorflow
from pathlib import Path
from utils.config import MODEL_CONFIG, TRAINING_CONFIG, MLFLOW_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ Found {len(gpus)} GPU(s). Memory growth enabled.")
        logger.info(f"GPU Details: {gpus}")
        
        # Enable Mixed Precision (FP16)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("✅ Mixed Precision (FP16) enabled")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(f"❌ Error configuring GPU: {e}")
else:
    logger.warning("⚠️ No GPU found. Running on CPU - Expect slower performance.")

class HallucinationDetector:
    """Main model class for hallucination detection using TensorFlow BERT"""
    
    def __init__(self, model_path=None):
        """
        Initialize detector
        Args:
            model_path: Path to saved model (None = create new)
        """
        logger.info("Initializing Hallucination Detector...")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['base_model'])
        self.labels = MODEL_CONFIG['labels']
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            self.model = TFBertForSequenceClassification.from_pretrained(model_path)
        else:
            logger.info("Creating new model")
            self.model = TFBertForSequenceClassification.from_pretrained(
                MODEL_CONFIG['base_model'],
                num_labels=MODEL_CONFIG['num_labels']
            )
    
    def preprocess(self, premise, hypothesis):
        """Tokenize text pairs for TensorFlow"""
        return self.tokenizer(
            premise,
            hypothesis,
            max_length=MODEL_CONFIG['max_length'],
            truncation=True,
            padding=True,
            return_tensors='tf'
        )
    
    def predict(self, original_text, summary_text):
        """
        Predict if summary contains hallucinations
        
        Args:
            original_text: Source document
            summary_text: AI-generated summary
            
        Returns:
            dict with result, confidence, all_scores
        """
        logger.info("Making prediction...")
        
        # Tokenize
        inputs = self.preprocess(original_text, summary_text)
        
        # Get prediction
        outputs = self.model(inputs)
        logits = outputs.logits
        
        # Convert to probabilities
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        
        # Get best prediction
        prediction_idx = int(np.argmax(probs))
        confidence = float(probs[prediction_idx])
        
        result = {
            'result': self.labels[prediction_idx],
            'confidence': confidence,
            'prediction_idx': prediction_idx,
            'all_scores': {
                'correct': float(probs[0]),
                'unclear': float(probs[1]),
                'hallucination': float(probs[2])
            }
        }
        
        logger.info(f"Prediction: {result['result']} (confidence: {confidence:.2%})")
        return result
    
    def batch_predict(self, text_pairs):
        """Predict on multiple text pairs"""
        logger.info(f"Batch prediction on {len(text_pairs)} pairs...")
        results = []
        for original, summary in text_pairs:
            result = self.predict(original, summary)
            results.append(result)
        return results
    
    def train(self, num_samples=None, epochs=None):
        """
        Train the model on MNLI dataset
        
        Args:
            num_samples: Number of training examples (default from config)
            epochs: Training epochs (default from config)
        """
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
        mlflow.set_experiment(MLFLOW_CONFIG['experiment_name'])
        
        # Get config
        num_samples = num_samples or TRAINING_CONFIG['num_samples']
        epochs = epochs or TRAINING_CONFIG['epochs']
        batch_size = TRAINING_CONFIG['batch_size']
        learning_rate = TRAINING_CONFIG['learning_rate']
        
        logger.info(f"Starting training with {num_samples} samples, {epochs} epochs")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'num_samples': num_samples,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'base_model': MODEL_CONFIG['base_model'],
                'framework': 'tensorflow'
            })
            
            # Load dataset
            logger.info("Loading MNLI dataset...")
            dataset = load_dataset('glue', 'mnli', split=f'train[:{num_samples}]')
            
            # Split train/val
            val_size = int(num_samples * TRAINING_CONFIG['validation_split'])
            train_size = num_samples - val_size
            
            train_data = dataset.select(range(train_size))
            val_data = dataset.select(range(train_size, num_samples))
            
            logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
            
            # Tokenize training data
            logger.info("Tokenizing training data...")
            train_encodings = self.tokenizer(
                train_data['premise'],
                train_data['hypothesis'],
                max_length=MODEL_CONFIG['max_length'],
                truncation=True,
                padding=True,
                return_tensors='tf'
            )
            
            train_labels = tf.constant(train_data['label'])
            
            # Create TensorFlow dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_encodings),
                train_labels
            )).shuffle(1000).batch(batch_size)
            
            # Tokenize validation data
            logger.info("Tokenizing validation data...")
            val_encodings = self.tokenizer(
                val_data['premise'],
                val_data['hypothesis'],
                max_length=MODEL_CONFIG['max_length'],
                truncation=True,
                padding=True,
                return_tensors='tf'
            )
            
            val_labels = tf.constant(val_data['label'])
            
            # Create validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((
                dict(val_encodings),
                val_labels
            )).batch(batch_size)
            
            # Compile model
            logger.info("Compiling model...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=TRAINING_CONFIG['early_stopping_patience'],
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=1,
                    min_lr=1e-7
                ),
                MLflowCallback()
            ]
            
            # Train
            logger.info("Training model...")
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            # Log final metrics
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            mlflow.log_metrics({
                'final_train_accuracy': final_train_acc,
                'final_val_accuracy': final_val_acc,
                'final_val_loss': final_val_loss
            })
            
            logger.info(f"Training complete!")
            logger.info(f"Final train accuracy: {final_train_acc:.4f}")
            logger.info(f"Final val accuracy: {final_val_acc:.4f}")
            
            # Save model
            save_path = MODEL_CONFIG['model_save_path']
            logger.info(f"Saving model to {save_path}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Log model to MLflow
            mlflow.tensorflow.log_model(self.model, "model")
            
            logger.info("✅ Training pipeline complete!")
            
        return history

class MLflowCallback(tf.keras.callbacks.Callback):
    """Custom callback to log metrics to MLflow during training"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch"""
        if logs:
            mlflow.log_metrics({
                'train_loss': logs.get('loss'),
                'train_accuracy': logs.get('accuracy'),
                'val_loss': logs.get('val_loss'),
                'val_accuracy': logs.get('val_accuracy')
            }, step=epoch)

def test_model():
    """Test the trained model with examples"""
    logger.info("Testing model...")
    detector = HallucinationDetector(MODEL_CONFIG['model_save_path'])
    
    test_cases = [
        {
            'original': "The Eiffel Tower was completed in 1889 and is located in Paris, France. It stands 330 meters tall.",
            'summary': "The Eiffel Tower is in Paris and was completed in 1889.",
            'expected': 'CORRECT'
        },
        {
            'original': "The Eiffel Tower was completed in 1889 and is located in Paris, France. It stands 330 meters tall.",
            'summary': "The Eiffel Tower is in London.",
            'expected': 'HALLUCINATION'
        },
        {
            'original': "The Eiffel Tower was completed in 1889 and is located in Paris, France. It stands 330 meters tall.",
            'summary': "The Eiffel Tower is painted red.",
            'expected': 'UNCLEAR'
        }
    ]
    
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"Summary: {test['summary']}")
        result = detector.predict(test['original'], test['summary'])
        print(f"Result: {result['result']} (Confidence: {result['confidence']:.2%})")
        print(f"Expected: {test['expected']}")
        print(f"All scores: {result['all_scores']}")
        print("-" * 60 + "\n")

if __name__ == '__main__':
    detector = HallucinationDetector()
    detector.train()
    test_model()

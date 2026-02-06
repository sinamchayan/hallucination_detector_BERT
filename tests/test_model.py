import unittest
import os
import shutil
from pathlib import Path
from src.model import HallucinationDetector
from utils.config import MODEL_CONFIG

class TestHallucinationDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Ensure we are using a temporary model path or the default one
        # For testing purposes, we might want to check if model exists or mock it
        # But here we will just instantiate the detector
        cls.detector = HallucinationDetector()

    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.tokenizer)

    def test_prediction_structure(self):
        """Test if prediction returns correct structure"""
        original = "The apple is red."
        summary = "The apple is green."
        result = self.detector.predict(original, summary)
        
        self.assertIn('result', result)
        self.assertIn('confidence', result)
        self.assertIn('prediction_idx', result)
        self.assertIn('all_scores', result)
        self.assertIn('correct', result['all_scores'])
        self.assertIn('unclear', result['all_scores'])
        self.assertIn('hallucination', result['all_scores'])

    def test_batch_prediction(self):
        """Test batch prediction"""
        pairs = [
            ("The sky is blue.", "The sky is blue."),
            ("The sky is blue.", "The sky is green.")
        ]
        results = self.detector.batch_predict(pairs)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('result', result)

    def test_preprocessing(self):
        """Test preprocessing/tokenization"""
        premise = "Test premise"
        hypothesis = "Test hypothesis"
        inputs = self.detector.preprocess(premise, hypothesis)
        self.assertIn('input_ids', inputs)
        self.assertIn('attention_mask', inputs)
        # TensorFlow tensors
        self.assertEqual(inputs['input_ids'].shape[0], 1)

if __name__ == '__main__':
    unittest.main()

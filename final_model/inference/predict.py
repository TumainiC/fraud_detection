import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add the preprocessing directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

# Import the preprocessing function
try:
    from preprocessing_function import preprocess_transaction
except ImportError:
    print("Warning: Could not import preprocessing_function. Using fallback preprocessing.")

class FraudDetectionModel:
    """
    Production-ready fraud detection model wrapper
    """

    def __init__(self, model_dir=None):
        """Initialize the fraud detection model"""
        if model_dir is None:
            model_dir = os.path.dirname(__file__)

        self.model_dir = model_dir
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.threshold = 0.5

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessing artifacts"""
        try:
            # Load the trained model
            model_path = os.path.join(self.model_dir, '..', 'model', 'fraud_detection_model.pkl')
            self.model = joblib.load(model_path)
            print(f" Model loaded from: {model_path}")

            # Load feature names
            feature_names_path = os.path.join(self.model_dir, '..', 'preprocessing', 'feature_names.pkl')
            self.feature_names = joblib.load(feature_names_path)
            print(f" Feature names loaded: {len(self.feature_names)} features")

            # Load metadata
            metadata_path = os.path.join(self.model_dir, '..', 'metadata', 'model_metadata.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f" Metadata loaded")

        except Exception as e:
            print(f" Error loading artifacts: {e}")
            raise

    def preprocess_data(self, data):
        """Apply preprocessing to input data"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)

        # Basic preprocessing (fallback if preprocessing_function not available)
        try:
            processed_data = preprocess_transaction(data, self.feature_names)
        except NameError:
            # Fallback preprocessing
            processed_data = self._fallback_preprocessing(data)

        return processed_data

    def _fallback_preprocessing(self, df):
        """Simple fallback preprocessing if main function unavailable"""
        df_clean = df.copy()

        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna('Missing', inplace=True)

        # Add missing features as zeros
        for feature in self.feature_names:
            if feature not in df_clean.columns:
                df_clean[feature] = 0

        # Keep only required features
        df_clean = df_clean[self.feature_names]

        return df_clean

    def predict_single(self, transaction_data, threshold=None):
        """Predict fraud probability for a single transaction"""
        if threshold is None:
            threshold = self.threshold

        # Preprocess the data
        processed_data = self.preprocess_data(transaction_data)

        # Get prediction probability
        fraud_probability = self.model.predict_proba(processed_data)[0, 1]

        # Make binary prediction
        is_fraud = fraud_probability >= threshold

        # Determine risk level
        if fraud_probability >= 0.8:
            risk_level = "CRITICAL"
        elif fraud_probability >= 0.6:
            risk_level = "HIGH"
        elif fraud_probability >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            'fraud_probability': round(fraud_probability, 4),
            'is_fraud': bool(is_fraud),
            'risk_level': risk_level,
            'threshold_used': threshold,
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': self.metadata.get('model_info', {}).get('version', 'unknown')
        }

    def predict_batch(self, transactions_data, threshold=None):
        """Predict fraud probability for multiple transactions"""
        if threshold is None:
            threshold = self.threshold

        # Preprocess the data
        processed_data = self.preprocess_data(transactions_data)

        # Get prediction probabilities
        fraud_probabilities = self.model.predict_proba(processed_data)[:, 1]

        # Make binary predictions
        is_fraud = fraud_probabilities >= threshold

        results = []
        for i, prob in enumerate(fraud_probabilities):
            # Determine risk level
            if prob >= 0.8:
                risk_level = "CRITICAL"
            elif prob >= 0.6:
                risk_level = "HIGH"
            elif prob >= 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            results.append({
                'transaction_index': i,
                'fraud_probability': round(prob, 4),
                'is_fraud': bool(is_fraud[i]),
                'risk_level': risk_level,
                'threshold_used': threshold
            })

        return {
            'predictions': results,
            'summary': {
                'total_transactions': len(results),
                'flagged_as_fraud': int(sum(is_fraud)),
                'average_fraud_probability': round(float(np.mean(fraud_probabilities)), 4),
                'high_risk_transactions': len([r for r in results if r['risk_level'] in ['HIGH', 'CRITICAL']])
            },
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': self.metadata.get('model_info', {}).get('version', 'unknown')
        }

    def get_model_info(self):
        """Get model information and metadata"""
        return {
            'model_name': self.metadata.get('model_info', {}).get('name', 'fraud-detection-model'),
            'version': self.metadata.get('model_info', {}).get('version', 'unknown'),
            'algorithm': self.metadata.get('model_info', {}).get('algorithm', 'XGBoost'),
            'training_date': self.metadata.get('model_info', {}).get('training_date', 'unknown'),
            'performance_metrics': self.metadata.get('performance_metrics', {}),
            'feature_count': len(self.feature_names),
            'current_threshold': self.threshold,
            'supported_thresholds': {
                'conservative': 0.3,
                'balanced': 0.5,
                'aggressive': 0.7
            }
        }

    def update_threshold(self, new_threshold):
        """Update the decision threshold"""
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        old_threshold = self.threshold
        self.threshold = new_threshold

        return {
            'old_threshold': old_threshold,
            'new_threshold': new_threshold,
            'updated_at': datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    print(" Testing Fraud Detection Model")
    print("=" * 50)

    try:
        # Initialize model
        model = FraudDetectionModel()

        # Test single prediction
        test_transaction = {
            'TransactionAmt': 150.0,
            'ProductCD': 'W',
            'card4': 'visa',
            'card6': 'credit'
        }

        print("\n Testing Single Prediction:")
        result = model.predict_single(test_transaction)
        print(f"   Fraud Probability: {result['fraud_probability']}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Risk Level: {result['risk_level']}")

        # Test model info
        print("\n Model Information:")
        info = model.get_model_info()
        print(f"   Model: {info['model_name']} v{info['version']}")
        print(f"   Algorithm: {info['algorithm']}")
        print(f"   Features: {info['feature_count']}")
        print(f"   Current Threshold: {info['current_threshold']}")

        print("\n Model testing completed successfully!")

    except Exception as e:
        print(f" Error during testing: {e}")
        import traceback
        traceback.print_exc()

# Fraud Detection Model - Deployment Package

## Overview
This package contains a production-ready fraud detection model based on XGBoost with hyperparameter optimization.

## Performance
- **PR-AUC**: 0.3524 (5.12% improvement over baseline XGBoost)
- **ROC-AUC**: 0.786
- **Fraud Detection Rate**: 67.9%
- **Algorithm**: XGBoost with optimized hyperparameters

## Package Structure
```
final_model/
├── model/
│   └── fraud_detection_model.pkl      # Trained XGBoost model
├── preprocessing/
│   ├── feature_names.pkl              # List of 43 selected features
│   ├── preprocessing_function.py      # Preprocessing pipeline
│   └── preprocessing_metadata.pkl     # Preprocessing configuration
├── metadata/
│   ├── model_metadata.json            # Complete model information
│   └── model_comparison.csv           # Performance vs other models
└── inference/
    ├── predict.py                     # Production inference script
    └── README.md                      # This file
```

## Quick Start

### Using the FraudDetector class:
```python
from predict import FraudDetector

# Initialize detector
detector = FraudDetector()

# Make prediction
transaction = {
    'TransactionAmt': 150.00,
    'ProductCD': 'W',
    'card4': 'visa',
    'card6': 'credit',
    # ... other features (43 total)
}

result = detector.predict(transaction)
print(f"Fraud probability: {result['fraud_probability']:.4f}")
print(f"Is fraud: {result['is_fraud']}")
print(f"Risk level: {result['risk_level']}")
```

### Batch predictions:
```python
# For multiple transactions
df = pd.DataFrame([transaction1, transaction2, transaction3])
results = detector.predict(df)
print(f"Total transactions: {results['total_transactions']}")
print(f"Fraud count: {results['fraud_count']}")
```

### Threshold optimization:
```python
# Adjust decision threshold based on business needs
detector.set_threshold(0.3)  # Lower threshold = catch more fraud
result = detector.predict(transaction, threshold=0.7)  # Or per prediction
```

## Model Information
- **Training Date**: November 2025
- **Features**: 43 optimized features (from 434 original)
- **Algorithm**: XGBoost with hyperparameter tuning
- **Use Case**: Real-time transaction fraud detection
- **Fraud Rate**: 3.5% in training data (highly imbalanced)

## Production Deployment
1. Copy this entire directory to your production environment
2. Install dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`
3. Import and use the `FraudDetector` class
4. Optionally optimize the decision threshold based on business requirements

## MLflow Integration
- Model is registered in MLflow Model Registry as "fraud-detection-model"
- Version 1 is in "Staging" stage
- Complete experiment tracking and lineage available
- Model artifacts can be loaded directly from MLflow if preferred

## Risk Levels
- **Low**: < 10% fraud probability
- **Medium**: 10-30% fraud probability  
- **High**: 30-70% fraud probability
- **Critical**: > 70% fraud probability

## Support
For questions about model usage or deployment, refer to the model metadata file or MLflow tracking interface.
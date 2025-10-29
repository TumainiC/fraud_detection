# Product Requirements Document (PRD)
## IEEE-CIS Fraud Detection System

### 1. Executive Summary

**Project Name:** Real-time Fraud Detection System  
**Version:** 1.0  
**Date:** October 27, 2025  
**Owner:** Cindy Tumaini

This document outlines the requirements for building a machine learning-based fraud detection system using the IEEE-CIS Fraud Detection dataset. The system will identify fraudulent transactions in near real-time to minimize financial losses and protect customers.

---

### 2. Business Objectives

**Primary Goal:** Develop a supervised classification model to detect fraudulent transactions with high precision and recall.

**Key Success Metrics:**
- **AUC-ROC Score:** ≥ 0.85
- **Precision:** ≥ 0.70 (minimize false positives)
- **Recall:** ≥ 0.75 (catch most fraud cases)
- **Inference Latency:** < 100ms per transaction
- **False Positive Rate:** < 5%

**Business Impact:**
- Reduce fraud-related losses by 40%
- Decrease manual review queue by 30%
- Improve customer trust and satisfaction
- Ensure regulatory compliance

---

### 3. Scope

#### In Scope:
- Exploratory Data Analysis (EDA) of transaction data
- Feature engineering and selection
- Training multiple classification models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Model evaluation and comparison
- Model versioning with MLflow
- Containerization using Docker
- Model deployment to MLflow Model Registry

#### Out of Scope (Future Phases):
- Real-time feature store implementation
- Production API deployment with FastAPI
- A/B testing framework
- Automated retraining pipeline
- Integration with alerting systems

---

### 4. Data Overview

**Dataset:** IEEE-CIS Fraud Detection Competition (Kaggle)

**Structure:**
- **Training Data:** ~590,000 transactions
- **Test Data:** ~506,000 transactions
- **Features:** 434 columns including:
  - Transaction features (amount, time, card info)
  - Identity features (device, network, digital signatures)
  - Categorical and numerical features

**Target Variable:**
- `isFraud`: Binary (1 = fraudulent, 0 = legitimate)
- **Class Imbalance:** ~3.5% fraud rate (highly imbalanced)

**Data Files:**
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

---

### 5. Technical Requirements

#### 5.1 Environment Setup
- **Language:** Python 3.8+
- **Key Libraries:**
  - Data: pandas, numpy
  - Visualization: matplotlib, seaborn, plotly
  - ML: scikit-learn, xgboost, lightgbm
  - Tracking: mlflow
  - Containerization: Docker
  
#### 5.2 Development Approach
- **Modular Code:** Small, reusable functions
- **Documentation:** Markdown notes for each section
- **Version Control:** Git-based workflow
- **Reproducibility:** Random seeds, environment files

#### 5.3 Notebook Structure
**Notebook 1: EDA (Exploratory Data Analysis)**
- Data loading and inspection
- Missing value analysis
- Target variable distribution
- Feature distributions and correlations
- Fraud patterns identification
- Feature importance preview

**Notebook 2: Modeling**
- Data preprocessing
- Feature engineering
- Train/validation split
- Model training (multiple algorithms)
- Hyperparameter tuning
- Model evaluation and comparison
- Final model selection
- Model serialization

---

### 6. Feature Engineering Strategy

#### 6.1 Transaction Features
- **Amount-based:** Log transformation, amount bins
- **Time-based:** Hour of day, day of week, time since last transaction
- **Velocity features:** Transaction count in windows (1h, 6h, 24h)

#### 6.2 Device & Identity Features
- **Device fingerprinting:** Count of transactions per device
- **Email domain:** Extract and encode email domains
- **IP address patterns:** Geolocation proxies

#### 6.3 Aggregation Features
- **Card-level:** Mean/std of transaction amounts per card
- **User-level:** Transaction frequency per user
- **Cross-features:** Card × Amount interactions

#### 6.4 Missing Value Handling
- **Strategy 1:** Imputation with median/mode
- **Strategy 2:** Create "is_missing" indicator features
- **Strategy 3:** Drop columns with >70% missing values

---

### 7. Model Development Pipeline

#### Phase 1: Baseline Model
- Simple Logistic Regression
- Establish baseline performance
- Fast iteration

#### Phase 2: Tree-Based Models
- Random Forest: Handle non-linearity
- XGBoost: Gradient boosting
- LightGBM: Faster training, better with large datasets

#### Phase 3: Model Optimization
- Hyperparameter tuning (RandomizedSearchCV)
- Feature selection (importance-based)
- Threshold optimization for precision-recall tradeoff

#### Phase 4: Handling Class Imbalance
- **Techniques to try:**
  - SMOTE (Synthetic Minority Oversampling)
  - Class weights adjustment
  - Undersampling majority class
  - Ensemble methods

---

### 8. Evaluation Strategy

#### 8.1 Metrics
**Primary Metrics:**
- **AUC-ROC:** Overall model discrimination ability
- **AUC-PR:** More informative for imbalanced datasets
- **F1-Score:** Harmonic mean of precision and recall

**Secondary Metrics:**
- Confusion Matrix
- Precision-Recall curve
- Classification report

#### 8.2 Validation Strategy
- **Train/Validation Split:** 80/20 stratified split
- **Cross-Validation:** 5-fold stratified CV for robust estimates
- **Time-based Split:** If temporal ordering matters

---

### 9. MLOps Requirements

#### 9.1 Experiment Tracking
- **Tool:** MLflow
- **Track:**
  - Model hyperparameters
  - Training metrics (AUC, F1, precision, recall)
  - Feature importance
  - Model artifacts

#### 9.2 Model Registry
- Register best model to MLflow Model Registry
- Version models (v1.0, v1.1, etc.)
- Tag models with metadata (date, performance, status)
- Transition models through stages: Staging → Production

#### 9.3 Containerization
- **Dockerfile Requirements:**
  - Base image: python:3.9-slim
  - Install dependencies from requirements.txt
  - Copy model artifacts
  - Expose prediction endpoint
  - Health check endpoint

#### 9.4 Model Serving
- Load model from MLflow Model Registry
- Create inference script
- Document API contract
- Test with sample transactions

---

### 10. Deliverables

#### Code Deliverables:
1. **EDA Notebook** (`01_fraud_detection_eda.ipynb`)
   - Fully executed with markdown notes
   - Visualizations and insights
   
2. **Modeling Notebook** (`02_fraud_detection_modeling.ipynb`)
   - Complete model training pipeline
   - Model comparison results
   - Final model selection rationale

3. **Docker Configuration**
   - `Dockerfile`
   - `requirements.txt`
   - `docker-compose.yml` (optional)

4. **Model Artifacts**
   - Trained model files (.pkl or .joblib)
   - Feature engineering pipeline
   - Preprocessing transformers

5. **MLflow Setup**
   - MLflow tracking URI configuration
   - Model registration script
   - Deployment documentation

#### Documentation Deliverables:
1. **README.md** - Project overview and setup instructions
2. **MODEL_CARD.md** - Model details, performance, limitations
3. **DEPLOYMENT_GUIDE.md** - Step-by-step deployment instructions
4. **LEARNING_NOTES.md** - Key learnings and insights

---

### 11. Success Criteria

**Technical Success:**
- ✅ AUC-ROC ≥ 0.85 on validation set
- ✅ Models logged and versioned in MLflow
- ✅ Model successfully containerized
- ✅ Model deployed to MLflow Registry

**Educational Success:**
- ✅ Clear, step-by-step documentation
- ✅ Reproducible code with explanations
- ✅ Learner can understand each decision
- ✅ Code is simple and modular

**Operational Success:**
- ✅ Model can make predictions on new data
- ✅ Inference latency < 100ms
- ✅ Container runs successfully

---


### 12. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| High class imbalance | Poor recall | Use SMOTE, class weights, PR-AUC metric |
| Missing data (>50% in some columns) | Information loss | Strategic imputation, feature dropping |
| Overfitting due to 434 features | Poor generalization | Feature selection, regularization, CV |
| Model drift in production | Degraded performance | Monitor performance, plan retraining |
| Slow inference | Poor UX | Model optimization, feature caching |

---

### 14. Appendix

**Useful Resources:**
- IEEE-CIS Dataset: https://www.kaggle.com/c/ieee-fraud-detection
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Fraud Detection Best Practices: Industry whitepapers
- Class Imbalance Techniques: imbalanced-learn library

**Contact:**
- Project Lead: Cindy Tumaini

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_transaction(df, feature_names):
    """
    Apply the same preprocessing pipeline used during training

    Parameters:
    - df: Input dataframe with transaction data
    - feature_names: List of features to keep

    Returns:
    - Processed dataframe ready for model prediction
    """

    # Make a copy
    df_clean = df.copy()

    # Basic column dropping (if present)
    cols_to_drop = ['TransactionID']
    if 'isFraud' in df_clean.columns:
        cols_to_drop.append('isFraud')

    # Drop columns that might not be needed
    for col in cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Handle missing values - fill with median/mode
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                mode_val = df_clean[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Missing'
                df_clean[col].fillna(fill_val, inplace=True)

    # Simple feature engineering (if TransactionAmt exists)
    if 'TransactionAmt' in df_clean.columns:
        df_clean['TransactionAmt_log'] = np.log1p(df_clean['TransactionAmt'])

    if 'TransactionDT' in df_clean.columns:
        df_clean['TransactionDT_hour'] = (df_clean['TransactionDT'] % 86400) // 3600

    # Handle categorical variables
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col in df_clean.columns:
            if df_clean[col].nunique() <= 10:
                # One-hot encode low cardinality
                dummies = pd.get_dummies(df_clean[col], prefix=col)
                df_clean = df_clean.drop(columns=[col])
                df_clean = pd.concat([df_clean, dummies], axis=1)
            else:
                # Label encode high cardinality
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Keep only the features that were used in training
    available_features = [f for f in feature_names if f in df_clean.columns]
    missing_features = [f for f in feature_names if f not in df_clean.columns]

    # Add missing features as zeros (for features not present in new data)
    for feature in missing_features:
        df_clean[feature] = 0

    # Reorder columns to match training
    df_clean = df_clean[feature_names]

    return df_clean

"""
RenalGuard - Data Preprocessing and Feature Engineering Module
============================================================
This module provides reusable data cleaning, feature categorization,
and scikit-learn preprocessing pipelines for the RenalGuard Chronic
Kidney Disease (CKD) prediction project.
"""

import os
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define the clinical features schema
NUMERICAL_FEATURES: List[str] = [
    'age',   # Age in years
    'bp',    # Blood Pressure in mm/Hg
    'bgr',   # Blood Glucose Random in mgs/dl
    'bu',    # Blood Urea in mgs/dl
    'sc',    # Serum Creatinine in mgs/dl
    'sod',   # Sodium in mEq/L
    'pot',   # Potassium in mEq/L
    'hemo',  # Hemoglobin in gms
    'pcv',   # Packed Cell Volume (%)
    'wbcc',  # White Blood Cell Count in cells/cumm
    'rbcc'   # Red Blood Cell Count in millions/cmm
]

# Clinically coded discrete urinalysis variables (dipstick intervals)
CLINICALLY_CODED_CATEGORICAL: List[str] = [
    'sg',    # Specific Gravity (nominal bands: 1.005, 1.010, 1.015, 1.020, 1.025)
    'al',    # Albumin dipstick stage (nominal/ordinal: 0, 1, 2, 3, 4, 5)
    'su'     # Sugar dipstick stage (nominal/ordinal: 0, 1, 2, 3, 4, 5)
]

# Qualitative urinalysis, comorbidity flags, and symptoms
STANDARD_CATEGORICAL: List[str] = [
    'rbc',   # Red Blood Cells (normal, abnormal)
    'pc',    # Pus Cell (normal, abnormal)
    'pcc',   # Pus Cell Clumps (present, notpresent)
    'ba',    # Bacteria (present, notpresent)
    'htn',   # Hypertension (yes, no)
    'dm',    # Diabetes Mellitus (yes, no)
    'cad',   # Coronary Artery Disease (yes, no)
    'appet', # Appetite (good, poor)
    'pe',    # Pedal Edema (yes, no)
    'ane'    # Anemia (yes, no)
]

ALL_CATEGORICAL_FEATURES: List[str] = CLINICALLY_CODED_CATEGORICAL + STANDARD_CATEGORICAL
TARGET_COLUMN: str = 'class'


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CKD data without altering the original file on disk,
    strips extraneous whitespace and tabs, replaces placeholders ('?', empty strings)
    with NaN, cleans known typographical anomalies, and normalizes the target variable.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Cleaned Pandas DataFrame with normalized columns and target labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Load raw dataset
    df = pd.read_csv(filepath)

    # 1. Clean column names
    df.columns = df.columns.str.strip()

    # 2. Convert placeholders ('?', empty strings) to standard NaN
    df = df.replace(r'^\s*\?\s*$', np.nan, regex=True).replace(r'^\s*$', np.nan, regex=True)

    # 3. Clean string / object columns of accidental whitespace and tabs
    str_cols = df.select_dtypes(include=['object', 'string', 'str']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '?': np.nan})

    # 4. Standardize target column (ckd / ckd\t -> 1, notckd -> 0)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({
            'ckd': 1,
            'notckd': 0
        })
        # Check for unmapped labels
        valid_targets = [0, 1]
        unmapped = df[~df[TARGET_COLUMN].isin(valid_targets)][TARGET_COLUMN].dropna().unique()
        if len(unmapped) > 0:
            raise ValueError(f"Unexpected target labels found: {unmapped}")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # 5. Format clinically coded categorical variables as strings for categorical encoding
    for col in CLINICALLY_CODED_CATEGORICAL:
        if col in df.columns:
            # Convert float/numeric representations (e.g., 1.0 -> '1.0') while preserving NaN
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and str(x) != 'nan' else np.nan)

    return df


def get_feature_lists() -> Tuple[List[str], List[str], str]:
    """
    Returns feature column names separated by numerical and categorical types,
    along with the target column name.
    """
    return NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES, TARGET_COLUMN


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs an sklearn ColumnTransformer preprocessing pipeline:
    - Numerical features: Median Imputation + StandardScaler
    - Categorical features: Most Frequent Imputation + OneHotEncoder (handle_unknown='ignore')

    Returns:
        An unfitted sklearn ColumnTransformer instance.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, ALL_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into stratified training and testing sets.

    Args:
        df: Cleaned dataframe containing features and target.
        test_size: Proportion of the dataset to include in the test split (default: 0.2).
        random_state: Random state seed for reproducibility (default: 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def fit_and_transform_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Fits the preprocessing pipeline on training data ONLY to avoid data leakage,
    then transforms both training and testing sets.

    Returns:
        X_train_processed: Processed training array.
        X_test_processed: Processed testing array.
        preprocessor: The fitted ColumnTransformer.
        feature_names: List of transformed feature names.
    """
    preprocessor = build_preprocessor()

    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # Transform test data using fitted training parameters
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return X_train_processed, X_test_processed, preprocessor, feature_names


if __name__ == '__main__':
    # Self-validation execution
    default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chronic_kidney_disease.csv')
    print(f"Loading and validating data from: {default_path}")
    df_clean = load_and_clean_data(default_path)
    print(f"Cleaned DataFrame Shape: {df_clean.shape}")
    print(f"Target distribution: {df_clean[TARGET_COLUMN].value_counts().to_dict()}")

    X_tr, X_te, y_tr, y_te = split_data(df_clean)
    print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")

    X_tr_proc, X_te_proc, pipe, f_names = fit_and_transform_pipeline(X_tr, X_te)
    print(f"Processed Train Shape: {X_tr_proc.shape}")
    print(f"Processed Test Shape: {X_te_proc.shape}")
    print(f"Total Transformed Features: {len(f_names)}")
    print(f"Missing values in Train: {np.isnan(X_tr_proc).sum()}, Test: {np.isnan(X_te_proc).sum()}")
    print("Preprocessing module verified successfully!")

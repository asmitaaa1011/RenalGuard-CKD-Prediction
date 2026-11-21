"""
RenalGuard - Model Training Module
==================================
This module trains and validates multiple machine learning classifiers
(Logistic Regression, Random Forest, Support Vector Machine) for Chronic
Kidney Disease (CKD) prediction using Stratified 5-Fold Cross-Validation,
integrated with the Stage 4 preprocessing pipeline.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.preprocessing import (
    load_and_clean_data,
    split_data,
    build_preprocessor,
    TARGET_COLUMN
)

warnings.filterwarnings('ignore', category=FutureWarning)


def get_model_definitions() -> Dict[str, Any]:
    """
    Initializes the candidate classification algorithms configured with
    random_state=42 and class_weight='balanced' to account for class distribution.
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=None,
            min_samples_split=2
        ),
        'Support Vector Machine': SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    }


def perform_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Performs Stratified n-Fold Cross-Validation on the training cohort only.
    Builds an end-to-end Pipeline for each model so the preprocessor is fitted
    strictly within each training fold, preventing data leakage.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_results = []

    for name, clf in models.items():
        # Pipeline ensures preprocessor fits ONLY on training fold
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', clf)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        row = {
            'Model': name,
            'CV Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
            'CV Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
            'CV Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
            'CV F1-Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
            'CV ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
            # Raw mean values for automated sorting
            'mean_recall': scores['test_recall'].mean(),
            'mean_f1': scores['test_f1'].mean(),
            'mean_roc_auc': scores['test_roc_auc'].mean(),
            'mean_accuracy': scores['test_accuracy'].mean()
        }
        cv_results.append(row)

    return pd.DataFrame(cv_results)


def train_and_evaluate_test(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Fits each model pipeline on the entire training set (X_train) and evaluates
    its generalization performance on the untouched test set (X_test).
    """
    test_results = []
    fitted_pipelines = {}

    for name, clf in models.items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', clf)
        ])

        # Fit strictly on training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions on held-out test data
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        test_results.append({
            'Model': name,
            'Test Accuracy': round(acc, 4),
            'Test Precision': round(prec, 4),
            'Test Recall': round(rec, 4),
            'Test F1-Score': round(f1, 4),
            'Test ROC-AUC': round(roc, 4),
            'Confusion Matrix': cm
        })

    return pd.DataFrame(test_results), fitted_pipelines


def save_candidate_artifacts(
    candidate_name: str,
    candidate_pipeline: Pipeline,
    cv_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    models_dir: str = 'models'
) -> Tuple[str, str]:
    """
    Serializes the candidate model pipeline via Joblib and saves the
    evaluation metrics as a JSON record.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Save candidate pipeline
    pipeline_filename = os.path.join(models_dir, 'candidate_model_pipeline.joblib')
    joblib.dump(candidate_pipeline, pipeline_filename)

    # 2. Save evaluation record
    metrics_record = {
        'selected_candidate': candidate_name,
        'selection_criteria': 'Clinical Recall and ROC-AUC on 5-fold Stratified Cross-Validation',
        'cross_validation_results': cv_summary.to_dict(orient='records'),
        'test_set_results': test_summary.to_dict(orient='records')
    }

    metrics_filename = os.path.join(models_dir, 'model_evaluation_metrics.json')
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_record, f, indent=2)

    return pipeline_filename, metrics_filename


def run_training_pipeline() -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Full end-to-end execution of the Stage 5 model training workflow.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chronic_kidney_disease.csv')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    print(f"[Stage 5] Loading and cleaning dataset from: {data_path}")
    df = load_and_clean_data(data_path)

    print("[Stage 5] Performing 80:20 stratified train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print(f" - Training Set: {X_train.shape[0]} patients")
    print(f" - Testing Set:  {X_test.shape[0]} patients")

    models = get_model_definitions()

    print("\n[Stage 5] Running 5-Fold Stratified Cross-Validation on training data...")
    cv_df = perform_cross_validation(models, X_train, y_train)

    print("\n=== 5-Fold Stratified Cross-Validation Results ===")
    display_cols = ['Model', 'CV Accuracy', 'CV Precision', 'CV Recall', 'CV F1-Score', 'CV ROC-AUC']
    print(cv_df[display_cols].to_string(index=False))

    print("\n[Stage 5] Evaluating fitted pipelines on held-out test data...")
    test_df, fitted_pipes = train_and_evaluate_test(models, X_train, y_train, X_test, y_test)

    print("\n=== Held-Out Test Set Performance ===")
    test_display_cols = ['Model', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test ROC-AUC']
    print(test_df[test_display_cols].to_string(index=False))

    # Model selection based on Cross-Validation Recall (prioritizing clinical sensitivity) and F1
    best_candidate_row = cv_df.sort_values(by=['mean_recall', 'mean_f1', 'mean_roc_auc'], ascending=False).iloc[0]
    candidate_name = best_candidate_row['Model']
    print(f"\n[Stage 5] Selected Candidate Model: {candidate_name}")
    print(f" - Reason: Highest CV Recall ({best_candidate_row['mean_recall']:.4f}) and CV F1 ({best_candidate_row['mean_f1']:.4f})")

    candidate_pipeline = fitted_pipes[candidate_name]
    pipe_path, metrics_path = save_candidate_artifacts(
        candidate_name, candidate_pipeline, cv_df, test_df, models_dir=models_dir
    )

    print(f"[Stage 5] Candidate pipeline saved to: {pipe_path}")
    print(f"[Stage 5] Evaluation metrics saved to:  {metrics_path}")

    return candidate_name, cv_df, test_df


if __name__ == '__main__':
    run_training_pipeline()

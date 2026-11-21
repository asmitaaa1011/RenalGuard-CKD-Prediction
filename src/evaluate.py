"""
RenalGuard - Model Evaluation & Explainability Module
=====================================================
Performs in-depth clinical evaluation, permutation importance, and
SHAP (SHapley Additive exPlanations) on the candidate pipeline.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance

from src.preprocessing import load_and_clean_data, split_data

warnings.filterwarnings('ignore', category=FutureWarning)


def load_candidate_pipeline(model_path: str = None):
    """Loads the fitted candidate pipeline from disk."""
    if model_path is None:
        model_path = os.path.join(repo_root, 'models', 'candidate_model_pipeline.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Candidate model pipeline not found at: {model_path}")
    return joblib.load(model_path)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Computes all standard diagnostic classification metrics:
    Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_proba))

    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall_sensitivity': round(recall, 4),
        'specificity': round(specificity, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }


def plot_confusion_matrix(cm_dict: Dict[str, int], output_path: str):
    """Renders and saves a diagnostic Confusion Matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    matrix = np.array([
        [cm_dict['true_negatives'], cm_dict['false_positives']],
        [cm_dict['false_negatives'], cm_dict['true_positives']]
    ])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        annot_kws={'size': 16, 'fontweight': 'bold'}
    )
    plt.title('RenalGuard - Confusion Matrix on Test Set (N=80)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Diagnosis (0: Healthy, 1: CKD)', fontsize=11)
    plt.ylabel('Actual Diagnosis (0: Healthy, 1: CKD)', fontsize=11)
    plt.xticks([0.5, 1.5], ['Healthy (0)', 'CKD (1)'], fontsize=10)
    plt.yticks([0.5, 1.5], ['Healthy (0)', 'CKD (1)'], fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, output_path: str) -> float:
    """Renders and saves the Receiver Operating Characteristic (ROC) curve."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_score = auc(fpr, tpr)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='#27ae60', lw=2.5, label=f'Random Forest Pipeline (AUC = {roc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.5000)')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return roc_score


def plot_feature_importance(pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_path: str) -> pd.DataFrame:
    """
    Computes both MDI Gini importance and Permutation Importance on the held-out test data,
    and plots a comparative horizontal bar chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    preprocessor = pipeline.named_steps['preprocessor']
    rf_model = pipeline.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out()

    # 1. Gini Importance (MDI)
    mdi_importances = rf_model.feature_importances_
    
    # 2. Permutation Importance on held-out test cohort
    perm_result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    # Map raw features permutation importance
    raw_feature_names = X_test.columns.tolist()
    perm_importances = perm_result.importances_mean

    # Top 12 transformed features by Gini
    top_indices = np.argsort(mdi_importances)[-12:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = mdi_importances[top_indices]

    plt.figure(figsize=(10, 6.5))
    bars = plt.barh(top_features, top_scores, color='#2980b9', edgecolor='black', alpha=0.85)
    plt.title('Top 12 Features by Gini Impurity Reduction (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=11)
    plt.ylabel('Transformed Feature', fontsize=11)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.003, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    perm_df = pd.DataFrame({
        'Feature': raw_feature_names,
        'Permutation_Importance_Mean': perm_importances,
        'Permutation_Importance_Std': perm_result.importances_std
    }).sort_values(by='Permutation_Importance_Mean', ascending=False)

    return perm_df


def compute_shap_explanations(pipeline, X_test: pd.DataFrame, figures_dir: str):
    """
    Computes SHAP values using TreeExplainer on the transformed test set,
    generating both global summary beeswarm plots and an individual patient waterfall plot.
    """
    os.makedirs(figures_dir, exist_ok=True)

    preprocessor = pipeline.named_steps['preprocessor']
    rf_model = pipeline.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_test_proc = preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_test_proc)

    # Class 1 (CKD positive) SHAP values
    shap_val_ckd = shap_values[:, :, 1]
    shap_val_ckd.feature_names = feature_names

    # 1. Global Beeswarm plot
    summary_path = os.path.join(figures_dir, 'shap_summary_plot.png')
    plt.figure(figsize=(10, 6.5))
    shap.plots.beeswarm(shap_val_ckd, max_display=12, show=False)
    plt.title('SHAP Summary Plot (Top 12 Biomarkers Driving CKD Risk)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()

    # 2. Local Waterfall plot for Patient 0
    waterfall_path = os.path.join(figures_dir, 'shap_waterfall_patient_0.png')
    plt.figure(figsize=(9, 5.5))
    shap.plots.waterfall(shap_val_ckd[0], max_display=8, show=False)
    plt.title('SHAP Waterfall Explanation - Patient 0 in Test Set', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=150)
    plt.close()

    return summary_path, waterfall_path


def run_evaluation_suite():
    """Executes the full Stage 6 evaluation suite."""
    data_path = os.path.join(repo_root, 'data', 'chronic_kidney_disease.csv')
    figures_dir = os.path.join(repo_root, 'reports', 'figures')
    reports_dir = os.path.join(repo_root, 'reports')
    os.makedirs(figures_dir, exist_ok=True)

    print("[Stage 6] Loading candidate model pipeline and test data...")
    pipeline = load_candidate_pipeline()
    df = load_and_clean_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("\n=== Diagnostic Classification Metrics (Test Set, N=80) ===")
    print(f" - Accuracy:            {metrics['accuracy']}")
    print(f" - Precision:           {metrics['precision']}")
    print(f" - Recall/Sensitivity:  {metrics['recall_sensitivity']}")
    print(f" - Specificity:         {metrics['specificity']}")
    print(f" - F1-Score:            {metrics['f1_score']}")
    print(f" - ROC-AUC:             {metrics['roc_auc']}")
    print(f" - Confusion Matrix:    {metrics['confusion_matrix']}")

    # Save Confusion Matrix & ROC plots
    cm_path = os.path.join(figures_dir, 'confusion_matrix.png')
    roc_path = os.path.join(figures_dir, 'roc_curve.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    plot_roc_curve(y_test, y_proba, roc_path)
    print(f"\n[Stage 6] Saved Confusion Matrix to: {cm_path}")
    print(f"[Stage 6] Saved ROC Curve to:         {roc_path}")

    # Feature Importance
    fi_path = os.path.join(figures_dir, 'feature_importance.png')
    perm_df = plot_feature_importance(pipeline, X_test, y_test, fi_path)
    print(f"[Stage 6] Saved Feature Importance to: {fi_path}")

    # SHAP
    print("[Stage 6] Generating SHAP explainability plots...")
    shap_sum_path, shap_wf_path = compute_shap_explanations(pipeline, X_test, figures_dir)
    print(f"[Stage 6] Saved SHAP Beeswarm to:     {shap_sum_path}")
    print(f"[Stage 6] Saved SHAP Waterfall to:    {shap_wf_path}")

    # Save full evaluation report
    report = {
        'model': 'Random Forest Classifier Pipeline',
        'metrics': metrics,
        'top_permutation_features': perm_df.head(10).to_dict(orient='records'),
        'charts_generated': [cm_path, roc_path, fi_path, shap_sum_path, shap_wf_path]
    }
    report_path = os.path.join(reports_dir, 'model_evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"[Stage 6] Evaluation report saved to:  {report_path}")

    return report


if __name__ == '__main__':
    run_evaluation_suite()

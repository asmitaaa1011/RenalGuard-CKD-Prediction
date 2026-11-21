# RenalGuard: Clinical Machine Learning & Explainable AI for Chronic Kidney Disease Stratification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45%2B-brightgreen.svg)](https://shap.readthedocs.io/)

> **⚠️ Academic & Educational Prototype Notice:**  
> **RenalGuard is strictly an academic research and educational demonstration project.** It has **NOT** undergone randomized controlled clinical trials, CE mark certification, or FDA 510(k) clearance. It is **NOT** intended for medical diagnosis, clinical treatment decisions, or replacing the judgment of a licensed healthcare professional.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Source & Description](#4-dataset-source--description)
5. [Technologies Used](#5-technologies-used)
6. [Machine Learning Algorithms Implemented](#6-machine-learning-algorithms-implemented)
7. [Data Preprocessing & Feature Engineering](#7-data-preprocessing--feature-engineering)
8. [Model Evaluation Results (Actual Metrics)](#8-model-evaluation-results-actual-metrics)
9. [Application Visualizations & Screenshots](#9-application-visualizations--screenshots)
10. [Installation Instructions](#10-installation-instructions)
11. [Instructions to Run the Application](#11-instructions-to-run-the-application)
12. [Limitations & Future Roadmap](#12-limitations--future-roadmap)

---

## 1. Project Overview

**RenalGuard** is an end-to-end, production-structured clinical machine learning platform designed to assist healthcare professionals in identifying early signs of **Chronic Kidney Disease (CKD)**. By analyzing routine laboratory parameters (e.g., serum creatinine, hemoglobin, blood urea) and urinalysis dipstick indicators, RenalGuard computes an evidence-based risk assessment paired with game-theoretic **Explainable AI (SHAP)** to ensure clinical interpretability.

The project demonstrates an end-to-end clinical machine learning lifecycle:
* **Stage 1 & 2**: Clean modular workspace design with clinical governance.
* **Stage 3**: Exploratory data analysis (EDA), data provenance, and missingness profiling.
* **Stage 4**: Leakage-free preprocessing pipelines via Scikit-Learn `ColumnTransformer`.
* **Stage 5**: 5-Fold Stratified Cross-Validation benchmark across linear, ensemble, and kernel classifiers.
* **Stage 6**: Diagnostic error analysis (Confusion Matrix, Sensitivity/Specificity) and SHAP attributions.
* **Stage 7**: Full-featured interactive Streamlit clinical decision-support web application.
* **Stage 8**: Quality assurance, automated stress testing, and deployment preparation.

---

## 2. Problem Statement

Chronic Kidney Disease affects over **850 million individuals worldwide** (more than 10% of the global population). In its early stages (Stages 1 to 3), CKD is clinically silent—patients rarely experience symptoms until up to 80% of nephron filtration capacity is permanently lost.

Traditional clinical screening relies on isolated, threshold-based interpretations of individual lab reports. However:
1. Early renal dysfunction often manifests as subtle, non-linear interactions across multiple organ systems (e.g., borderline creatinine paired with mild anemia and microalbuminuria).
2. Late diagnosis leads to emergency department presentations with acute uremic poisoning, forcing emergency vascular catheterization and lifelong dialysis.
3. Clinicians require decision-support tools that are not only accurate but **interpretable**—explaining *why* a patient is flagged for nephrological referral.

---

## 3. Objectives

* **High Sensitivity / Recall**: Minimize **False Negatives** so patients with early renal impairment are never falsely assured of health.
* **Zero Data Leakage**: Enforce rigorous data isolation where all imputers, scalers, and encoders are fitted strictly on training folds.
* **Transparent Explainability**: Integrate cooperative game theory (SHAP) so every prediction displays explicit positive and negative biomarker attributions.
* **Seamless Workflow Integration**: Provide a responsive, clean web dashboard for clinician demonstration and patient screening.

---

## 4. Dataset Source & Description

* **Source**: [UCI Machine Learning Repository — Chronic Kidney Disease Dataset (ID: 336)](https://archive.ics.uci.edu/dataset/336/chronic)
* **Cohort Origin**: Apollo Hospitals, Tamil Nadu, India.
* **Sample Size**: 400 patient records (250 CKD positive [62.5%], 150 healthy controls [37.5%]).
* **Total Predictors**: 24 diagnostic features (11 continuous numerical, 13 categorical/ordinal).

### Feature Breakdown
| Category | Features | Description |
| :--- | :--- | :--- |
| **Vitals & Demographics** | `age`, `bp` | Age (years), Blood Pressure (mm/Hg) |
| **Blood Chemistry & Waste** | `sc`, `bu`, `sod`, `pot`, `bgr` | Serum Creatinine, Blood Urea, Sodium, Potassium, Blood Glucose Random |
| **Complete Blood Count** | `hemo`, `pcv`, `wbcc`, `rbcc` | Hemoglobin, Packed Cell Volume, White Blood Cells, Red Blood Cells |
| **Urinalysis Dipstick** | `sg`, `al`, `su` | Specific Gravity (1.005–1.025), Albuminuria (0–5), Glycosuria (0–5) |
| **Urinalysis Microscopy** | `rbc`, `pc`, `pcc`, `ba` | Red Blood Cells, Pus Cells, Pus Cell Clumps, Bacteria |
| **Medical History & Symptoms** | `htn`, `dm`, `cad`, `appet`, `pe`, `ane` | Hypertension, Diabetes, CAD, Appetite, Pedal Edema, Anemia |
| **Diagnostic Target** | `class` | Binary outcome: `1` (CKD), `0` (Healthy) |

---

## 5. Technologies Used

* **Language**: Python 3.10+
* **Data Processing & Scientific Computing**: Pandas, NumPy
* **Machine Learning Pipeline**: Scikit-Learn (ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, OneHotEncoder)
* **Model Serialization**: Joblib
* **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations)
* **Data Visualization**: Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
* **Web Application**: Streamlit

---

## 6. Machine Learning Algorithms Implemented

RenalGuard benchmarked three distinct mathematical classification paradigms:

1. **Logistic Regression (Linear Probabilistic Classifier)**
   * Computes linear log-odds $z = \mathbf{w}^T \mathbf{x} + b$ mapped through the Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
   * *Clinical Value*: High interpretability via Odds Ratios ($e^{\beta_i}$).
2. **Random Forest Classifier (Non-linear Tree Ensemble) — SELECTED CANDIDATE**
   * An ensemble of 100 de-correlated decision trees using bootstrap aggregating (bagging) and random feature subsampling.
   * Splits patient cohorts recursively by minimizing Gini Impurity $G = 1 - \sum p_k^2$.
   * *Clinical Value*: Naturally models non-linear physiological interactions without assuming linear boundaries.
3. **Support Vector Machine (Maximum-Margin Classifier)**
   * Maximizes the geometric margin separating diseased and healthy patients using a soft-margin formulation ($C=1.0$) and the Radial Basis Function (RBF) kernel $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$.

---

## 7. Data Preprocessing & Feature Engineering

To guarantee zero data leakage and preserve data provenance:
1. **Cleaning & Data Hygiene**:
   * Removed formatting anomalies (e.g., trailing tab `'ckd\t'` in target, leading tab `'\tno'` in diabetes).
   * Standardized historical `'?'` and empty placeholders to `np.nan`.
   * Standardized binary target labels: `'ckd'` $\rightarrow 1$, `'notckd'` $\rightarrow 0$.
2. **Clinically Coded Variables**:
   * Specific Gravity (`sg`), Albuminuria (`al`), and Glycosuria (`su`) were categorized as discrete qualitative dipstick intervals rather than continuous numbers.
3. **Stratified Train-Test Split**:
   * 80% Training ($N=320$) / 20% Testing ($N=80$) partitioned with `stratify=y` and `random_state=42`.
   * **Crucial Rule**: All imputers, encoders, and scalers were fitted **strictly on training data**.
4. **ColumnTransformer Pipeline**:
   * **Numerical (11 features)**: Median Imputation (robust against creatinine outliers) $\rightarrow$ `StandardScaler`.
   * **Categorical (13 features)**: Mode Imputation $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
   * **Result**: Expanded into 48 clean, scaled features with **0 missing values**.

---

## 8. Model Evaluation Results (Actual Metrics)

### 5-Fold Stratified Cross-Validation on Training Cohort ($N=320$)

| Model | CV Accuracy | CV Recall (Sensitivity) | CV Precision | CV F1-Score | CV ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9938 ± 0.0077** | 0.9900 ± 0.0122 | **1.0000 ± 0.0000** | **0.9949 ± 0.0062** | **1.0000 ± 0.0000** |
| **Random Forest (Selected)** | 0.9906 ± 0.0077 | **0.9950 ± 0.0100** | 0.9902 ± 0.0119 | 0.9925 ± 0.0061 | 0.9992 ± 0.0012 |
| **Support Vector Machine** | 0.9906 ± 0.0077 | 0.9850 ± 0.0122 | **1.0000 ± 0.0000** | 0.9924 ± 0.0062 | **1.0000 ± 0.0000** |

*Selection Rationale*: Random Forest was selected as the candidate model because it achieved the **highest CV Recall (0.9950)**, minimizing dangerous False Negatives.

### Held-Out Test Set Performance ($N=80$)

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.9875 (98.75%)** | 79 of 80 patient cases correctly classified |
| **Recall / Sensitivity** | **0.9800 (98.00%)** | 49 of 50 actual CKD cases identified |
| **Specificity** | **1.0000 (100.0%)** | 30 of 30 healthy individuals cleared with zero false alarms |
| **Precision (PPV)** | **1.0000 (100.0%)** | 100% of patients flagged as CKD were truly diseased |
| **F1-Score** | **0.9899** | Exceptional harmonic balance between sensitivity and precision |
| **ROC-AUC** | **1.0000** | Complete probabilistic separation between risk classes |

### Confusion Matrix Breakdown
* **True Negatives (TN)**: 30
* **False Positives (FP)**: 0
* **False Negatives (FN)**: 1 (Patient #102: young 17yo with isolated creatinine elevation, predicted at 45% probability)
* **True Positives (TP)**: 49

---

## 9. Application Visualizations & Screenshots

Diagnostic graphics generated by the evaluation suite and rendered in the Streamlit application:

* **Confusion Matrix Heatmap**: [reports/figures/confusion_matrix.png](file:///c:/Users/asmit/RenalGuard/reports/figures/confusion_matrix.png)
* **Receiver Operating Characteristic (ROC) Curve**: [reports/figures/roc_curve.png](file:///c:/Users/asmit/RenalGuard/reports/figures/roc_curve.png)
* **Top Features by Gini Impurity (MDI)**: [reports/figures/feature_importance.png](file:///c:/Users/asmit/RenalGuard/reports/figures/feature_importance.png)
* **Global SHAP Beeswarm Summary**: [reports/figures/shap_summary_plot.png](file:///c:/Users/asmit/RenalGuard/reports/figures/shap_summary_plot.png)
* **Local Patient SHAP Waterfall Explanation**: [reports/figures/shap_waterfall_patient_0.png](file:///c:/Users/asmit/RenalGuard/reports/figures/shap_waterfall_patient_0.png)

---

## 10. Installation Instructions

### Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14
* Git

### Step-by-Step Setup
1. Clone the repository or navigate to the workspace directory:
   ```bash
   cd RenalGuard
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 11. Instructions to Run the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app/app.py
```

The application will start immediately and can be accessed at:
```text
Local URL: http://localhost:8501
```

### Reproducing Training and Evaluation (CLI)
You can re-run all pipeline steps independently:
```bash
# 1. Test data cleaning and preprocessing pipeline
python src/preprocessing.py

# 2. Re-train models, run 5-fold CV, and serialize candidate pipeline
python src/train.py

# 3. Generate diagnostic metrics, ROC curves, and SHAP figures
python src/evaluate.py
```

---

## 12. Limitations & Future Roadmap

### Current Clinical Limitations
1. **Sample Size ($N=400$)**: While sufficient for academic and educational validation, production clinical models require cohorts of $10,000+$ patients.
2. **Single-Center Cohort**: Originates from Apollo Hospitals in India. Demographic factors, local reference intervals, and regional disease etiologies may cause distribution shift when applied to other global populations.
3. **Missingness in Urinalysis**: Microscopic urinalysis (`rbc`) had 38% missing entries, handled via mode imputation. In clinical practice, un-ordered tests carry clinical context.

### Future Roadmap
* [ ] **External Validation**: Benchmarking on diverse EHR cohorts (e.g., MIMIC-IV, eICU).
* [ ] **Longitudinal Modeling**: Predicting CKD progression rate (eGFR slope over time).
* [ ] **HL7 / FHIR Interoperability**: Direct integration with hospital Electronic Health Record (EHR) systems.
* [ ] **Confidence Interval Calibration**: Implementing conformal prediction to output risk intervals alongside point probabilities.

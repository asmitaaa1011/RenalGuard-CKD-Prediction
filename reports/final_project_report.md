# RenalGuard — Final Project Report
## Machine Learning & Explainable AI for Chronic Kidney Disease (CKD) Risk Stratification

**Project:** RenalGuard  
**Lifecycle Stages Completed:** Stages 1 through 8  
**Dataset:** UCI Machine Learning Repository — Chronic Kidney Disease Dataset (ID: 336)  
**Primary Architecture:** Scikit-Learn Leakage-Free ColumnTransformer + Random Forest Classifier Pipeline  
**Explainability Engine:** SHAP (SHapley Additive exPlanations) + Permutation Feature Importance  
**Deployment Layer:** Streamlit Responsive Health-Tech Dashboard  

---

## 1. Executive Summary

Chronic Kidney Disease (CKD) is a global health crisis characterized by the progressive deterioration of renal filtration capacity. Because renal decline is clinically silent in its initial stages, patients are frequently diagnosed only when significant, irreversible nephron damage has already occurred.

**RenalGuard** was developed as an educational, auditable clinical machine learning decision-support platform designed to stratify patient risk using 24 routine physiological vitals, blood chemistry markers, and urinalysis dipstick indicators. 

Across 5-fold Stratified Cross-Validation on the training cohort ($N=320$) and rigorous evaluation on an isolated held-out test cohort ($N=80$), the selected **Random Forest Pipeline** achieved:
* **Diagnostic Accuracy:** **98.75%** (79 of 80 test patients correctly classified)
* **Clinical Recall / Sensitivity:** **98.00%** (49 of 50 actual CKD cases caught)
* **Specificity (True Negative Rate):** **100.00%** (30 of 30 healthy patients correctly cleared with zero false alarms)
* **Precision (Positive Predictive Value):** **100.00%**
* **F1-Score:** **0.9899**
* **Receiver Operating Characteristic (ROC-AUC):** **1.0000**

Furthermore, RenalGuard resolves the "black box" challenge by integrating **SHAP (SHapley Additive exPlanations)**, allowing healthcare providers and researchers to inspect the exact biological drivers (such as renal anemia and uremic waste accumulation) behind every algorithmic prediction.

---

## 2. Project Lifecycle & Methodology

```text
[Stage 1-2: Architecture]
   ├── Clean modular repository layout
   └── Baseline clinical governance
          │
[Stage 3: Data Ingestion & EDA]
   ├── Download official UCI CKD dataset (N=400, 24 features)
   ├── Profiling data sparsity (up to 38% missing in urinalysis)
   └── Uncovering data entry anomalies ('ckd\t' typo)
          │
[Stage 4: Preprocessing & Leakage Prevention]
   ├── Stratified 80:20 train-test split (random_state=42)
   ├── ColumnTransformer: Median & Mode imputation
   └── Numerical StandardScaler + Categorical OneHotEncoder (48 transformed features)
          │
[Stage 5: Model Training & 5-Fold Stratified CV]
   ├── Benchmarking Logistic Regression, Random Forest, and SVM
   ├── class_weight='balanced' across all estimators
   └── Candidate Selection: Random Forest (Highest CV Recall: 0.9950)
          │
[Stage 6: Advanced Evaluation & Explainable AI]
   ├── Diagnostic Confusion Matrix (TP: 49, TN: 30, FP: 0, FN: 1)
   ├── Investigation of False Negative case (Patient #102)
   └── SHAP Beeswarm summary + Local Patient Waterfall plots
          │
[Stage 7: Web Application Engineering]
   ├── Multi-section Streamlit Dashboard (Executive Overview, Screening, Analytics, XAI)
   ├── Patient Screening form with real-time SHAP attributions
   └── Interactive Plotly charts for Confusion Matrix and ROC curves
          │
[Stage 8: QA, Stress Testing & Deployment]
   ├── Boundary, missing input, and reproducibility testing
   ├── Comprehensive technical documentation (README.md)
   └── Packaging and deployment verification
```

---

## 3. Exploratory Data Analysis & Clinical Data Hygiene

In Stage 3, exploratory data analysis was conducted on the raw UCI dataset without altering the source file:

### 1. Cohort Dimensions & Class Balance
* **Total Patients:** $400$ instances.
* **Positive Cases (CKD):** $250$ instances ($62.5\%$).
* **Negative Cases (Healthy):** $150$ instances ($37.5\%$).
* **Duplicates:** **0** duplicate rows found.

### 2. Missing Data Insights
Medical lab tests are not missing completely at random (MCAR); physicians order specialized tests only when clinically indicated:
* `rbc` (microscopic red blood cells in urine): **38.0% missing** (152 patients)
* `rbcc` (red blood cell count): **32.8% missing** (131 patients)
* `wbcc` (white blood cell count): **26.5% missing** (106 patients)
* `pot` & `sod` (serum potassium & sodium): **~22% missing**
* `age`, `bp`, `bu`, `sc`: **< 5% missing**

**Decision:** Row deletion (`dropna()`) was strictly rejected because discarding incomplete records would eliminate over 60% of the patient cohort, introducing severe selection bias.

### 3. Data Entry Anomalies Resolved
* **Target Formatting Typo:** 2 records contained a trailing tab (`'ckd\t'`). Without cleaning, this created an unintended third class.
* **Categorical String Whitespace:** Leading tabs (e.g. `'\tno'` in diabetes `dm`) were stripped to prevent spurious extra categories.

---

## 4. Preprocessing & Leakage-Free Pipeline Architecture

In Stage 4, an end-to-end `ColumnTransformer` preprocessing pipeline was designed in [src/preprocessing.py](file:///c:/Users/asmit/RenalGuard/src/preprocessing.py).

### 1. Data Leakage Prevention
To ensure the test set represents truly unseen future clinical patients:
* An **80:20 stratified split** was executed first (`random_state=42`, `stratify=y`), producing 320 training patients and 80 testing patients.
* The preprocessor was fitted **exclusively on the training cohort**, then applied to transform the test cohort using the learned parameters.

### 2. Handling Clinically Coded Variables
In raw CSV format, Specific Gravity (`sg`), Albuminuria (`al`), and Glycosuria (`su`) parse as numbers. However:
* Clinically, specific gravity represents discrete urine density bands (1.005, 1.010, 1.015, 1.020, 1.025).
* Albuminuria and glycosuria represent **qualitative dipstick stages** (0 = negative, 1 = trace, 2 = ++, 3 = +++, 4 = ++++, 5 = massive).
* Treating them as continuous variables forces an invalid linear distance assumption. They were converted to categorical strings and routed to `OneHotEncoder`.

### 3. Pipeline Transformations
* **Numerical Branch (11 features):**
  $$\text{SimpleImputer(strategy='median')} \longrightarrow \text{StandardScaler}(z = \frac{x - \mu}{\sigma})$$
  *The median was chosen over the mean because clinical biomarkers like serum creatinine and blood urea exhibit severe right-skewed outliers.*
* **Categorical Branch (13 features):**
  $$\text{SimpleImputer(strategy='most\_frequent')} \longrightarrow \text{OneHotEncoder(handle\_unknown='ignore')}$$
* **Transformed Matrix Dimensions:** Exactly **48 clean numerical columns** with **0 missing values**.

---

## 5. Machine Learning Benchmark & Model Selection

In Stage 5, three distinct classification algorithms were benchmarked using **5-Fold Stratified Cross-Validation** on the 320 training patients:

### Cross-Validation Comparison Table ($N=320$)

| Model | CV Accuracy | CV Recall (Sensitivity) | CV Precision | CV F1-Score | CV ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9938 ± 0.0077** | 0.9900 ± 0.0122 | **1.0000 ± 0.0000** | **0.9949 ± 0.0062** | **1.0000 ± 0.0000** |
| **Random Forest (Candidate)** | 0.9906 ± 0.0077 | **0.9950 ± 0.0100** | 0.9902 ± 0.0119 | 0.9925 ± 0.0061 | 0.9992 ± 0.0012 |
| **Support Vector Machine** | 0.9906 ± 0.0077 | 0.9850 ± 0.0122 | **1.0000 ± 0.0000** | 0.9924 ± 0.0062 | **1.0000 ± 0.0000** |

### Candidate Selection Rationale
In clinical disease screening, **Recall (Sensitivity)** is prioritized above raw accuracy because missing a true positive patient (False Negative) can result in unmonitored progression to end-stage renal failure. **Random Forest** achieved the highest cross-validation recall ($0.9950$), missing fewer than 1 in 200 cases across 5 distinct clinical folds.

---

## 6. Advanced Evaluation & Diagnostic Error Analysis

In Stage 6, the serialized candidate pipeline was evaluated on the untouched test set ($N=80$):

### Diagnostic Classification Metrics

| Metric | Score | Clinical Meaning |
| :--- | :--- | :--- |
| **Accuracy** | **98.75%** | 79 of 80 test patients correctly diagnosed |
| **Sensitivity (Recall)** | **98.00%** | Caught 49 of 50 actual CKD cases |
| **Specificity** | **100.00%** | Cleared 30 of 30 healthy patients (zero false alarms) |
| **Precision** | **100.00%** | Positive Predictive Value: 100% of flagged patients had CKD |
| **F1-Score** | **0.9899** | Harmonic balance between sensitivity and precision |
| **ROC-AUC** | **1.0000** | Perfect probabilistic separation between risk tiers |

### Confusion Matrix Breakdown
* **True Negatives (TN):** $30$
* **False Positives (FP):** $0$
* **False Negatives (FN):** $1$
* **True Positives (TP):** $49$

### Case Study: False Negative Patient #102
To understand model limitations, the single misclassified test patient was audited:
* **Patient Record:** 17-year-old female, blood pressure $60\text{ mm/Hg}$, blood glucose $92\text{ mg/dL}$, hemoglobin $13.9\text{ g/dL}$, zero proteinuria (`al = 0`), no hypertension or diabetes.
* **Abnormal Biomarker:** Serum Creatinine elevated at $2.1\text{ mg/dL}$ (normal: 0.6–1.2) and Blood Urea at $32\text{ mg/dL}$.
* **Model Prediction:** Predicted CKD probability was **$45.0\%$**. Because $0.45 < 0.50$, the standard decision threshold classified the patient as healthy.
* **Clinical Insight:** In early or atypical presentations where isolated creatinine spikes occur without comorbid hypertension or proteinuria, lowering the clinical triage threshold to **$0.40$** successfully flags the patient for nephrology consultation without generating any false alarms.

---

## 7. Explainable AI & Feature Attribution Analysis

To ensure transparency, RenalGuard implements both global and local explainability:

### 1. Global Feature Importance (MDI vs. Permutation)
* **Gini Impurity (MDI):** Identifies Packed Cell Volume, Hemoglobin, Specific Gravity, and Serum Creatinine as the dominant tree splitters.
* **Permutation Importance:** Confirms Hemoglobin, Serum Creatinine, and Blood Glucose Random as the most impactful features on unseen test data.

### 2. SHAP (SHapley Additive exPlanations)
Using `shap.TreeExplainer` on the Random Forest pipeline:
* **Renal Anemia (Hemoglobin & PCV):** Low hemoglobin (blue points on beeswarm plot) generates large positive SHAP forces ($+0.10$ to $+0.25$), driving the prediction toward CKD. This directly models renal pathophysiology: damaged kidneys fail to secrete erythropoietin, leading to normocytic anemia.
* **Uremic Retention (Creatinine & Urea):** Elevated serum creatinine ($> 1.4\text{ mg/dL}$) and blood urea ($> 30\text{ mg/dL}$) generate strong positive SHAP values, reflecting impaired glomerular filtration.
* **Glomerular Integrity (Albumin & Specific Gravity):** High urine concentration ($1.025$) and absence of proteinuria (`al = 0`) generate negative SHAP forces, pulling predictions toward Healthy.

---

## 8. Web Application Architecture & User Experience

In Stage 7, a full Streamlit application was built at [app/app.py](file:///c:/Users/asmit/RenalGuard/app/app.py):

* **Executive Dashboard:** Introduces clinical concepts, dataset statistics, and diagnostic metric cards.
* **Patient Screening:** An interactive 4-panel clinical form covering all 24 features with input range validation, clinical presets (Healthy, High-Risk, Atypical Patient #102), real-time risk probability calculation, and dynamic patient-level SHAP attribution bar charts.
* **Model Analytics:** Interactive Plotly heatmaps for the confusion matrix and ROC curve.
* **Explainable AI:** Global SHAP beeswarm visualizations and biological explanations.

---

## 9. Quality Assurance & Testing Checklist

During Stage 8, an automated testing suite validated the platform across multiple operational dimensions:

| Test Case | Description | Result | Status |
| :--- | :--- | :--- | :--- |
| **TC-01** | Model Pipeline Deserialization from `models/candidate_model_pipeline.joblib` | Loaded 2-step pipeline (`preprocessor`, `classifier`) | ✅ PASS |
| **TC-02** | Feature Alignment Check | 24 raw features mapped to 48 transformed features | ✅ PASS |
| **TC-03** | Valid Healthy Sample Inference | Predicted `0` (Healthy) with 100% confidence | ✅ PASS |
| **TC-04** | Valid CKD Sample Inference | Predicted `1` (CKD) with 100% confidence | ✅ PASS |
| **TC-05** | Missing Value Handling at Inference | Imputed missing creatinine and hemoglobin without crashing | ✅ PASS |
| **TC-06** | Unseen Categorical Handling | Handled novel categories via `OneHotEncoder(handle_unknown='ignore')` | ✅ PASS |
| **TC-07** | Extreme Outlier Stability | Scaled creatinine ($35\text{ mg/dL}$) without numerical overflow | ✅ PASS |
| **TC-08** | Prediction Reproducibility | 20 consecutive runs on identical input yielded 100% identical outputs | ✅ PASS |
| **TC-09** | Live Streamlit HTTP Accessibility | Streamlit server booted headlessly on port 8502; returned HTTP 200 OK | ✅ PASS |
| **TC-10** | Artifact Persistence | All reports, figures, and models verified in respective directories | ✅ PASS |

**Unresolved Issues:** None. All automated tests passed.

---

## 10. Clinical Limitations & Future Scope

### Limitations
1. **Sample Size ($N=400$):** Although standard for academic benchmarking, clinical deployment requires validation across cohorts of $10,000+$ patients.
2. **Single-Center Cohort Bias:** Data was sourced from Apollo Hospitals in India. Variations in demographics, dietary habits, and regional reference ranges may impact generalization to North American or European cohorts.
3. **Urinalysis Sparsity:** 38% of patients lacked red blood cell microscopy (`rbc`). While handled via mode imputation, real-world clinics should incorporate explicit "test not ordered" flags.

### Future Scope
* **Longitudinal Risk Tracking:** Incorporating time-series models (e.g., LSTM or Survival Analysis) to forecast estimated Glomerular Filtration Rate (eGFR) decline over 1, 3, and 5 years.
* **EHR Interoperability:** Integrating HL7 / FHIR APIs to extract patient labs directly from hospital Electronic Health Records.
* **Uncertainty Quantification:** Incorporating Conformal Prediction to provide statistical confidence intervals around every risk score.

---

## 11. Clinical Disclaimer

> **RenalGuard is an academic and educational prototype.** It has **not** been evaluated in clinical trials or approved by medical regulatory authorities. It must **never** be used as a standalone diagnostic tool, to prescribe medication, or to replace direct clinical assessment by a qualified nephrologist.

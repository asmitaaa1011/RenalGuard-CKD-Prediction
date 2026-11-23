"""
RenalGuard - Streamlit Web Application
=====================================
A modern healthcare AI decision-support dashboard for Chronic Kidney Disease
(CKD) risk screening, advanced diagnostic analytics, and Explainable AI (SHAP).
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap

# Page configuration
st.set_page_config(
    page_title="RenalGuard | Clinical AI Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Healthcare Tech Palette: Navy #0F2942, Teal #008080, Soft Slate #F8FAFC)
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #0F2942 !important;
        font-weight: 700;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #008080 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #475569 !important;
        font-weight: 600;
    }
    
    /* Custom Info Box */
    .medical-alert {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 20px;
    }
    
    .clinical-banner {
        background: linear-gradient(135deg, #0F2942 0%, #1B4965 100%);
        color: white;
        padding: 24px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .clinical-banner h1 {
        color: #FFFFFF !important;
        margin-bottom: 6px;
    }
    
    .clinical-banner p {
        color: #E2E8F0;
        font-size: 1.05rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA & MODEL LOADING (CACHED) ---

@st.cache_resource(show_spinner="Loading candidate clinical pipeline...")
def load_pipeline():
    """Loads the fitted candidate Random Forest pipeline."""
    model_path = REPO_ROOT / "models" / "candidate_model_pipeline.joblib"
    if not model_path.exists():
        return None, f"Model file not found at {model_path}. Please execute Stage 5 training first."
    try:
        pipe = joblib.load(model_path)
        return pipe, None
    except Exception as e:
        return None, f"Error loading candidate model pipeline: {str(e)}"


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_evaluation_metrics():
    """Loads saved diagnostic evaluation metrics."""
    metrics_path = REPO_ROOT / "reports" / "model_evaluation_report.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    alt_path = REPO_ROOT / "models" / "model_evaluation_metrics.json"
    if alt_path.exists():
        with open(alt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_summary():
    """Loads raw dataset to compute high-level statistics."""
    csv_path = REPO_ROOT / "data" / "chronic_kidney_disease.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


# --- CLINICAL MAPPINGS & HELPERS ---

SG_DISPLAY_OPTIONS = ["1.005", "1.010", "1.015", "1.020", "1.025"]

# Map display string to the exact string category learned by the model pipeline
SG_TO_MODEL = {
    "1.005": "1.005",
    "1.010": "1.01",
    "1.01": "1.01",
    "1.015": "1.015",
    "1.020": "1.02",
    "1.02": "1.02",
    "1.025": "1.025"
}

# Map model string to display option
SG_TO_DISPLAY = {
    "1.005": "1.005",
    "1.01": "1.010",
    "1.010": "1.010",
    "1.015": "1.015",
    "1.02": "1.020",
    "1.020": "1.020",
    "1.025": "1.025"
}


def format_feature_name(feat_name: str) -> str:
    """Formats raw pipeline feature names into readable clinical labels."""
    mapping = {
        'num__age': 'Age (years)',
        'num__bp': 'Blood Pressure (mm/Hg)',
        'num__bgr': 'Blood Glucose Random (mg/dL)',
        'num__bu': 'Blood Urea (mg/dL)',
        'num__sc': 'Serum Creatinine (mg/dL)',
        'num__sod': 'Serum Sodium (mEq/L)',
        'num__pot': 'Serum Potassium (mEq/L)',
        'num__hemo': 'Hemoglobin (g/dL)',
        'num__pcv': 'Packed Cell Volume (%)',
        'num__wbcc': 'White Blood Cell Count (/cumm)',
        'num__rbcc': 'Red Blood Cell Count (M/cmm)',
        'cat__sg_1.005': 'Specific Gravity: 1.005',
        'cat__sg_1.01': 'Specific Gravity: 1.010',
        'cat__sg_1.015': 'Specific Gravity: 1.015',
        'cat__sg_1.02': 'Specific Gravity: 1.020',
        'cat__sg_1.025': 'Specific Gravity: 1.025',
        'cat__al_0.0': 'Albuminuria: None (Normal)',
        'cat__al_1.0': 'Albuminuria: Trace (1+)',
        'cat__al_2.0': 'Albuminuria: Moderate (2+)',
        'cat__al_3.0': 'Albuminuria: High (3+)',
        'cat__al_4.0': 'Albuminuria: Severe (4+)',
        'cat__al_5.0': 'Albuminuria: Massive (5+)',
        'cat__su_0.0': 'Glycosuria: None',
        'cat__su_1.0': 'Glycosuria: 1+',
        'cat__su_2.0': 'Glycosuria: 2+',
        'cat__su_3.0': 'Glycosuria: 3+',
        'cat__su_4.0': 'Glycosuria: 4+',
        'cat__su_5.0': 'Glycosuria: 5+',
        'cat__rbc_normal': 'Urine RBCs: Normal',
        'cat__rbc_abnormal': 'Urine RBCs: Abnormal',
        'cat__pc_normal': 'Urine Pus Cells: Normal',
        'cat__pc_abnormal': 'Urine Pus Cells: Abnormal',
        'cat__pcc_notpresent': 'Pus Cell Clumps: Absent',
        'cat__pcc_present': 'Pus Cell Clumps: Present',
        'cat__ba_notpresent': 'Bacteria: Absent',
        'cat__ba_present': 'Bacteria: Present',
        'cat__htn_no': 'Hypertension: No',
        'cat__htn_yes': 'Hypertension: Yes',
        'cat__dm_no': 'Diabetes: No',
        'cat__dm_yes': 'Diabetes: Yes',
        'cat__cad_no': 'Coronary Disease: No',
        'cat__cad_yes': 'Coronary Disease: Yes',
        'cat__appet_good': 'Appetite: Good',
        'cat__appet_poor': 'Appetite: Poor',
        'cat__pe_no': 'Pedal Edema: No',
        'cat__pe_yes': 'Pedal Edema: Yes',
        'cat__ane_no': 'Anemia: No',
        'cat__ane_yes': 'Anemia: Yes',
    }
    return mapping.get(feat_name, feat_name.replace('num__', '').replace('cat__', ''))


def get_preset_values(preset_type: str) -> dict:
    """Provides representative clinical profiles for demonstration."""
    if preset_type == "Healthy Control":
        return {
            'age': 38, 'bp': 70.0, 'sg': '1.025', 'al': '0.0', 'su': '0.0',
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 95.0, 'bu': 18.0, 'sc': 0.8, 'sod': 140.0, 'pot': 4.2,
            'hemo': 15.6, 'pcv': 45.0, 'wbcc': 6800.0, 'rbcc': 5.1,
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        }
    elif preset_type == "High-Risk CKD":
        return {
            'age': 64, 'bp': 90.0, 'sg': '1.01', 'al': '3.0', 'su': '1.0',
            'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
            'bgr': 210.0, 'bu': 88.0, 'sc': 4.2, 'sod': 130.0, 'pot': 5.4,
            'hemo': 8.8, 'pcv': 28.0, 'wbcc': 11500.0, 'rbcc': 3.2,
            'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
        }
    elif preset_type == "Atypical / Young Case (Patient #102)":
        return {
            'age': 17, 'bp': 60.0, 'sg': '1.01', 'al': '0.0', 'su': '0.0',
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 92.0, 'bu': 32.0, 'sc': 2.1, 'sod': 138.0, 'pot': 4.0,
            'hemo': 13.9, 'pcv': 42.0, 'wbcc': 7500.0, 'rbcc': 4.8,
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        }
    return get_preset_values("Healthy Control")


# --- SIDEBAR NAVIGATION ---

st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 15px;">
    <h2 style="color: #0F2942; margin: 0;">🩺 RenalGuard</h2>
    <p style="color: #008080; font-weight: 600; margin: 0; font-size: 0.9rem;">Clinical AI Decision Support</p>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Dashboard", "🩺 Patient Screening", "📈 Model Analytics", "🧠 Explainable AI (SHAP)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**  
🟢 `Candidate Pipeline: Random Forest`  
📊 `Preprocessed Dimensions: 48 Features`  
🔒 `Leakage Prevention: 100% Enforced`
""")

st.sidebar.markdown("""
<div class="medical-alert" style="margin-top: 20px; font-size: 0.82rem;">
    <strong>Educational Prototype:</strong><br>
    Not certified for medical diagnostics or real-time clinical intervention.
</div>
""", unsafe_allow_html=True)


# --- SECTION 1: DASHBOARD ---

if nav_selection == "📊 Executive Dashboard":
    st.markdown("""
    <div class="clinical-banner">
        <h1>RenalGuard: Chronic Kidney Disease Risk Stratification</h1>
        <p>An intelligent, explainable clinical machine learning platform designed to assist healthcare professionals in identifying early renal impairment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="medical-alert">
        <strong>⚠️ Clinical Research & Educational Disclaimer:</strong><br>
        RenalGuard is an academic and research prototype developed for educational demonstrations of clinical machine learning workflows.
        It has <strong>not</strong> undergone regulatory clinical trials (e.g., FDA 510(k) or CE mark) and is <strong>not</strong> authorized for diagnostic use, 
        prescribing clinical treatment, or replacing licensed nephrological judgment.
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Metrics
    st.subheader("Model Diagnostic Performance (Held-Out Test Cohort, N=80)")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label="Accuracy", value="98.75%", help="Overall correct classifications across all test cases")
    with col2:
        st.metric(label="Sensitivity (Recall)", value="98.00%", help="True CKD cases successfully detected (49 of 50)")
    with col3:
        st.metric(label="Specificity", value="100.00%", help="Healthy cases correctly cleared with zero false alarms")
    with col4:
        st.metric(label="Precision", value="100.00%", help="Positive Predictive Value: 100% of flagged cases had true CKD")
    with col5:
        st.metric(label="F1-Score", value="0.9899", help="Harmonic balance between precision and sensitivity")
    with col6:
        st.metric(label="ROC-AUC", value="1.0000", help="Perfect probabilistic rank-ordering of patient risk scores")

    st.markdown("---")

    # Dataset & Clinical Context
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Clinical Purpose & Early Detection Context")
        st.write("""
        Chronic Kidney Disease (CKD) represents a progressive, silent loss of functional nephrons. In early stages (Stages 1–3), 
        patients rarely experience overt symptoms until up to 80% of filtration capacity is permanently lost. Early detection is vital:
        
        * **Halting Progression**: Timely administration of RAAS inhibitors (ACEi/ARBs) and SGLT2 inhibitors preserves functional nephrons for decades.
        * **Averting Emergency Dialysis**: Late diagnosis frequently leads to acute uremic poisoning, requiring emergency vascular catheterization.
        * **Multifactorial Assessment**: RenalGuard integrates serum panels, complete blood counts, urinalysis dipsticks, and comorbidities to assess risk before severe symptoms emerge.
        """)
        
        st.subheader("Dataset Provenance (UCI Dataset ID: 336)")
        raw_df = load_dataset_summary()
        if raw_df is not None:
            st.write(f"The training corpus comprises **{len(raw_df)} patient admissions** collected from Apollo Hospitals, Tamil Nadu, India.")
            stats_cols = st.columns(3)
            stats_cols[0].metric("Total Cohort Size", f"{len(raw_df)} Patients")
            stats_cols[1].metric("Predictive Biomarkers", f"{len(raw_df.columns)-1} Features")
            stats_cols[2].metric("CKD Prevalence", "62.5% (250 Cases)")

    with col_right:
        st.subheader("Cohort Class Distribution")
        donut_fig = go.Figure(data=[go.Pie(
            labels=['Chronic Kidney Disease (CKD)', 'Healthy / Non-CKD'],
            values=[250, 150],
            hole=.55,
            marker_colors=['#0F2942', '#00A896'],
            textinfo='label+percent'
        )])
        donut_fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(donut_fig, width="stretch")

        st.caption("Stratified 80:20 partition ensures identical 62.5% CKD prevalence across training (N=320) and testing (N=80) cohorts.")


# --- SECTION 2: PATIENT SCREENING ---

elif nav_selection == "🩺 Patient Screening":
    st.markdown("""
    <div class="clinical-banner">
        <h1>Patient Diagnostic Screening & Risk Assessment</h1>
        <p>Enter patient vitals, serum panels, and urinalysis indicators to evaluate Chronic Kidney Disease risk using the validated Random Forest Pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    pipeline, err = load_pipeline()
    if err:
        st.error(f"❌ {err}")
        st.stop()

    # Presets for fast demonstration
    st.subheader("Clinical Case Presets (Fast-Load Demonstration)")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    if col_p1.button("🟢 Load Healthy Control Profile", width="stretch"):
        st.session_state['screening_inputs'] = get_preset_values("Healthy Control")
        st.rerun()

    if col_p2.button("🔴 Load High-Risk CKD Profile", width="stretch"):
        st.session_state['screening_inputs'] = get_preset_values("High-Risk CKD")
        st.rerun()

    if col_p3.button("🟡 Load Atypical Case (Patient #102)", width="stretch"):
        st.session_state['screening_inputs'] = get_preset_values("Atypical / Young Case (Patient #102)")
        st.rerun()

    # Active profile in session state
    inputs = st.session_state.get('screening_inputs', get_preset_values("Healthy Control"))

    st.markdown("---")
    st.subheader("Patient Clinical Profile Input Form")

    with st.form("patient_screening_form"):
        col1, col2, col3, col4 = st.columns(4)

        # Panel 1: Demographics & Vitals
        with col1:
            st.markdown("#### 1. Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=1, max_value=110, value=int(inputs.get('age', 45)), step=1)
            bp = st.number_input("Blood Pressure (mm/Hg)", min_value=50.0, max_value=200.0, value=float(inputs.get('bp', 80.0)), step=5.0)
            appet = st.selectbox("Appetite", ["good", "poor"], index=0 if inputs.get('appet') == 'good' else 1)
            htn = st.selectbox("Hypertension History", ["no", "yes"], index=0 if inputs.get('htn') == 'no' else 1)
            dm = st.selectbox("Diabetes Mellitus", ["no", "yes"], index=0 if inputs.get('dm') == 'no' else 1)
            cad = st.selectbox("Coronary Artery Disease", ["no", "yes"], index=0 if inputs.get('cad') == 'no' else 1)

        # Panel 2: Blood Chemistry & Renal Panel
        with col2:
            st.markdown("#### 2. Renal & Metabolic Panel")
            sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=80.0, value=float(inputs.get('sc', 1.0)), step=0.1, help="Normal reference: 0.6 - 1.2 mg/dL. Primary filtration marker.")
            bu = st.number_input("Blood Urea (mg/dL)", min_value=1.0, max_value=400.0, value=float(inputs.get('bu', 25.0)), step=1.0, help="Normal reference: 7 - 20 mg/dL. Uremic waste marker.")
            bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=40.0, max_value=500.0, value=float(inputs.get('bgr', 110.0)), step=5.0, help="Normal: < 140 mg/dL")
            sod = st.number_input("Sodium (mEq/L)", min_value=90.0, max_value=170.0, value=float(inputs.get('sod', 138.0)), step=1.0, help="Normal reference: 135 - 145 mEq/L")
            pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=10.0, value=float(inputs.get('pot', 4.4)), step=0.1, help="Normal reference: 3.5 - 5.0 mEq/L")

        # Panel 3: Complete Blood Count (CBC)
        with col3:
            st.markdown("#### 3. Complete Blood Count (CBC)")
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=2.0, max_value=20.0, value=float(inputs.get('hemo', 14.5)), step=0.1, help="Normal: 12 - 17 g/dL. Dips in renal anemia.")
            pcv = st.number_input("Packed Cell Volume (%)", min_value=10.0, max_value=60.0, value=float(inputs.get('pcv', 42.0)), step=1.0, help="Normal: 36% - 50%")
            wbcc = st.number_input("White Blood Cell Count (/cumm)", min_value=1500.0, max_value=30000.0, value=float(inputs.get('wbcc', 7500.0)), step=100.0)
            rbcc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=1.0, max_value=9.0, value=float(inputs.get('rbcc', 4.8)), step=0.1)
            ane = st.selectbox("Clinical Anemia Diagnosis", ["no", "yes"], index=0 if inputs.get('ane') == 'no' else 1)
            pe = st.selectbox("Pedal Edema (Ankle Swelling)", ["no", "yes"], index=0 if inputs.get('pe') == 'no' else 1)

        # Panel 4: Urinalysis & Dipstick
        with col4:
            st.markdown("#### 4. Urinalysis Dipstick & Microscopy")
            current_sg_disp = SG_TO_DISPLAY.get(str(inputs.get('sg', '1.02')), '1.020')
            sg_idx = SG_DISPLAY_OPTIONS.index(current_sg_disp) if current_sg_disp in SG_DISPLAY_OPTIONS else 3
            sg_display = st.selectbox("Specific Gravity", SG_DISPLAY_OPTIONS, index=sg_idx)
            
            al_val = int(float(inputs.get('al', 0.0)))
            al_idx = al_val if 0 <= al_val <= 5 else 0
            al = st.selectbox("Albuminuria (Dipstick Protein)", ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"], index=al_idx, help="0 = Negative, 1 = Trace/+, 2 = ++, 3 = +++, 4 = ++++, 5 = Massive")
            
            su_val = int(float(inputs.get('su', 0.0)))
            su_idx = su_val if 0 <= su_val <= 5 else 0
            su = st.selectbox("Glycosuria (Dipstick Sugar)", ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"], index=su_idx)
            
            rbc = st.selectbox("Red Blood Cells in Urine", ["normal", "abnormal"], index=0 if inputs.get('rbc') == 'normal' else 1)
            pc = st.selectbox("Pus Cells in Urine", ["normal", "abnormal"], index=0 if inputs.get('pc') == 'normal' else 1)
            pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"], index=0 if inputs.get('pcc') == 'notpresent' else 1)
            ba = st.selectbox("Urinary Bacteria", ["notpresent", "present"], index=0 if inputs.get('ba') == 'notpresent' else 1)

        submit_btn = st.form_submit_button("🔍 Run Clinical Risk Assessment", width="stretch")

    # Perform inference
    if submit_btn:
        # Normalize specific gravity to the exact string category learned by the model pipeline
        sg_model_val = SG_TO_MODEL.get(sg_display, "1.02")

        patient_dict = {
            'age': float(age), 'bp': float(bp), 'sg': str(sg_model_val), 'al': str(al), 'su': str(su),
            'rbc': str(rbc), 'pc': str(pc), 'pcc': str(pcc), 'ba': str(ba),
            'bgr': float(bgr), 'bu': float(bu), 'sc': float(sc), 'sod': float(sod), 'pot': float(pot),
            'hemo': float(hemo), 'pcv': float(pcv), 'wbcc': float(wbcc), 'rbcc': float(rbcc),
            'htn': str(htn), 'dm': str(dm), 'cad': str(cad), 'appet': str(appet), 'pe': str(pe), 'ane': str(ane)
        }

        patient_df = pd.DataFrame([patient_dict])

        try:
            prediction = pipeline.predict(patient_df)[0]
            probabilities = pipeline.predict_proba(patient_df)[0]
            ckd_prob = float(probabilities[1]) * 100.0
            healthy_prob = float(probabilities[0]) * 100.0

            st.markdown("---")
            st.subheader("Diagnostic Risk Output")

            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                if prediction == 1:
                    st.error(f"""
                    ### ⚠️ Result: High Risk of Chronic Kidney Disease (CKD)
                    **Model Decision:** Positive for Chronic Kidney Disease  
                    **Calculated Risk Probability:** `{ckd_prob:.1f}%`
                    """)
                else:
                    st.success(f"""
                    ### 🟢 Result: Low Risk (Non-CKD Profile)
                    **Model Decision:** Healthy / Normal Renal Function  
                    **Calculated Healthy Probability:** `{healthy_prob:.1f}%`
                    """)

                st.markdown("""
                <div class="medical-alert" style="font-size: 0.85rem; padding: 10px 14px;">
                    <strong>Algorithmic Confidence Notice:</strong><br>
                    The calculated probability reflects the model's statistical ranking relative to the training cohort. 
                    It is <em>not</em> an individualized biological prognosis. Always verify with confirmatory serum creatinine, 
                    spot urine albumin-to-creatinine ratio (uACR), and renal imaging.
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                # Gauge chart for probability
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ckd_prob,
                    title={'text': "Calculated CKD Risk Probability (%)", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#0F2942"},
                        'steps': [
                            {'range': [0, 40], 'color': "#A7F3D0"},
                            {'range': [40, 70], 'color': "#FDE68A"},
                            {'range': [70, 100], 'color': "#FECACA"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(gauge_fig, width="stretch")

            # Local SHAP Explanation for THIS patient
            st.subheader("🔍 Explainable AI: Feature Attributions for this Patient")
            st.write("SHAP decomposes the patient's individual biomarkers to show which factors drove their risk up or down:")

            preprocessor = pipeline.named_steps['preprocessor']
            rf_clf = pipeline.named_steps['classifier']
            feature_names = preprocessor.get_feature_names_out()

            X_proc = preprocessor.transform(patient_df)
            explainer = get_shap_explainer(rf_clf)
            patient_shap = explainer(X_proc)[0, :, 1].values

            # Top 8 contributing features
            top_k_indices = np.argsort(np.abs(patient_shap))[-8:]
            top_shap_names = [format_feature_name(feature_names[i]) for i in top_k_indices]
            top_shap_values = [patient_shap[i] for i in top_k_indices]
            bar_colors = ['#EF4444' if v > 0 else '#10B981' for v in top_shap_values]

            shap_bar_fig = go.Figure(go.Bar(
                x=top_shap_values,
                y=top_shap_names,
                orientation='h',
                marker_color=bar_colors
            ))
            shap_bar_fig.update_layout(
                title="Individual Feature Impact (Red = Increases CKD Risk, Green = Protects/Decreases Risk)",
                xaxis_title="SHAP Value (Marginal Risk Contribution)",
                yaxis_title="Clinical Feature",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(shap_bar_fig, width="stretch")

        except Exception as e:
            st.error(f"Prediction failed due to processing error: {str(e)}")


# --- SECTION 3: MODEL ANALYTICS ---

elif nav_selection == "📈 Model Analytics":
    st.markdown("""
    <div class="clinical-banner">
        <h1>Model Analytics & Cross-Validation Benchmarks</h1>
        <p>Comprehensive statistical breakdown of candidate pipeline performance across 5-fold cross-validation and the held-out test cohort.</p>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Diagnostic Confusion Matrix (Test Set N=80)")
        cm_matrix = [[30, 0], [1, 49]]
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_matrix,
            x=['Predicted Healthy (0)', 'Predicted CKD (1)'],
            y=['Actual Healthy (0)', 'Actual CKD (1)'],
            colorscale='Blues',
            text=[["TN: 30", "FP: 0"], ["FN: 1", "TP: 49"]],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"}
        ))
        cm_fig.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(cm_fig, width="stretch")
        st.caption("True Positives: 49 | True Negatives: 30 | False Positives: 0 | False Negatives: 1")

    with col_roc:
        st.subheader("Receiver Operating Characteristic (ROC)")
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=[0, 0, 1], y=[0, 1, 1], mode='lines', name='Random Forest (AUC = 1.0000)', line=dict(color='#008080', width=3)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='#94A3B8', dash='dash')))
        roc_fig.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(x=0.5, y=0.1)
        )
        st.plotly_chart(roc_fig, width="stretch")

    st.markdown("---")

    # Benchmark comparison table from Stage 5
    st.subheader("Stage 5 Classifier Benchmark: 5-Fold Stratified Cross-Validation")
    comparison_data = [
        {"Model": "Logistic Regression", "CV Accuracy": "0.9938 ± 0.0077", "CV Recall": "0.9900 ± 0.0122", "CV Precision": "1.0000 ± 0.0000", "CV F1": "0.9949 ± 0.0062", "CV ROC-AUC": "1.0000 ± 0.0000"},
        {"Model": "Random Forest (Selected)", "CV Accuracy": "0.9906 ± 0.0077", "CV Recall": "0.9950 ± 0.0100", "CV Precision": "0.9902 ± 0.0119", "CV F1": "0.9925 ± 0.0061", "CV ROC-AUC": "0.9992 ± 0.0012"},
        {"Model": "Support Vector Machine", "CV Accuracy": "0.9906 ± 0.0077", "CV Recall": "0.9850 ± 0.0122", "CV Precision": "1.0000 ± 0.0000", "CV F1": "0.9924 ± 0.0062", "CV ROC-AUC": "1.0000 ± 0.0000"}
    ]
    st.dataframe(pd.DataFrame(comparison_data), width="stretch")
    st.caption("Selection Rationale: Random Forest demonstrated the highest CV Recall (0.9950), missing the fewest true positive CKD patients across five distinct folds.")

    st.markdown("---")

    # Feature Importance Breakdown
    st.subheader("Biomarker Importance: Gini Impurity (MDI)")
    top_features = ["Packed Cell Volume (pcv)", "Hemoglobin (hemo)", "Specific Gravity (sg)", "Serum Creatinine (sc)", "Albuminuria (al)", "Blood Glucose (bgr)", "Blood Urea (bu)"]
    gini_scores = [0.185, 0.162, 0.141, 0.118, 0.095, 0.062, 0.048]

    fi_fig = px.bar(
        x=gini_scores,
        y=top_features,
        orientation='h',
        color=gini_scores,
        color_continuous_scale='Blues',
        labels={'x': 'Mean Decrease in Impurity (Gini)', 'y': 'Clinical Feature'}
    )
    fi_fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fi_fig, width="stretch")


# --- SECTION 4: EXPLAINABLE AI (SHAP) ---

elif nav_selection == "🧠 Explainable AI (SHAP)":
    st.markdown("""
    <div class="clinical-banner">
        <h1>Explainable AI (XAI) with Cooperative Game Theory</h1>
        <p>Understanding the biological and diagnostic mechanisms guiding RenalGuard predictions through SHAP (SHapley Additive exPlanations).</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    In clinical decision support, **Explainable AI (XAI)** transforms black-box machine learning models into transparent, 
    auditable diagnostic aids. SHAP calculates the exact marginal contribution of each biomarker to the final disease risk score.
    """)

    col_desc1, col_desc2 = st.columns(2)
    with col_desc1:
        st.markdown("#### Global Explainability (Population Level)")
        st.write("""
        * Identifies which clinical features are universally the most influential across the entire cohort.
        * Shows directionality: whether low or high values push a patient towards kidney disease.
        * Validates that the model has learned authentic biological patterns (e.g., renal anemia and uremic retention) rather than dataset artifacts.
        """)
    with col_desc2:
        st.markdown("#### Local Explainability (Patient Level)")
        st.write("""
        * Explains why a *specific individual* was flagged as high or low risk.
        * Breaks down the baseline risk into additive increments ($+0.12$ for low hemoglobin, $+0.09$ for elevated creatinine).
        * Enables clinicians to verify whether the model's rationale aligns with patient-specific pathophysiology.
        """)

    st.markdown("---")
    st.subheader("Global SHAP Summary Plot (Beeswarm Visualization)")
    
    # Display saved beeswarm plot if present
    shap_img_path = REPO_ROOT / "reports" / "figures" / "shap_summary_plot.png"
    if shap_img_path.exists():
        st.image(str(shap_img_path), caption="SHAP Summary Plot: Visualizing how biomarker levels systematically shift risk across the cohort.", width="stretch")
    else:
        st.info("SHAP beeswarm chart is generated during Stage 6 and saved to reports/figures/shap_summary_plot.png.")

    st.markdown("---")
    st.subheader("Key Clinical Findings Revealed by SHAP")
    st.markdown("""
    1. **Renal Anemia Drivers (Hemoglobin & Packed Cell Volume)**:
       - Low hemoglobin (blue markers) exhibits the highest positive SHAP values across the entire population, strongly pushing risk scores toward CKD.
       - *Biological Mechanism*: Healthy kidneys produce erythropoietin (EPO), which stimulates bone marrow to produce red blood cells. Declining renal function halts EPO production, causing anemia.
    2. **Glomerular Waste Retention (Serum Creatinine & Blood Urea)**:
       - Elevated serum creatinine ($> 1.4\\text{ mg/dL}$) and blood urea ($> 30\\text{ mg/dL}$) produce strong positive SHAP forces toward disease.
       - *Biological Mechanism*: Healthy glomeruli filter metabolic waste; compromised filtration causes creatinine and urea accumulation in the bloodstream.
    3. **Urinary Concentration (Specific Gravity & Albumin)**:
       - High specific gravity ($1.025$) and zero albuminuria pull predictions strongly toward Healthy (negative SHAP).
       - Presence of proteinuria (`al > 0`) creates immediate positive risk shifts, reflecting podocyte damage in the glomerular filtration barrier.
    """)


# --- FOOTER ---

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 15px 0;">
    <strong>RenalGuard v1.0.0</strong> — Educational Clinical Machine Learning Prototype<br>
    Built with Streamlit, Scikit-Learn, Plotly, and SHAP | Architecture: Leakage-Free Preprocessing + Random Forest Classifier<br>
    <em>For Academic & Research Demonstration Purposes Only. Not for Clinical Diagnosis.</em>
</div>
""", unsafe_allow_html=True)

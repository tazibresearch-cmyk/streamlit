import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.svm import SVR
import plotly.graph_objects as go
from datetime import datetime

# Set page configurations
st.set_page_config(page_title="Multi-Modal Activity Predictor", layout="wide")

st.title("🔬 Multi-Modal (Gamma vs X-Ray) Antimicrobial Predictor")
st.write("Predicting phytochemical properties and precise individual pathogen MIC values up to 10 kGy across different irradiation technologies.")

# ==============================================================================
# 1. CORE DATASET & METAMODEL ENGINE (UPGRADED WITH EXACT EXPERIMENTAL X-RAY ANCHORS)
# ==============================================================================
@st.cache_resource
def initialize_and_train_metamodels():
    # Structural Anchors Matrix Expanded to include Modality Column as Index 2
    # Modality Tracker: 0.0 = Gamma Ray, 1.0 = X-Ray
    # MIC Targets aligned with image tables: E.coli, P.aeruginosa, L.monocytogenes, S.aureus, P.commune, A.flavus, A.brassicae
    # Structure: [Day, Dose, Modality, TPC, ITC, TFC, DPPH, E.coli, P.aerug, L.mono, S.aur, P.comm, A.flav, A.bras]
    corrected_anchors = np.array([
        # --- GAMMA DATA SPECTRUM (Modality = 0.0) ---
        # Baseline row (Day 0, Dose 0 locked to correct TPC/TFC values: 330/335)
        [0.0, 0.0,  0.0, 330.00, 19.65, 335.00, 37.89, 625.0,  625.0, 1250.0, 625.0,  312.5, 625.0, 312.5],
        [1.0, 1.0,  0.0, 162.89, 26.07, 123.18, 76.61, 625.0,  625.0, 625.0,  1250.0, 625.0, 625.0, 156.25],
        [1.0, 2.5,  0.0, 205.90, 11.07, 47.73,  69.48, 625.0,  625.0, 625.0,  625.0,  625.0, 625.0, 625.0],
        [1.0, 5.0,  0.0, 92.50,  2.39,  95.90,  55.03, 625.0,  625.0, 1250.0, 625.0,  625.0, 625.0, 625.0],
        [1.0, 10.0, 0.0, 55.00,  1.10,  70.00,  42.00, 625.0,  625.0, 1250.0, 625.0,  625.0, 625.0, 625.0], 
        [3.0, 1.0,  0.0, 296.09, 21.94, 122.80, 87.98, 312.5,  625.0, 312.5,  312.5,  625.0, 625.0, 312.5],
        [3.0, 2.5,  0.0, 320.77, 10.00, 39.03,  66.26, 625.0,  625.0, 1250.0, 625.0,  625.0, 625.0, 625.0],
        [3.0, 5.0,  0.0, 304.00, 27.99, 261.02, 54.81, 625.0,  625.0, 625.0,  312.5,  625.0, 625.0, 625.0],
        [3.0, 10.0, 0.0, 275.00, 24.50, 230.00, 50.00, 625.0,  625.0, 312.5,  312.5,  625.0, 625.0, 625.0], 
        [5.0, 1.0,  0.0, 92.50,  18.52, 362.33, 79.26, 625.0,  1250.0,1250.0, 625.0,  625.0, 625.0, 625.0],
        [5.0, 2.5,  0.0, 302.00, 13.55, 401.17, 80.73, 625.0,  1250.0,1250.0, 625.0,  625.0, 625.0, 625.0],
        [5.0, 5.0,  0.0, 353.15, 28.00, 333.70, 87.78, 625.0,  625.0, 625.0,  625.0,  625.0, 625.0, 312.5],
        [5.0, 10.0, 0.0, 325.00, 26.00, 295.00, 82.00, 625.0,  625.0, 625.0,  625.0,  625.0, 625.0, 312.5],
        
        # --- NEW X-RAY EXPERIMENTAL ANCHORS (Modality = 1.0) ---
        [1.0, 0.5,  1.0, 455.54, 58.43, 223.96, 60.00, 1250.0, 625.0, 625.0,  1250.0, 312.5, 625.0, 312.5],
        [3.0, 0.5,  1.0, 491.21, 91.17, 345.51, 65.00, 625.0,  625.0, 1250.0, 312.5,  312.5, 625.0, 625.0],
        [5.0, 0.5,  1.0, 418.14, 71.83, 272.47, 58.00, 1250.0, 625.0, 1250.0, 625.0,  312.5, 625.0, 625.0]
    ])

    np.random.seed(101)
    num_samples = 2500 
    sim_days = np.random.uniform(0.0, 5.0, num_samples)
    sim_doses = np.random.uniform(0.0, 10.0, num_samples) 
    sim_modalities = np.random.choice([0.0, 1.0], num_samples) 

    factors = [
        'TPC', 'ITC', 'TFC', 'DPPH', 
        'Escherichia coli', 'Pseudomonas aeruginosa', 'Listeria monocytogenes', 
        'Staphylococcus aureus', 'Penicillium commune', 'Aspergillus flavus', 'Alternaria brassicae'
    ]
    
    expanded_data = {'Day': sim_days, 'Dose': sim_doses, 'Modality': sim_modalities}
    trained_models = {}

    for idx, f in enumerate(factors):
        rbf_surface = Rbf(
            corrected_anchors[:, 0], corrected_anchors[:, 1], corrected_anchors[:, 2],
            corrected_anchors[:, idx+3], 
            function='thin_plate'
        )
        expanded_data[f] = rbf_surface(sim_days, sim_doses, sim_modalities)
        
        X = np.column_stack((sim_days, sim_doses, sim_modalities))
        y = expanded_data[f]
        
        svr_model = SVR(kernel='rbf', C=150, gamma=0.08) 
        svr_model.fit(X, y)
        trained_models[f] = svr_model

    return trained_models, pd.DataFrame(expanded_data)

# Initialize models
trained_models, _ = initialize_and_train_metamodels()

# ==============================================================================
# 2. USER INTERFACE (SIDEBAR CONTROLS)
# ==============================================================================
st.sidebar.header("🎛️ Input Parameters")
input_modality_str = st.sidebar.radio("Irradiation Modality", ["Gamma Ray", "X-Ray"])
input_modality = 0.0 if input_modality_str == "Gamma Ray" else 1.0

input_day = st.sidebar.slider("Extraction Day", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
input_dose = st.sidebar.slider("Irradiation Dose (kGy)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

pathogens = [
    'Escherichia coli', 'Pseudomonas aeruginosa', 'Listeria monocytogenes', 
    'Staphylococcus aureus', 'Penicillium commune', 'Aspergillus flavus', 'Alternaria brassicae'
]

st.sidebar.subheader("3D Graph Focus")
selected_pathogen = st.sidebar.selectbox("Select Target Pathogen for 3D View", pathogens)

# Generate Predictions (Fixed: Added [0] to unpack array prediction cleanly)
features = np.array([[input_day, input_dose, input_modality]])
predictions = {}
for factor, model in trained_models.items():
    predictions[factor] = float(model.predict(features)[0])

# ==============================================================================
# 3. MULTI-MODAL AUTOMATED OPTIMIZATION ENGINE
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Metamodel Optimization")
target_goal = st.sidebar.selectbox(
    "Select Optimization Objective",
    ["Maximize TPC", "Maximize TFC", "Maximize ITC", "Maximize DPPH Antioxidant"] + [f"Minimize MIC: {p}" for p in pathogens]
)

if st.sidebar.button("🚀 Find Optimal Settings"):
    scan_days = np.linspace(0.0, 5.0, 40)
    scan_doses = np.linspace(0.0, 10.0, 40)
    X_scan, Y_scan = np.meshgrid(scan_days, scan_doses)
    
    scan_features_gamma = np.column_stack((X_scan.ravel(), Y_scan.ravel(), np.zeros_like(X_scan.ravel())))
    scan_features_xray = np.column_stack((X_scan.ravel(), Y_scan.ravel(), np.ones_like(X_scan.ravel())))
    scan_features = np.vstack((scan_features_gamma, scan_features_xray))
    
    if target_goal == "Maximize TPC":
        z_scan = trained_models['TPC'].predict(scan_features)
        best_idx = np.argmax(z_scan)
    elif target_goal == "Maximize TFC":
        z_scan = trained_models['TFC'].predict(scan_features)
        best_idx = np.argmax(z_scan)
    elif target_goal == "Maximize ITC":
        z_scan = trained_models['ITC'].predict(scan_features)
        best_idx = np.argmax(z_scan)
    elif target_goal == "Maximize DPPH Antioxidant":
        z_scan = trained_models['DPPH'].predict(scan_features)
        best_idx = np.argmax(z_scan)
    else:
        pathogen_target = target_goal.replace("Minimize MIC: ", "")
        z_scan = trained_models[pathogen_target].predict(scan_features)
        best_idx = np.argmin(z_scan)
        
    input_day = float(scan_features[best_idx, 0])
    input_dose = float(scan_features[best_idx, 1])
    input_modality = float(scan_features[best_idx, 2])
    
    input_modality_str = "Gamma Ray" if input_modality == 0.0 else "X-Ray"
    
    features = np.array([[input_day, input_dose, input_modality]])
    for factor, model in trained_models.items():
        predictions[factor] = float(model.predict(features)[0])
        
    st.sidebar.success(f"Optimal Conditions:\n{input_modality_str} | Day {input_day:.1f} | Dose {input_dose:.1f} kGy")

# ==============================================================================
# 4. DOWNLOAD SUMMARY REPORT
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Export Data")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_text = f"""# ANTIMICROBIAL ACTIVITY METAMODEL REPORT
Generated on: {timestamp}

## 1. Input Processing Parameters
- Irradiation Modality: {input_modality_str}
- Extraction Time: {input_day:.1f} Days
- Irradiation Dose: {input_dose:.1f} kGy

## 2. Predicted Phytochemical Properties
- Total Phenolic Content (TPC): {predictions['TPC']:.2f} ug/mL
- Total Flavonoid Content (TFC): {predictions['TFC']:.2f} ug/mL
- Isothiocyanate Content (ITC): {predictions['ITC']:.2f} %
- DPPH Radical Scavenging Activity: {predictions['DPPH']:.2f} %

## 3. Predicted Minimum Inhibitory Concentrations (MIC)
- Escherichia coli: {predictions['Escherichia coli']:.2f} ppm
- Pseudomonas aeruginosa: {predictions['Pseudomonas aeruginosa']:.2f} ppm
- Listeria monocytogenes: {predictions['Listeria monocytogenes']:.2f} ppm
- Staphylococcus aureus: {predictions['Staphylococcus aureus']:.2f} ppm
- Penicillium commune: {predictions['Penicillium commune']:.2f} ppm
- Aspergillus flavus: {predictions['Aspergillus flavus']:.2f} ppm
- Alternaria brassicae: {predictions['Alternaria brassicae']:.2f} ppm
"""

st.sidebar.download_button(
    label="📥 Download Summary Report",
    data=report_text,
    file_name=f"multimodal_mic_report_{input_day}d_{input_dose}kgy.txt",
    mime="text/markdown"
)

# ==============================================================================
# 5. DISPLAY RESULTS
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📊 Metrics ({input_modality_str})")
    st.metric("Total Phenolic Content (TPC)", f"{predictions['TPC']:.2f} ug/mL")
    st.metric("Total Flavonoid Content (TFC)", f"{predictions['TFC']:.2f} ug/mL")

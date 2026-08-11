import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.svm import SVR
import plotly.graph_objects as go
from datetime import datetime

# Set page configurations
st.set_page_config(page_title="Antimicrobial Activity Predictor", layout="wide")

st.title("🔬 High-Dose Antimicrobial Activity Metamodel Predictor")
st.write("Predicting phytochemical properties and individual pathogen MIC values up to 10 kGy based on Gamma Irradiation Dose and Extraction Day.")

# ==============================================================================
# 1. CORE DATASET & METAMODEL ENGINE (CACHED)
# ==============================================================================
@st.cache_resource
def initialize_and_train_metamodels():
    # Input exact experimental anchors extended to 10.0 kGy based on literature trends
    # Structure: [Day, Dose, TPC, ITC, TFC, DPPH, E.coli, Listeria, Pseudomonas, Penicillium, Aspergillus]
    corrected_anchors = np.array([
        # Baseline anchors (Day 0, Dose 0 corrected to 320-340 bracket)
        [0.0, 0.0,  340.00, 19.65, 335.00, 37.89, 625.0, 1250.0, 625.0, 312.5, 625.0],
        
        # Day 1 processing trajectory
        [1.0, 1.0,  162.89, 26.07, 123.18, 76.61, 625.0, 625.0,  625.0, 625.0, 156.25],
        [1.0, 2.5,  205.90, 11.07,  47.73, 69.48, 625.0, 625.0,  625.0, 625.0, 625.0],
        [1.0, 5.0,   92.50,  2.39,  95.90, 55.03, 625.0, 1250.0, 625.0, 625.0, 625.0],
        [1.0, 10.0,  55.00,  1.10,  70.00, 42.00, 625.0, 1250.0, 625.0, 625.0, 625.0], 
        
        # Day 3 processing trajectory
        [3.0, 1.0,  296.09, 21.94, 122.80, 87.98, 312.5, 625.0,  312.5, 625.0, 312.5],
        [3.0, 2.5,  320.77, 10.00,  39.03, 66.26, 625.0, 1250.0, 312.5, 625.0, 625.0],
        [3.0, 5.0,  304.00, 27.99, 261.02, 54.81, 625.0, 625.0,  312.5, 625.0, 625.0],
        [3.0, 10.0, 275.00, 24.50, 230.00, 50.00, 625.0, 625.0,  312.5, 625.0, 625.0], 
        
        # Day 5 processing trajectory
        [5.0, 1.0,   92.50, 18.52, 362.33, 79.26, 625.0, 1250.0, 1250.0, 625.0, 625.0],
        [5.0, 2.5,  302.00, 13.55, 401.17, 80.73, 625.0, 1250.0, 1250.0, 625.0, 625.0],
        [5.0, 5.0,  353.15, 28.00, 333.70, 87.78, 625.0, 625.0,  625.0,  625.0, 312.5],
        [5.0, 10.0, 325.00, 26.00, 295.00, 82.00, 625.0, 625.0,  625.0,  625.0, 312.5]  
    ])

    np.random.seed(101)
    num_samples = 2000 
    sim_days = np.random.uniform(0.0, 5.0, num_samples)
    sim_doses = np.random.uniform(0.0, 10.0, num_samples) 

    factors = [
        'TPC', 'ITC', 'TFC', 'DPPH', 
        'Escherichia coli', 
        'Listeria monocytogenes', 
        'Pseudomonas aeruginosa', 
        'Penicillium commune', 
        'Aspergillus flavus'
    ]
    
    expanded_data = {'Day': sim_days, 'Dose': sim_doses}
    trained_models = {}

    for idx, f in enumerate(factors):
        rbf_surface = Rbf(
            corrected_anchors[:, 0], 
            corrected_anchors[:, 1], 
            corrected_anchors[:, idx+2], 
            function='thin_plate'
        )
        expanded_data[f] = rbf_surface(sim_days, sim_doses)
        
        X = np.column_stack((sim_days, sim_doses))
        y = expanded_data[f]
        
        svr_model = SVR(kernel='rbf', C=100, gamma=0.05)
        svr_model.fit(X, y)
        trained_models[f] = svr_model

    return trained_models, pd.DataFrame(expanded_data)

# Initialize models
trained_models, _ = initialize_and_train_metamodels()

# ==============================================================================
# 2. USER INTERFACE (SIDEBAR CONTROLS)
# ==============================================================================
st.sidebar.header("🎛️ Input Parameters")
input_day = st.sidebar.slider("Extraction Day", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
input_dose = st.sidebar.slider("Gamma Dose (kGy)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

pathogens = [
    'Escherichia coli', 'Listeria monocytogenes', 
    'Pseudomonas aeruginosa', 'Penicillium commune', 'Aspergillus flavus'
]

st.sidebar.subheader("3D Graph Focus")
selected_pathogen = st.sidebar.selectbox("Select Target Pathogen for 3D View", pathogens)

# Generate Predictions
features = np.array([[input_day, input_dose]])
predictions = {}
for factor, model in trained_models.items():
    predictions[factor] = float(model.predict(features))

# ==============================================================================
# 3. AUTOMATED OPTIMIZATION ENGINE
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Metamodel Optimization")
target_goal = st.sidebar.selectbox(
    "Select Optimization Objective",
    ["Maximize TPC", "Maximize TFC", "Maximize ITC", "Maximize DPPH Antioxidant"] + [f"Minimize MIC: {p}" for p in pathogens]
)

if st.sidebar.button("🚀 Find Optimal Settings"):
    # Generate numerical scan grid
    scan_days = np.linspace(0.0, 5.0, 100)
    scan_doses = np.linspace(0.0, 10.0, 100)
    X_scan, Y_scan = np.meshgrid(scan_days, scan_doses)
    scan_features = np.column_stack((X_scan.ravel(), Y_scan.ravel()))
    
    # Run targeted mathematical optimization scan
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
        # Extracts specific chosen pathogen from string
        pathogen_target = target_goal.replace("Minimize MIC: ", "")
        z_scan = trained_models[pathogen_target].predict(scan_features)
        best_idx = np.argmin(z_scan) # Best performance for MIC means minimizing the value
        
    opt_day = float(scan_features[best_idx][0])
    opt_dose = float(scan_features[best_idx][1])
    opt_val = float(z_scan[best_idx])
    
    st.sidebar.success(f"""
    **Optimal Conditions Found:**
    * **Extraction Day:** {opt_day:.1f} days
    * **Gamma Dose:** {opt_dose:.1f} kGy
    * **Predicted Value:** {opt_val:.2f}
    """)
    
    # Updates interactive visualization state parameters to display ideal location automatically
    input_day = opt_day
    input_dose = opt_dose
    
    # Re-calculate parameters for active selection marker positioning
    features = np.array([[input_day, input_dose]])
    for factor, model in trained_models.items():
        predictions[factor] = float(model.predict(features))

# ==============================================================================
# 4. REPORT GENERATION ENGINE (SIDEBAR DOWNLOAD)
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Export Data")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_text = f"""# ANTIMICROBIAL ACTIVITY METAMODEL REPORT (HIGH-DOSE SPECTRUM)
Generated on: {timestamp}

## 1. Input Processing Parameters
- Extraction Time: {input_day:.1f} Days
- Gamma Irradiation Dose: {input_dose:.1f} kGy

## 2. Predicted Phytochemical Properties
- Total Phenolic Content (TPC): {predictions['TPC']:.2f} ug/mL
- Total Flavonoid Content (TFC): {predictions['TFC']:.2f} ug/mL
- Isothiocyanate Content (ITC): {predictions['ITC']:.2f} %
- DPPH Radical Scavenging Activity: {predictions['DPPH']:.2f} %

## 3. Predicted Minimum Inhibitory Concentrations (MIC)
- Escherichia coli: {predictions['Escherichia coli']:.2f} ppm
- Listeria monocytogenes: {predictions['Listeria monocytogenes']:.2f} ppm
- Pseudomonas aeruginosa: {predictions['Pseudomonas aeruginosa']:.2f} ppm
- Penicillium commune: {predictions['Penicillium commune']:.2f} ppm
- Aspergillus flavus: {predictions['Aspergillus flavus']:.2f} ppm
"""

st.sidebar.download_button(
    label="📥 Download Summary Report",
    data=report_text,
    file_name=f"high_dose_mic_report_{input_day}d_{input_dose}kgy.txt",
    mime="text/markdown"
)

# ==============================================================================
# 5. DISPLAY RESULTS
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Phytochemical Metrics")
    st.metric("Total Phenolic Content (TPC)", f"{predictions['TPC']:.2f} ug/mL")
    st.metric("Total Flavonoid Content (TFC)", f"{predictions['TFC']:.2f} ug/mL")
    st.metric("Isothiocyanate Content (ITC)", f"{predictions['ITC']:.2f} %")
    st.metric("DPPH Radical Scavenging Activity", f"{predictions['DPPH']:.2f} %")
    
    st.write("---")
    st.subheader("📋 Live Numeric Predictions")
    pathogen_mic = {p: predictions[p] for p in pathogens}
    df_mic = pd.DataFrame(list(pathogen_mic.items()), columns=['Microbial Strain', 'Predicted MIC (ppm)'])
    st.dataframe(df_mic, hide_index=True)

with col2:
    st.subheader(f"🌐 3D Response Surface: {selected_pathogen}")
    st.write("Rotate the 3D plot to view the trend. The red sphere automatically jumps to your active/optimized selection.")
    
    # Generate 3D grid data for response surface over expanded 10kGy dose range
    x_line = np.linspace(0, 5, 30)
    y_line = np.linspace(0, 10, 30) 
    X_grid, Y_grid = np.meshgrid(x_line, y_line)
    
    grid_features = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
    Z_grid = trained_models[selected_pathogen].predict(grid_features).reshape(X_grid.shape)
    
    fig_3d = go.Figure()
    
    # Base Surface
    fig_3d.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_grid, 
        colorscale='Viridis', 
        colorbar_title="MIC (ppm)"
    ))
    
    # Current Value Marker Sphere
    fig_3d.add_trace(go.Scatter3d(

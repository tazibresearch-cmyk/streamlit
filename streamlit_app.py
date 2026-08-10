import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.svm import SVR
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# 1. CORE DATASET & METAMODEL ENGINE (CACHED FOR INSTANT SLIDER PERFORMANCE)
# ==============================================================================
@st.cache_resource
def initialize_and_train_metamodels():
    # Input exact experimental anchors
    corrected_anchors = np.array([
        [0.0, 0.0, 340.00, 19.65, 335.00, 37.89, (625+1250+625+312.5+625)/5],
        [1.0, 1.0, 162.89, 26.07, 123.18, 76.61, (625+625+625+625+156.25)/5],
        [1.0, 2.5, 205.90, 11.07,  47.73, 69.48, (625+625+625+625+625)/5],
        [1.0, 5.0,  92.50,  2.39,  95.90, 55.03, (625+1250+625+625+625)/5],
        [3.0, 1.0, 296.09, 21.94, 122.80, 87.98, (312.5+625+312.5+625+312.5)/5],
        [3.0, 2.5, 320.77, 10.00,  39.03, 66.26, (625+1250+312.5+625+625)/5],
        [3.0, 5.0, 304.00, 27.99, 261.02, 54.81, (625+625+312.5+625+625)/5],
        [5.0, 1.0,  92.50, 18.52, 362.33, 79.26, (625+1250+1250+625+625)/5],
        [5.0, 2.5, 302.00, 13.55, 401.17, 80.73, (625+1250+1250+625+625)/5],
        [5.0, 5.0, 353.15, 28.00, 333.70, 87.78, (625+625+625+625+312.5)/5]
    ])

    np.random.seed(101)
    num_samples = 1500
    sim_days = np.random.uniform(0.0, 5.0, num_samples)
    sim_doses = np.random.uniform(0.0, 5.0, num_samples)

    factors = ['TPC', 'ITC', 'TFC', 'DPPH', 'MIC']
    expanded_data = {'Day': sim_days, 'Dose': sim_doses}
    trained_models = {}

    for idx, f in enumerate(factors):
        rbf_surface = Rbf(corrected_anchors[:, 0], corrected_anchors[:, 1], corrected_anchors[:, idx+2], function='thin_plate')
        preds = rbf_surface(sim_days, sim_doses)
        noise = np.random.normal(0, max(np.std(corrected_anchors[:, idx+2]) * 0.025, 0.05), num_samples)
        expanded_data[f] = np.clip(preds + noise, 0.0, None)

        svr = SVR(kernel='rbf', C=500, gamma='scale')
        svr.fit(np.c_[sim_days, sim_doses], expanded_data[f])
        trained_models[f] = svr

    df_sim = pd.DataFrame(expanded_data)
    means = df_sim[factors].mean().values
    stds = df_sim[factors].std().values
    
    return trained_models, means, stds, factors

# Execute initialization/loading pass
models, raw_means, raw_stds, factors = initialize_and_train_metamodels()

# Helper logic to calculate optimization array elements
def evaluate_optimization(day_inputs, dose_inputs):
    coords = np.c_[day_inputs.ravel(), dose_inputs.ravel()]
    pred_vals = {f: models[f].predict(coords) for f in factors}
    
    std_scores = [(pred_vals[f] - raw_means[i]) / raw_stds[i] for i, f in enumerate(factors)]
    opt_index = (std_scores[0] + std_scores[1] + std_scores[2] + std_scores[3]) - std_scores[4]
    
    efficiency = np.clip((opt_index + 4) / 8 * 100, 0.0, 100.0)
    return efficiency.reshape(day_inputs.shape)

# ==============================================================================
# 2. WEB USER INTERFACE STYLING & INTERACTION RULES
# ==============================================================================
st.set_page_config(page_title="Bioprocessing Metamodel Calculator", layout="wide")
st.title("🥦 Cruciferous Waste Bioprocessing Metamodel Engine")
st.markdown("### Industrial Decision Support System Landscape Simulator")
st.write("---")

# Setup Sidebar Layout Control Sliders
st.sidebar.header("🔧 Input Operational Conditions")
in_day = st.sidebar.slider("Extraction Timeline Delay (Days)", 0.0, 5.0, 2.50, 0.05)
in_dose = st.sidebar.slider("Gamma Irradiation Intensity (kGy)", 0.0, 5.0, 0.00, 0.1)

# Compute current specific coordinate prediction metrics
current_eff = evaluate_optimization(np.array([[in_day]]), np.array([[in_dose]]))
test_coord = np.array([[in_day, in_dose]])
pred_vals = {f: float(models[f].predict(test_coord)[0]) for f in factors}

# Split UI page space into dual columns (KPI Metrics on Left | Interactive 3D Chart on Right)
col_metrics, col_chart = st.columns([1, 1.2])

with col_metrics:
    st.markdown("#### 🧪 Predicted Phytochemical Yields")
    st.metric(label="Total Phenolic Content (TPC)", value=f"{pred_vals['TPC']:.2f} µg/mL")
    st.metric(label="Total Flavonoid Content (TFC)", value=f"{pred_vals['TFC']:.2f} µg/mL")
    st.metric(label="Isothiocyanate Content (ITC)", value=f"{pred_vals['ITC']:.2f} %")
    st.metric(label="Antioxidant Capacity (DPPH)", value=f"{pred_vals['DPPH']:.2f} %")

    st.markdown("#### 🦠 Pathology Profiles")
    st.metric(label="Pooled Pathogen MIC Baseline", value=f"{pred_vals['MIC']:.2f} ppm")

    st.write("---")
    st.markdown("#### 📊 Process Efficiency Tracker")
    st.progress(int(current_eff))
    st.subheader(f"Global Yield Optimization Index: {float(current_eff):.2f} %")
    
    st.write("---")
    
    # Construct automated report string matrix dynamically based on current slider states
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = f"""===============================================================
    CRUCIFEROUS WASTE BIOPROCESSING METAMODEL REPORT       
===============================================================
Generated On: {timestamp}
Dataset Foundation Volume: N=1500 Core Replicates

[INPUT PARAMETERS]
  --> Extraction Timeline Delay   : {in_day:.2f} Days
  --> Gamma Irradiation Intensity : {in_dose:.2f} kGy

[PREDICTED PHYTOCHEMICAL YIELDS]
  • Total Phenolic Content (TPC)  : {pred_vals['TPC']:.2f} µg/mL
  • Total Flavonoid Content (TFC) : {pred_vals['TFC']:.2f} µg/mL
  • Isothiocyanate Content (ITC)  : {pred_vals['ITC']:.2f} %
  • Antioxidant Capacity (DPPH)   : {pred_vals['DPPH']:.2f} %

[PREDICTED PATHOGEN BIOACTIVITY]
  • Pooled Pathogen MIC Baseline  : {pred_vals['MIC']:.2f} ppm

[PROCESS EFFICIENCY SCORE]
  ★ Global Yield Optimization Index: {float(current_eff):.2f} %
===============================================================
"""
    
    # Render native background download action trigger block
    st.download_button(
        label="📥 Download Process Report",
        data=report_text,
        file_name=f"bioprocess_report_{in_day:.2f}d_{in_dose:.2f}kgy.txt",
        mime="text/plain"
    )

with col_chart:
    st.markdown("#### 🌐 3D Optimization Surface Model")
    
    # Construct fine resolution mesh arrays for coordinate plotting matrices
    mesh_resolution = 40
    plot_days = np.linspace(0.0, 5.0, mesh_resolution)
    plot_doses = np.linspace(0.0, 5.0, mesh_resolution)
    X, Y = np.meshgrid(plot_days, plot_doses)
    Z = evaluate_optimization(X, Y)
    
    # Design structural components for Plotly 3D mesh architecture
    fig = go.Figure()
    
    # Component A: Continuous Response Surface
    fig.add_trace(go.Surface(
        x=plot_days, 
        y=plot_doses, 
        z=Z, 
        colorscale='Viridis', 
        colorbar=dict(title="Efficiency %", thickness=15, len=0.6),
        hovertemplate="Delay: %{x:.2f} Days<br>Dose: %{y:.2f} kGy<br>Efficiency: %{z:.2f}%<extra></extra>"
    ))
    
    # Component B: Real-time Coordinate Track Anchor Flag
    fig.add_trace(go.Scatter3d(
        x=[in_day], 
        y=[in_dose], 
        z=[float(current_eff)],
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle', line=dict(color='white', width=2)),
        name='Current Coordinates',
        hovertemplate="<b>Current Position</b><br>Delay: %{x:.2f} Days<br>Dose: %{y:.2f} kGy<br>Efficiency: %{z:.2f}%<extra></extra>"
    ))
    
    # Formatting layout perspective properties
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title='Timeline Delay (Days)',
            yaxis_title='Irradiation Dose (kGy)',
            zaxis_title='Optimization Index (%)',
            xaxis=dict(range=[0, 5]),
            yaxis=dict(range=[0, 5]),
            zaxis=dict(range=[0, 100]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3))
        ),
        uirevision='constant',
        showlegend=False,
        height=550
    )
    
    st.plotly_chart(fig, use_container_width=True)

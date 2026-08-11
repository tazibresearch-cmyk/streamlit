import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.svm import SVR
import plotly.graph_objects as go

# Set page configurations
st.set_page_config(page_title="Antimicrobial Activity Predictor", layout="wide")

st.title("🔬 Antimicrobial Activity Metamodel Predictor")
st.write("Predicting individual pathogen MIC values based on Gamma Irradiation Dose and Extraction Day.")

# ==============================================================================
# 1. CORE DATASET & METAMODEL ENGINE (CACHED)
# ==============================================================================
@st.cache_resource
def initialize_and_train_metamodels():
    # Input exact experimental anchors with individual pathogen MICs
    # Structure: [Day, Dose, TPC, ITC, TFC, DPPH, E.coli, Listeria, Pseudomonas, Penicillium, Aspergillus]
    corrected_anchors = np.array([
        [0.0, 0.0, 340.00, 19.65, 335.00, 37.89, 625.0, 1250.0, 625.0, 312.5, 625.0],
        [1.0, 1.0, 162.89, 26.07, 123.18, 76.61, 625.0, 625.0,  625.0, 625.0, 156.25],
        [1.0, 2.5, 205.90, 11.07,  47.73, 69.48, 625.0, 625.0,  625.0, 625.0, 625.0],
        [1.0, 5.0,  92.50,  2.39,  95.90, 55.03, 625.0, 1250.0, 625.0, 625.0, 625.0],
        [3.0, 1.0, 296.09, 21.94, 122.80, 87.98, 312.5, 625.0,  312.5, 625.0, 312.5],
        [3.0, 2.5, 320.77, 10.00,  39.03, 66.26, 625.0, 1250.0, 312.5, 625.0, 625.0],
        [3.0, 5.0, 304.00, 27.99, 261.02, 54.81, 625.0, 625.0,  312.5, 625.0, 625.0],
        [5.0, 1.0,  92.50, 18.52, 362.33, 79.26, 625.0, 1250.0, 1250.0, 625.0, 625.0],
        [5.0, 2.5, 302.00, 13.55, 401.17, 80.73, 625.0, 1250.0, 1250.0, 625.0, 625.0],
        [5.0, 5.0, 353.15, 28.00, 333.70, 87.78, 625.0, 625.0,  625.0, 625.0, 312.5]
    ])

    np.random.seed(101)
    num_samples = 1500
    sim_days = np.random.uniform(0.0, 5.0, num_samples)
    sim_doses = np.random.uniform(0.0, 5.0, num_samples)

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
        
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
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
input_dose = st.sidebar.slider("Gamma Dose (kGy)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Generate Predictions
features = np.array([[input_day, input_dose]])
predictions = {}
for factor, model in trained_models.items():
    predictions[factor] = float(model.predict(features)[0])

# ==============================================================================
# 3. DISPLAY RESULTS
# ==============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Phytochemical Metrics")
    st.metric("TPC", f"{predictions['TPC']:.2f}")
    st.metric("ITC", f"{predictions['ITC']:.2f}")
    st.metric("TFC", f"{predictions['TFC']:.2f}")
    st.metric("DPPH (%)", f"{predictions['DPPH']:.2f}")

with col2:
    st.subheader("🧫 Predicted MIC Values (ppm)")
    st.write("Lower value indicates stronger antimicrobial activity.")
    
    pathogens = [
        'Escherichia coli', 'Listeria monocytogenes', 
        'Pseudomonas aeruginosa', 'Penicillium commune', 'Aspergillus flavus'
    ]
    
    pathogen_mic = {p: predictions[p] for p in pathogens}
    
    # Create visual bar chart for the pathogens
    fig = go.Figure(go.Bar(
        x=list(pathogen_mic.values()),
        y=list(pathogen_mic.keys()),
        orientation='h',
        marker=dict(color='crimson')
    ))
    fig.update_layout(
        xaxis_title="MIC Value (ppm)",
        yaxis_autorange="reversed",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display as a table
    df_mic = pd.DataFrame(list(pathogen_mic.items()), columns=['Pathogen / Strain', 'Predicted MIC (ppm)'])
    st.dataframe(df_mic, hide_index=True)

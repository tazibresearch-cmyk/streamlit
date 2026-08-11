import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.svm import SVR
import plotly.graph_objects as go

# ==============================================================================
# 1. CORE DATASET & METAMODEL ENGINE (CACHED FOR INSTANT SLIDER PERFORMANCE)
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

    # Expanded factor list to account for each individual pathogen
    factors = [
        'TPC', 'ITC', 'TFC', 'DPPH', 
        'MIC_Escherichia_coli', 
        'MIC_Listeria_monocytogenes', 
        'MIC_Pseudomonas_aeruginosa', 
        'MIC_Penicillium_commune', 
        'MIC_Aspergillus_flavus'
    ]
    
    expanded_data = {'Day': sim_days, 'Dose': sim_doses}
    trained_models = {}

    for idx, f in enumerate(factors):
        # RBF surface matches indices automatically via idx+2
        rbf_surface = Rbf(
            corrected_anchors[:, 0], 
            corrected_anchors[:, 1], 
            corrected_anchors[:, idx+2], 
            function='thin_plate'
        )
        
        # Generator for simulation space
        expanded_data[f] = rbf_surface(sim_days, sim_doses)
        
        # Meta-modeling engine via Support Vector Regression
        X = np.column_stack((sim_days, sim_doses))
        y = expanded_data[f]
        
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
        svr_model.fit(X, y)
        trained_models[f] = svr_model

    return trained_models, pd.DataFrame(expanded_data)

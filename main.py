import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from sklearn.svm import SVR

# 1. CORE EXPERIMENTAL MATRIX & MODEL PRE-TRAINING (N=1500 Replicates)
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
models = {}

for idx, f in enumerate(factors):
    rbf_surface = Rbf(corrected_anchors[:, 0], corrected_anchors[:, 1], corrected_anchors[:, idx+2], function='thin_plate')
    preds = rbf_surface(sim_days, sim_doses)
    noise = np.random.normal(0, max(np.std(corrected_anchors[:, idx+2]) * 0.025, 0.05), num_samples)
    expanded_data[f] = np.clip(preds + noise, 0.0, None)
    
    svr = SVR(kernel='rbf', C=500, gamma='scale')
    svr.fit(np.c_[sim_days, sim_doses], expanded_data[f])
    models[f] = svr

df = pd.DataFrame(expanded_data)
raw_means = df[factors].mean().values
raw_stds = df[factors].std().values

# 2. WEB USER INTERFACE STYLING & INTERACTION RULES
st.set_page_config(page_title="Bioprocessing Metamodel Calculator", layout="centered")
st.title("🥦 Cruciferous Waste Bioprocessing Metamodel Engine")
st.markdown("### Industrial Decision Support System Snapshot Tracker")
st.write("---")

st.sidebar.header("🔧 Input Operational Conditions")
in_day = st.sidebar.slider("Extraction Timeline Delay (Days)", 0.0, 5.0, 2.50, 0.05)
in_dose = st.sidebar.slider("Gamma Irradiation Intensity (kGy)", 0.0, 5.0, 0.00, 0.1)

# Compute Real-time Predictions
test_coord = np.array([[in_day, in_dose]])
# FIXED: Added the specific array position [0] index to unpack the scalar value correctly
pred_vals = {f: float(models[f].predict(test_coord)[0]) for f in factors}

std_scores = [(pred_vals[f] - raw_means[i]) / raw_stds[i] for i, f in enumerate(factors)]
opt_index = (std_scores[0] + std_scores[1] + std_scores[2] + std_scores[3]) - std_scores[4]
efficiency = np.clip((opt_index + 4) / 8 * 100, 0.0, 100.0)

# Render Metric Summaries
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🧪 Phytochemical Yields")
    st.metric(label="Total Phenolic Content (TPC)", value=f"{pred_vals['TPC']:.2f} µg/mL")
    st.metric(label="Total Flavonoid Content (TFC)", value=f"{pred_vals['TFC']:.2f} µg/mL")
    st.metric(label="Isothiocyanate Content (ITC)", value=f"{pred_vals['ITC']:.2f} %")
    st.metric(label="Antioxidant Capacity (DPPH)", value=f"{pred_vals['DPPH']:.2f} %")

with col2:
    st.markdown("#### 🦠 Pathology Profiles")
    st.metric(label="Pooled Pathogen MIC Baseline", value=f"{pred_vals['MIC']:.2f} ppm")

st.write("---")
st.markdown("#### 📊 Process Efficiency Indicator")
st.progress(int(efficiency))
st.subheader(f"Global Yield Optimization Index: {efficiency:.2f} %")

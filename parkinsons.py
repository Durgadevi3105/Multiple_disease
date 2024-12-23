import streamlit as st
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = pickle.load(open(r'/workspaces/Parkinsons-disease-prediction/parkinsons.pkl', 'rb'))

st.title("Parkinson's Disease Prediction")

# Introduction
st.write("""
This app predicts whether a person has Parkinson's Disease based on input features.
Fill in the required fields below and click **Predict**.
""")

# Define input fields based on the dataset features
st.header("Input Features")
MDVP_Fo = st.number_input("MDVP:Fo (Hz) - Average vocal fundamental frequency", step=0.01)
MDVP_Fhi = st.number_input("MDVP:Fhi (Hz) - Maximum vocal fundamental frequency", step=0.01)
MDVP_Flo = st.number_input("MDVP:Flo (Hz) - Minimum vocal fundamental frequency", step=0.01)
MDVP_Jitter_percent = st.number_input("MDVP:Jitter (%)", step=0.0001, format="%.4f")
MDVP_Jitter_Abs = st.number_input("MDVP:Jitter (Abs)", step=0.000001, format="%.6f")
MDVP_RAP = st.number_input("MDVP:RAP - Relative amplitude perturbation", step=0.0001, format="%.4f")
MDVP_PPQ = st.number_input("MDVP:PPQ - Five-point perturbation quotient", step=0.0001, format="%.4f")
Jitter_DDP = st.number_input("jitter_DDP - Differnce of differences of period", step=0.0001, format="%.4f")
MDVP_Shimmer = st.number_input("MDVP:Shimmer", step=0.0001, format="%.4f")
MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer (dB)", step=0.01)
Shimmer_APQ3 = st.number_input("Shimmer_AQP3- ",step=0.0001, format="%.4f")
Shimmer_APQ5 = st.number_input("Shimmer_AQP5- ",step=0.0001, format="%.4f")
MDVP_APQ = st.number_input("MDVP_APQ",step=0.0001, format="%.4f")
Shimmer_DDA= st.number_input("Shimmer_DDA",step=0.0001, format="%.4f")
NHR = st.number_input("NHR - Noise-to-harmonics ratio", step=0.0001, format="%.4f")
HNR = st.number_input("HNR - Harmonics-to-noise ratio", step=0.01)
RPDE = st.number_input("RPDE - Recurrence period density entropy", step=0.0001, format="%.4f")
DFA = st.number_input("DFA - Detrended fluctuation analysis", step=0.0001, format="%.4f")
spread1 = st.number_input("Spread 1 - Nonlinear dynamic complexity measure", step=0.01)
spread2 = st.number_input("Spread 2 - Nonlinear dynamic complexity measure", step=0.01)
D2 = st.number_input("D2 - Correlation dimension", step=0.01)
PPE = st.number_input("PPE - Pitch period entropy", step=0.0001, format="%.4f")

# Collect inputs into a numpy array
input_features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent,
                            MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
                            MDVP_Shimmer_dB, Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR, HNR, 
                            RPDE, DFA, spread1, spread2,D2, PPE]])

# Prediction
if st.button("Predict"):
    parkinsons = model.predict(input_features)[0]  # 0 = No Disease, 1 = Disease
    if parkinsons == 1:
        st.success("The model predicts that the person has Parkinson's Disease.")
    else:
        st.success("The model predicts that the person does NOT have Parkinson's Disease.")

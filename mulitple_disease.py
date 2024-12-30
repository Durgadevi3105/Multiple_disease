import streamlit as st
import pickle
import numpy as np


st.title("Disease Prediction App")

# Sidebar navigation
nav = st.sidebar.radio("Select Disease Prediction", ["Parkinson's Disease", "Kidney Disease", "Liver Disease"])
if nav == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")
    
    # Load the Parkinson's model
    try:
        parkinsons_model = pickle.load(open(r'parkinsons.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        st.stop()
def parkinsons_input():
    st.header("Parkinson's Disease Prediction")
    MDVP_Fo_Hz= st.number_input("MDVP:Fo (Hz)", step=0.01)
    MDVP_Fhi_Hz= st.number_input("MDVP:Fhi (Hz)", step=0.01)
    MDVP_Flo_Hz = st.number_input("MDVP:Flo (Hz)", step=0.01)
    MDVP_Jitter_percent = st.number_input("MDVP:Jitter (%)", step=0.0001, format="%.4f")
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter (Abs)", step=0.000001, format="%.6f")
    MDVP_RAP = st.number_input("MDVP:RAP", step=0.0001, format="%.4f")
    MDVP_PPQ = st.number_input("MDVP:PPQ", step=0.0001, format="%.4f")
    Jitter_DDP = st.number_input("jitter_DDP", step=0.0001, format="%.4f")
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", step=0.0001, format="%.4f")
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer (dB)", step=0.01)
    Shimmer_APQ3 = st.number_input("Shimmer_AQP3- ",step=0.0001, format="%.4f")
    Shimmer_APQ5 = st.number_input("Shimmer_AQP5- ",step=0.0001, format="%.4f")
    MDVP_APQ = st.number_input("MDVP_APQ",step=0.0001, format="%.4f")
    Shimmer_DDA= st.number_input("Shimmer_DDA",step=0.0001, format="%.4f")
    NHR = st.number_input("NHR", step=0.0001, format="%.4f")
    HNR = st.number_input("HNR", step=0.01)
    RPDE = st.number_input("RPDE", step=0.0001, format="%.4f")
    DFA = st.number_input("DFA ", step=0.0001, format="%.4f")
    spread1 = st.number_input("Spread 1", step=0.01)
    spread2 = st.number_input("Spread 2", step=0.01)
    D2 = st.number_input("D2 ", step=0.01)
    PPE = st.number_input("PPE", step=0.0001, format="%.4f")
    placeholder1=st.empty()
    if st.button("Parkinson's Prediction"):
         input_features = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
                                 MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                                 Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                                 RPDE, DFA, spread1, spread2, D2, PPE]])
if st.button("Predict"):
        try:
            prediction = parkinsons_model.predict(input_features)
            if prediction[0] == 1:
                st.success("The model predicts that the individual has Parkinson's disease.")
            else:
                st.success("The model predicts that the individual does not have Parkinson's disease.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
elif nav == "Liver Disease":
    st.header("Liver Disease Prediction")

    # Load the liver's model
    try:
        liver_model = pickle.load(open('liver.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        st.stop()             
def liver_input():
    st.write("Liver Disease Prediction")
    Age = st.number_input("Age", min_value=1, max_value=120, value=50)

    Gender = st.selectbox("Gender",options = ["Male","Female"])
    Gender_map={"Male": 1, "Female": 0}
    Gender_value= Gender_map[Gender]

    Total_Bilirubin= st.number_input("Total_Bilirubin",step=0.01)
    Direct_Bilirubin = st.number_input("Direct_Bilirubin", step = 0.01)
    Alkaline_Phosphotase = st.number_input("Alkaline_Phosphotase", step = 1)
    Alamine_Aminotransferase = st.number_input("Alamine_Aminotransferase",step=1)
    Aspartate_Aminotransferase = st.number_input("Aspartate_Aminotransferase", step=1)
    Total_Protiens = st.number_input("Total_Protiens",step=1)
    Albumin = st.number_input("Albumin",step=0.01)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin_and_Globulin_Ratio",step=0.01)
    placeholder2=st.empty()
    if st.button("Liver Disease Prediction"):
        input_features = np.array([[ Age, Gender_value,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,
                            Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,
                               Albumin, Albumin_and_Globulin_Ratio   ]])
        
    for col in range(input_features.shape[1]):
        input_features[:, col] = [str(x).encode('utf-8').decode('utf-8') if isinstance(x, str) else x for x in input_features[:, col]]
    # Button for prediction
    if st.button("Predict"):
        try:
            prediction = liver_model.predict(input_features)
            if prediction[0] == 0:
                st.success("The model predicts that the individual does not have Liver disease.")
            else:
                st.success("The model predicts that the individual has Liver disease.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")      
    elif nav == "Kidney Disease":
      st.header("Kidney Disease Prediction")
    # Load the kidney model
    try:
        kidney_model = pickle.load(open(r'kidney.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        st.stop()         
def kidney_input():
# App Title
    st.write("Kidney Disease Prediction App")

# Function to get user input

    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=80)
    sg = st.number_input("Specific Gravity (e.g., 1.005)", min_value=1.000, max_value=1.030, step=0.001, format="%.3f")
    al = st.number_input("Albumin Level (0-5)", min_value=0, max_value=5, step=1)
    su = st.number_input("Sugar Level (0-5)", min_value=0, max_value=5, step=1)
    rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", options=["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", options=["present", "notpresent"])
    ba = st.selectbox("Bacteria", options=["present", "notpresent"])
    bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=0, max_value=500, value=150)
    bu = st.number_input("Blood Urea (mg/dL)", min_value=0, max_value=500, value=40)
    sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, step=0.1, value=1.2)
    sod = st.number_input("Sodium (mEq/L)", min_value=0.0, max_value=200.0, step=0.1, value=135.0)
    pot = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=20.0, step=0.1, value=4.5)
    hemo = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, step=0.1, value=15.0)
    pcv = st.number_input("Packed Cell Volume", min_value=0, max_value=60, value=40)
    wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=0, max_value=20000, value=8000)
    rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
    htn = st.selectbox("Hypertension", options=["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", options=["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", options=["yes", "no"])
    appet = st.selectbox("Appetite", options=["good", "poor"])
    pe = st.selectbox("Pedal Edema", options=["yes", "no"])
    ane = st.selectbox("Anemia", options=["yes", "no"])

    # Convert categorical inputs to numerical values
    rbc = 1 if rbc == "normal" else 0
    pc = 1 if pc == "normal" else 0
    pcc = 1 if pcc == "present" else 0
    ba = 1 if ba == "present" else 0
    htn = 1 if htn == "yes" else 0
    dm = 1 if dm == "yes" else 0
    cad = 1 if cad == "yes" else 0
    appet = 1 if appet == "good" else 0
    pe = 1 if pe == "yes" else 0
    ane = 1 if ane == "yes" else 0
    placeholder3=st.empty()
    # Create a dictionary of input data
    input_feature = np.array([[age,bp,sg,al,su,rbc,pc,pcc,ba,bgr, bu,sc,sod,pot,hemo,pcv, 
                           wc,rc,htn,dm,cad,appet, pe,ane]])

# Predict Button
    if st.button("Predict"):
        try:
            prediction = kidney_model.predict(input_features)
            if prediction[0] == 1:
                st.success("The model predicts that the individual has Kidney disease.")
            else:
                st.success("The model predicts that the individual does not have Kidney disease.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
 


if page == "Parkinson's Prediction":
    parkinsons_input()
elif page == "Liver Disease Prediction":
    liver_input()
elif page == "Kidney Disease Prediction":
    kidney_input()

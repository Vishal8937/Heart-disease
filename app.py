import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Entire App */
.stApp{
    background-color:#000000;
    color:white;
}

/* Headers */
h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

/* Input labels */
div[data-testid="stMarkdownContainer"] p{
    color:white;
}

/* Select Box */
div[data-baseweb="select"]{
    background-color:#1c1c1c !important;
    color:white !important;
}

/* Number Input */
.stNumberInput input{
    background-color:#1c1c1c !important;
    color:white !important;
}

/* Text Input */
.stTextInput input{
    background-color:#1c1c1c !important;
    color:white !important;
}

/* Slider */
.stSlider label{
    color:white !important;
}

/* Predict Button */
.stButton>button{
    width:100%;
    background:#ff4b4b;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#ff2020;
    color:white;
}

/* Success Box */
.stSuccess{
    background:#1d472b;
}

/* Error Box */
.stError{
    background:#5d1a1a;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align:center;color:#ff4b4b;'>❤️ Heart Disease Prediction by Vishal</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;font-size:18px;color:white;'>Provide the following details to check your heart disease risk.</p>",
    unsafe_allow_html=True
)

st.write("")

# ------------------ INPUTS ------------------

age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80,
    200,
    120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    100,
    600,
    200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0,
    6.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

st.write("")

# ------------------ PREDICTION ------------------

if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.write("")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
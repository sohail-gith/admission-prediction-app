import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("admission_model.pkl")

st.title("Admission Chance Predictor")

st.write("Enter your details to predict the chance of admission to a graduate program.")

gre = st.number_input("GRE Score (out of 340)", min_value=0, max_value=340, value=320)
toefl = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=110)
rating = st.slider("University Rating (1–5)", 1, 5, 3)
cgpa = st.number_input("CGPA (on 10 scale)", min_value=0.0, max_value=10.0, value=8.5, step=0.01)
research = st.radio("Research Experience", ["No", "Yes"])
research_val = 1 if research == "Yes" else 0

if st.button("Predict Admission Chance"):
    input_data = np.array([[gre, toefl, rating, cgpa, research_val]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Chance of Admission: {prediction*100:.2f}%")

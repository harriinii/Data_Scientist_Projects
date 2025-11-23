import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("Student Performance Predictor")

# LOAD MODEL
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# ---- INPUT UI ----
st.subheader("Enter student characteristics")

col1, col2, col3 = st.columns(3)

with col1:
    previous_score = st.number_input("Previous Semester Score", 0, 100, 75)
    attendance = st.number_input("Attendance Percentage", 0, 100, 80)
    study_hours = st.slider("Study Hours per Week", 0, 50, 20)
    library_usage = st.slider("Library Usage per Week", 0, 10, 5)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    parental_edu = st.selectbox("Parental Education", ["High School", "Graduate", "Postgraduate"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    internet = st.selectbox("Internet Access", ["Yes", "No"])

with col3:
    sleep = st.slider("Sleep Hours", 0, 12, 7)
    travel = st.slider("Travel Time (Hours)", 0.0, 5.0, 1.5)
    anxiety = st.slider("Test Anxiety (1-10)", 1, 10, 5)
    motivation = st.slider("Motivation Level (1-10)", 1, 10, 7)

teacher_feedback = st.selectbox("Teacher Feedback", ["Poor", "Average", "Good", "Excellent"])
tutoring = st.selectbox("Tutoring Classes", ["Yes", "No"])
sports = st.selectbox("Sports Activity", ["Yes", "No"])
extra = st.selectbox("Extra Curricular", ["Yes", "No"])
peer = st.slider("Peer Influence (1-10)", 1, 10, 5)

# ---- MAKE A DATAFRAME ----
sample = pd.DataFrame([{
    "Gender": gender,
    "Study_Hours_per_Week": study_hours,
    "Attendance_Percentage": attendance,
    "Previous_Sem_Score": previous_score,
    "Parental_Education": parental_edu,
    "Internet_Access": internet,
    "Family_Income": 50000,  # default static
    "Tutoring_Classes": tutoring,
    "Sports_Activity": sports,
    "Extra_Curricular": extra,
    "School_Type": school_type,
    "Sleep_Hours": sleep,
    "Travel_Time": travel,
    "Test_Anxiety_Level": anxiety,
    "Teacher_Feedback": teacher_feedback,
    "Motivation_Level": motivation,
    "Peer_Influence": peer,
    "Library_Usage_per_Week": library_usage
}])

# ---- PREDICT ----
if st.button("Predict Final Score"):
    pred = model.predict(sample)[0]
    st.success(f"Predicted Final Score: {pred:.2f}")

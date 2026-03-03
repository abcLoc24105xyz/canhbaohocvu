import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("academic_warning_model.pkl", "rb"))

st.title("Cảnh báo học tập")

# ===== INPUT ĐÚNG THEO MODEL =====

age = st.number_input("Age", 17, 30, 20)
training_score = st.number_input("Training Score Mixed", 0.0, 100.0, 60.0)
count_f = st.number_input("Count F Subjects", 0, 10, 0)
tuition_debt = st.selectbox("Tuition Debt", [0, 1])

gender = st.selectbox("Gender", ["Male", "Female"])
admission_mode = st.selectbox("Admission Mode", ["Exam", "Direct", "Transfer"])
club_member = st.selectbox("Club Member", [0, 1])

advisor_notes = st.text_area("Advisor Notes")
personal_essay = st.text_area("Personal Essay")

# ===== PREDICT =====

if st.button("Predict"):

    df_input = pd.DataFrame({
        "Age": [age],
        "Training_Score_Mixed": [training_score],
        "Count_F": [count_f],
        "Tuition_Debt": [tuition_debt],
        "Gender": [gender],
        "Admission_Mode": [admission_mode],
        "Club_Member": [club_member],
        "Advisor_Notes": [advisor_notes],
        "Personal_Essay": [personal_essay],
        "combined_text": [advisor_notes + " " + personal_essay]
    })

    prediction = model.predict(df_input)[0]

    label_map = {
        0: "Normal",
        1: "Academic Warning",
        2: "Dropout"
    }

    st.success(f"Prediction: {label_map[prediction]}")
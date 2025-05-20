import streamlit as st
import pandas as pd
import numpy as np
import joblib  
import os

# Load model

def load_model():
    return joblib.load('random.pkl')  

model = load_model()

# Streamlit UI
st.title("AI Developer Success Prediction")
st.write("Enter the following feature values:")

# Input form
with st.form(key='input_form'):
    hours_coding = st.number_input("Hours of Coding per Day", min_value=0, max_value=24, value=6)
    coffee_intake_mg = st.number_input("Daily Coffee Intake (mg)", min_value=0, max_value=2000, value=600)
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
    commits = st.number_input("Daily Code Commits", min_value=0, max_value=100, value=3)
    bugs_reported = st.number_input("Daily Bugs Reported", min_value=0, max_value=100, value=1)
    ai_usage_hours = st.number_input("AI Tool Usage (hours/day)", min_value=0, max_value=24, value=1)
    cognitive_load = st.number_input("Cognitive Load (1-10)", min_value=1, max_value=10, value=5)
    

    submit_button = st.form_submit_button(label='Predict Success')

# Prediction
if submit_button:
    input_data = np.array([[hours_coding, coffee_intake_mg, sleep_hours, commits,
                            bugs_reported, ai_usage_hours, cognitive_load]])
    prediction = model.predict(input_data)
    if {prediction[0]}==1:
        st.success(f"Predicted as: positive")
    else:
        st.success(f"Predicted as: negative")
        
    



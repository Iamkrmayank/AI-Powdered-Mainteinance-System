# Loading Lib
import streamlit as st
import openai
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define functions for each code block
def generate_recommendation(incident_description):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Based on the following incident description, provide maintenance recommendations: {incident_description}"}]
    )
    recommendation = response['choices'][0]['message']['content']
    return recommendation

def detect_anomalies(data):
    # Fit Isolation Forest to the data
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso_forest.fit(data)
    
    # Add predictions and anomaly scores
    data['anomaly'] = iso_forest.predict(data)
    data['anomaly_score'] = iso_forest.decision_function(data.drop(columns=['anomaly'], errors='ignore'))
    anomalies = data[data['anomaly'] == -1]
    
    return data, anomalies

# Streamlit layout
st.title("Maintenance Recommendation & Anomaly Detection")

# Create tabs for the three functionalities
tab1, tab2, tab3 = st.tabs(["Maintenance Recommendation", "Anomaly Detection", "Looker Studio Report"])

# Code Block 1: Maintenance Recommendation Generator
with tab1:
    st.header("Maintenance Recommendation")
    incident_description = st.text_area("Incident Description", "Engine RPM exceeded 2500 during takeoff, potential fuel system issues.")
    
    if st.button("Generate Recommendation"):
        if incident_description.strip() == "":
            st.error("Please enter an incident description.")
        else:
            recommendation = generate_recommendation(incident_description)
            st.write("Maintenance Recommendation:", recommendation)

# Code Block 2: Anomaly Detection with Isolation Forest
with tab2:
    st.header("Anomaly Detection")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
        
        if st.button("Detect Anomalies"):
            # Ensure data does not contain non-numeric types for Isolation Forest
            if data.select_dtypes(include='number').shape[1] == 0:
                st.error("The dataset must contain numeric columns for anomaly detection.")
            else:
                # Run anomaly detection
                data, anomalies = detect_anomalies(data)
                
                st.write(f"Number of anomalies detected: {len(anomalies)}")
                st.write(anomalies)

# Code Block 3: Embed Looker Studio report
with tab3:
    st.header("Aircraft Operational Insights")
    st.warning("To view the report, ensure you have accepted third-party cookies and are logged into Google.")
    st.markdown(
        "[Click here to view the report directly in Looker Studio](https://lookerstudio.google.com/reporting/48e9ea6f-8ccd-4c70-b389-e1dc8ad93f36)"
    )


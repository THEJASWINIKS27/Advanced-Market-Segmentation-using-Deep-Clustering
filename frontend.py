import streamlit as st
import pandas as pd
import requests

st.title("Customer Segmentation App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    
    if st.button("Predict Segments"):
        # Fix: send file as tuple (filename, file bytes)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            response = requests.post("http://127.0.0.1:8000/predict/", files=files)
            if response.status_code == 200:
                result = response.json()
                st.write("Segmented Data:")
                st.dataframe(pd.DataFrame(result))
            else:
                try:
                    error_message = response.json().get("error", "Error processing file")
                except Exception:
                    error_message = "Error processing file"
                st.error(f"Error: {error_message}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server at http://127.0.0.1:8000. "
                     "Please make sure the backend server is running.")

import streamlit as st
import numpy as np
import joblib

# Load trained model and encoder
model = joblib.load("model/box_office_model.pkl")
genre_encoder = joblib.load("model/genre_encoder.pkl")

st.set_page_config(page_title="Box Office Revenue Predictor", layout="centered")

st.title("🎬 Box Office Revenue Prediction App")
st.write("Predict worldwide box office revenue using Machine Learning")

st.divider()

# User Inputs
domestic = st.number_input("Domestic Revenue (USD)", min_value=0.0, step=1_000_000.0)
foreign = st.number_input("Foreign Revenue (USD)", min_value=0.0, step=1_000_000.0)
genre = st.selectbox("Movie Genre", genre_encoder.classes_)
year = st.number_input("Release Year", min_value=1950, max_value=2030, step=1)

if st.button("🎯 Predict Worldwide Revenue"):
    genre_encoded = genre_encoder.transform([genre])[0]
    input_data = np.array([[domestic, foreign, genre_encoded, year]])
    prediction = model.predict(input_data)[0]
    st.success(f"🌍 Predicted Worldwide Revenue: **${prediction:,.2f} USD**")

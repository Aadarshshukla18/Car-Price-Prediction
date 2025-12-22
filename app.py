import streamlit as st
import numpy as np
import joblib

# Load model and scaler

model = joblib.load("best_car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Predict used car prices using Machine Learning")

st.markdown("---")

# User Inputs

year = st.number_input("Manufacturing Year", 1990, 2025, 2018)
present_price = st.number_input("Present Price (in Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner = st.selectbox("Number of Owners", [0, 1, 2, 3])

# Encoding 

fuel_map = {"CNG": 0, "Diesel": 1, "Petrol": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Automatic": 0, "Manual": 1}

fuel_encoded = fuel_map[fuel]
seller_encoded = seller_map[seller]
trans_encoded = trans_map[transmission]

# Prepare Input 

input_data = np.array([[
    year,
    present_price,
    kms_driven,
    fuel_encoded,
    seller_encoded,
    trans_encoded,
    owner
]])

# Scale input

input_scaled = scaler.transform(input_data)

# Prediction

if st.button("🔮 Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"Estimated Car Price: ₹ {round(prediction[0], 2)} Lakhs")

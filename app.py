import streamlit as st
import pickle
import numpy as np

# Load model
with open('price_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("House Price Predictor")

# Inputs
total_sqft = st.number_input("Total Square Feet", value=1000.0)
bathroom = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
balcony = st.selectbox("Number of Balconies", [0, 1, 2, 3])
bhk = st.selectbox("Number of BHK", [1, 2, 3, 4, 5])

if st.button("Predict Price"):
    input_data = np.array([[total_sqft, bathroom, balcony, bhk]])
    price = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {price:.2f} Lakhs")

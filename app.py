import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "rf_model.pkl"

# load model and scaler
if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# title
st.title("House price prediction app")

# input variables
square_footage = st.number_input("Square Footage", min_value=200.0, value=1500.0)
num_bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
num_bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)
year_built = st.number_input("Year Built", min_value=1900, value=2000, step=1)
lot_size = st.number_input("Lot Size", min_value=0.1, value=2.5)
garage_size = st.number_input("Garage Size", min_value=0, value=1, step=1)
neighborhood_quality = st.number_input(
    "Neighborhood Quality (1-10)", min_value=1, max_value=10, value=7, step=1
)

# create a dataframe
input_features = pd.DataFrame(
    {
        "Square_Footage": [square_footage],
        "Num_Bedrooms": [num_bedrooms],
        "Num_Bathrooms": [num_bathrooms],
        "Year_Built": [year_built],
        "Lot_Size": [lot_size],
        "Garage_Size": [garage_size],
        "Neighborhood_Quality": [neighborhood_quality],
    }
)

if st.button("Predict"):
    prediction = model.predict(input_features.astype(float))
    price = prediction[0]
    st.success(f"Estimated price: ${price:,.2f}")

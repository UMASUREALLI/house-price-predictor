
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Configure the webpage
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")
st.write("Enter the details of a house to predict its estimated price.")

# Generate example training data
rng = np.random.default_rng(42)

area = rng.integers(500, 4000, 500)
bedrooms = rng.integers(1, 7, 500)
age = rng.integers(0, 40, 500)

price = (
    area * 4000
    + bedrooms * 500000
    - age * 30000
    + rng.normal(0, 300000, 500)
)
price = np.maximum(price, 500000)

X = np.column_stack([area, bedrooms, age])

# Train the ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X, price)

# Collect user inputs
st.subheader("Enter House Details")

house_area = st.slider(
    "Area (square feet)", 500, 4000, 1500
)
house_bedrooms = st.slider(
    "Number of bedrooms", 1, 6, 3
)
house_age = st.slider(
    "Age of house (years)", 0, 40, 5
)

# Predict price
if st.button("Predict Price"):
    features = np.array([[
        house_area,
        house_bedrooms,
        house_age
    ]])

    prediction = model.predict(features)[0]

    st.success(
        f"Estimated example price: ₹{prediction:,.0f}"
    )

st.caption(
    "Educational demonstration using synthetic data. "
    "This is not a real property valuation model."
)

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("medicine_model.pkl")

st.set_page_config(
    page_title="Medicine Demand Prediction",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Medicine Demand Prediction")
st.write("Machine Learning Based Demand Forecasting System")

st.divider()

# User Inputs
medicine = st.selectbox(
    "Select Medicine",
    [
        "Paracetamol",
        "Cetirizine",
        "Azithromycin",
        "Vitamin C",
        "Ibuprofen"
    ]
)

category_mapping = {
    "Paracetamol": "Pain Relief",
    "Cetirizine": "Allergy",
    "Azithromycin": "Antibiotic",
    "Vitamin C": "Supplement",
    "Ibuprofen": "Pain Relief"
}

category = category_mapping[medicine]

month = st.selectbox(
    "Select Month",
    list(range(1, 13))
)

previous_sales = st.number_input(
    "Previous Sales",
    min_value=0,
    value=100
)

price = st.number_input(
    "Medicine Price",
    min_value=1,
    value=50
)

stock = st.number_input(
    "Current Stock",
    min_value=0,
    value=100
)

# Prediction Button
if st.button("Predict Demand"):

    input_data = pd.DataFrame([{
        "medicine": medicine,
        "category": category,
        "month": month,
        "previous_sales": previous_sales,
        "price": price,
        "stock": stock
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(0, round(prediction))

    st.success(f"Predicted Demand: {prediction} units")

    if prediction > stock:
        st.warning(
            f"Possible shortage: predicted demand is "
            f"{prediction - stock} units more than current stock."
        )
    else:
        st.info(
            f"Current stock is {stock - prediction} units "
            f"above the predicted demand."
        )
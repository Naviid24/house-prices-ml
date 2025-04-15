import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set the title of the web page (browser tab)
st.set_page_config(page_title="House Price Prediction App")

model =  joblib.load("model.pkl")

st.title("House Price Prediction App")
st.divider()
st.write("This app uses machine learning for predicting house price with given features of the house. For using this app you can enter the inputs from this UI and then use predict button.")
st.divider()

bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0 )
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0 )
livingarea = st.number_input("Living area", min_value=0, value=0 )
condition = st.number_input("Condition", min_value=0, value=0)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=0)

st.divider()


# Format input as DataFrame (matching model's training format)
X = pd.DataFrame([{
    'number of bedrooms': bedrooms,
    'number of bathrooms': bathrooms,
    'living area': livingarea,
    'condition of the house': condition,
    'Number of schools nearby': numberofschools
}])

# Button with a unique key
if st.button("Predict!", key="predict_button"):
    prediction = model.predict(X)[0]
    st.balloons()
    st.success(f"🏠 Estimated House Price: £{prediction:,.2f}")
else:
    st.info("Enter values and click the Predict button to get a result.")

#order of x ['number of bedrooms', 'number of bathrooms', 'living area',
#'condition of the house', 'Number of schools nearby']
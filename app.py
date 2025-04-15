import streamlit as st
import joblib
import Numpy as np

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

X = [[bedrooms, bathrooms, livingarea, condition, numberofschools]]


 #creating predict button
predictbutton = st.button("Predict!")

if predictbutton:
    st.ballons()
    X_array = np.array(X)
    prediction = model.predict(X_array)
    st.write(f"Price prediction is {prediction}")

else:
     st.write("Please use predict button after entering values")

#order of x ['number of bedrooms', 'number of bathrooms', 'living area',
       #'condition of the house', 'Number of schools nearby']
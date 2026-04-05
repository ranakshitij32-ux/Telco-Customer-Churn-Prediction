#Senior_Citizen -> 1 Yes 0 No

#Churn -> 1 Yes 0 No

#Scaler and One Hot Encoder are exported as scaler.pkl and oh_encoder.pkl respectively
#InternetService = 'DSL'- >0, 'Fiber optic' -> 1, 'No' -> 2
#Contract = 'Month-to-month' -> 0, 'One year' -> 1, 'Two year' -> 2
# PaymentMethod = 'Bank transfer (automatic)'=0, 'Credit card (automatic)'=1, 'Electronic check'=2, 'Mailed check'=3
#Model is exported as model.pkl
#OH_columns=['InternetService','Contract','PaymentMethod']
# scaler_column = ["tenure","MonthlyCharges","TotalCharges"]

#Order of OH_X: 'tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService',
#'Contract', 'PaymentMethod', 'SeniorCitizen'

import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("scaler.pkl")
OH_encoder = joblib.load("oh_encoder.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button to get prediction")

st.divider()
age = st.number_input("Enter your Age",min_value=10,max_value=110,value=20)

tenure = st.number_input("Enter tenure",min_value=0,max_value=130,value=10)

monthly_charge = st.number_input("Enter the monthly charge",min_value=0,max_value=10000,value=20)

internet_service_map = {
    "DSL",
    "Fiber optic",
    "No"
}
internet_service = st.selectbox("Select your internet service",internet_service_map)


# Contract mapping
contract_map = {
    "Month-to-month",
    "One year",
    "Two year"
}
contract = st.selectbox("Select your plan", contract_map)
# 
# Payment Method mapping
payment_map = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
}
payment_method = st.selectbox("Select payment method", payment_map)

# [array(['DSL', 'Fiber optic', 'No'], dtype=object), array(['Month-to-month', 'One year', 'Two year'], dtype=object), array(['Bank transfer (automatic)', 'Credit card (automatic)',
#        'Electronic check', 'Mailed check'], dtype=object)]
st.divider()

predict_button = st.button("Predict")

if predict_button:



    total_charges = monthly_charge*tenure
    senior_citizen = 1 if age>=60 else 0
    
    # Create X in the correct order for scaler
    scaler_X = np.array([[tenure,monthly_charge,total_charges]])
    scaler_Xarray  = scaler.transform(scaler_X)

    # Create X in the correct order for O_encoded
    OH_encoder_X = np.array([[internet_service,contract,payment_method]])
    OH_encoder_X_array = OH_encoder.transform(OH_encoder_X)
    senior_citizen_array = np.array([[senior_citizen]])  # shape (1, 1)

    # main_x = pd.concat(scaler_Xarray,OH_encoder_X_array,senior_citizen_array)
    final_input = np.hstack((scaler_Xarray, OH_encoder_X_array, senior_citizen_array))  # shape (1, 7)

    preds = model.predict(final_input)[0]
    predicted = "Customer is likely to Churn" if preds == 1 else "Customer's Churn probability is low"
    st.write(predicted)
else:
    st.write("Please enter the values and use predict button")
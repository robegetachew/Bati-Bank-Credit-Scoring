import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the trained model
model = joblib.load(r'C:\Users\user\Desktop\Kifiya\Bati_Bank_Credit_Scoring\models\logistic_regression_model_selected.pkl')

# Load the imputer and scaler fitted with only the selected features
imputer = joblib.load(r'C:\Users\user\Desktop\Kifiya\Bati_Bank_Credit_Scoring\models\imputer_selected.pkl')
scaler = joblib.load(r'C:\Users\user\Desktop\Kifiya\Bati_Bank_Credit_Scoring\models\scaler_selected.pkl')

st.title('Credit Scoring Prediction')

# Input fields for the user
amount = st.number_input('Amount', min_value=0.0, step=1.0)
value = st.number_input('Value', min_value=0.0, step=1.0)
frequency = st.number_input('Frequency', min_value=0, step=1)
monetary = st.number_input('Monetary', min_value=0.0, step=1.0)
recency = st.number_input('Recency', min_value=0, step=1)

# Button to make prediction
if st.button('Predict'):
    # Prepare the input data
    features = np.array([[amount, value, frequency, monetary, recency]])
    
    # Preprocess the input data
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)[:, 1]

    # Display the prediction and probability
    if prediction[0] == 1:
        st.success(f'The prediction is GOOD with a probability of {proba[0]:.2f}')
    else:
        st.error(f'The prediction is BAD with a probability of {proba[0]:.2f}')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the retrained model for the selected features
model = joblib.load(r'C:\Users\user\Desktop\Kifiya\Bati_Bank_Credit_Scoring\models\logistic_regression_model_selected.pkl')

# Load the imputer and scaler fitted with only the selected features
imputer = joblib.load(r'C:\Users\user\Desktop\Kifiya\Bati_Bank_Credit_Scoring\models\imputer_selected.pkl')
scaler = joblib.load(r'C:\Users\user\Desktop\Kifiya\Bati_Bank_Credit_Scoring\models\scaler_selected.pkl')

# Define the request model with top 5 feature aliases
class CreditScoringRequest(BaseModel):
    Amount: float
    Value: float
    Frequency: int
    Monetary: float
    Recency: int

@app.get("/")
def home():
    return {"message": "Credit Scoring Model API is up and running!"}

@app.post("/predict")
def predict(request: CreditScoringRequest):
    try:
        features = np.array([[request.Amount, request.Value, request.Frequency, request.Monetary, request.Recency]])

        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)

        prediction = model.predict(features_scaled)
        proba = model.predict_proba(features_scaled)[:, 1]

        response = {'prediction': int(prediction[0]), 'probability': float(proba[0])}

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    app.run(debug=True)



#uvicorn main:app --reload --port 8080
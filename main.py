# This file will contain the FastAPI endpoint for my ml application
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field
import joblib

app = FastAPI()
model = joblib.load('models/decision_tree_model.pkl')

class InputData(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str

@app.post('/predict')
def predict(data: InputData):

    # Convert Pydantic input to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Prediction
    prediction = model.predict(df)[0]

    # Probabilities
    probability = model.predict_proba(df)[0]

    if prediction == 1:
        return {
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'The applicant is likely to default on the loan.'
        }

    else:
        return {
            'prediction': int(prediction),
            'probability': float(probability[0]),
            'message': 'The applicant is likely to repay the loan.'
        }


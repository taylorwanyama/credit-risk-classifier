# This file will contain the FastAPI endpoint for my ml application
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field
import joblib
import json
from pydantic import field_validator


app = FastAPI()
model = joblib.load('models/decision_tree_model.pkl')
ALLOWED = json.load(open('models/allowed_values.json')) 

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

    @field_validator(*ALLOWED.keys())
    @classmethod
    def check_allowed(cls, v, info):
        allowed = ALLOWED[info.field_name]
        if v not in allowed:
            raise ValueError(f"{info.field_name} must be one of {allowed}")
        return v

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


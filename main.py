# This file will contain the FastAPI endpoint for my ml application
from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from pydantic import BaseModel, Field
import joblib
import json
from pydantic import field_validator
import time 
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
#To turn down http library logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'models' / 'decision_tree_model.pkl')

with open(BASE_DIR / 'models' / 'allowed_values.json') as f:
    ALLOWED = json.load(f)

#print(type(ALLOWED), len(ALLOWED), list(ALLOWED.keys()))

if not ALLOWED:
    raise RuntimeError("allowed_values.json is empty, cannot build validators")

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

# Adding a middleware to log the request and response
@app.middleware("http")
async def measure_request_time(request: Request, call_next):

    start_time = time.perf_counter()

    response = await call_next(request)

    elapsed_time = time.perf_counter() - start_time

    logger.info(
        '%s %s - %.4f seconds',
        request.method,
        request.url.path,
        elapsed_time
    )
    return response    

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


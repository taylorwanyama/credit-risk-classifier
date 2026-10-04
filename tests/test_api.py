from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_prediction_endpoint():
    response = client.post(
      '/predict',
      json = {
         "checking_status": "<0",
         "duration": 6,
         "credit_history": "critical/other existing credit",
         "purpose": "radio/tv",
         "credit_amount": 1169,
         "savings_status": "no known savings",
         "employment": ">=7",
         "installment_commitment": 4,
         "personal_status": "male single",
         "other_parties": "none",
         "residence_since": 4,
         "property_magnitude": "real estate",
         "age": 67,
         "other_payment_plans": "none",
         "housing": "own",
         "existing_credits": 2,
         "job": "skilled",
         "num_dependents": 1,
         "own_telephone": "yes",
        "foreign_worker": "yes"
     }

    )
    assert response.status_code == 200

def test_predict_missing_fields():
    response = client.post(
        '/predict',
        json = {
            "checking_status": "<0",
            "duration": 6,
            "credit_history": "critical/other existing credit",
            "purpose": "radio/tv",
            "credit_amount": 1169,
            "savings_status": "no known savings",
            "employment": ">=7",
            "installment_commitment": 4,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 4,
            "property_magnitude": "real estate",
            "age": 67,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 2,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": ""
       }

    )
    assert response.status_code == 422



def test_predict_response():
    response = client.post(
        '/predict',
        json = {
            "checking_status": "<0",
            "duration": 6,
            "credit_history": "critical/other existing credit",
            "purpose": "radio/tv",
            "credit_amount": 1169,
            "savings_status": "no known savings",
            "employment": ">=7",
            "installment_commitment": 4,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 4,
            "property_magnitude": "real estate",
            "age": 67,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 2,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": "yes"
       }

    )
    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "probability" in body
    assert "message" in body

def test_health_check():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}    
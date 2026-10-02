import joblib
import pandas as pd

def test_model_can_be_loaded():
    model = joblib.load('models/decision_tree_model.pkl')
    assert model is not None

def test_model_can_make_predictions():
    model = joblib.load('models/decision_tree_model.pkl')
    sample_input = pd.DataFrame([{
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

    }])

    prediction = model.predict(sample_input)
    assert len(prediction) == 1

def test_model_predictions():
    model = joblib.load('models/decision_tree_model.pkl')
    sample_input = pd.DataFrame([{
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

    }])

    prediction = model.predict(sample_input)
    assert prediction[0] in [0, 1]



  


  
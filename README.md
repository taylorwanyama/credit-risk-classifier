<<<<<<< HEAD
# Credit Risk Prediction API

A machine learning API that predicts whether a loan applicant is likely to default, built on the German Credit dataset and served with FastAPI.

## Overview

This project trains and compares three classifiers: Logistic Regression, Decision Tree, and Random Forest — to predict credit risk (`good` vs `bad`), then exposes the best-performing model through a REST API for real-time predictions.

The target is imbalanced (roughly 30% `bad` / 70% `good`), and in credit risk, a missed default (false negative) is far more costly than a false alarm on a good applicant (false positive) — an applicant wrongly flagged can appeal or get manual review, but an approved default is a direct financial loss. Model selection and threshold choice below are made with that asymmetry in mind, not on accuracy alone.

## Dataset

The dataset contains 1,000 loan applications with 20 features covering checking account status, credit history, loan purpose, credit amount, employment, age, and other applicant attributes. No missing values were present. Numeric features (e.g. `duration`, `credit_amount`) vary widely in scale, so scaling was applied for the linear model; categorical features were one-hot encoded for all models.

Data was split 80/20 with stratification on the target to preserve class balance in both sets.

## Model comparison

All three models were evaluated on the same held-out test set (200 samples: 140 `good`, 60 `bad`). Decision Tree and Random Forest were tuned via cross-validated search (`GridSearchCV` / `RandomizedSearchCV`) optimizing an F2 score, which weights recall over precision to reflect the cost of missed defaults.

| Model | Accuracy | Recall (bad) | Precision (bad) | False Negatives | False Positives |
|---|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.53 | 0.67 | 28 | 16 |
| **Decision Tree (tuned)** | 0.60 | **0.87** | 0.42 | **8** | 71 |
| Random Forest (tuned) | 0.76 | 0.40 | 0.65 | 36 | 13 |

### Why Decision Tree was chosen

Ranking models on accuracy alone would favor Logistic Regression or Random Forest. But applying a cost-weighted comparison — assuming a missed default costs roughly 5x a false alarm (`cost = 5 × FN + FP`), consistent with the cost structure documented for this dataset — changes the ranking:

| Model | Cost (5·FN + FP) |
|---|---|
| Logistic Regression | 156 |
| **Decision Tree (tuned)** | **111** |
| Random Forest (tuned) | 193 |

The Decision Tree was selected as the model served by the API. It misses only 8 of 60 actual defaulters (87% recall), compared to 36 missed by Random Forest and 28 by Logistic Regression. This comes at a real cost: overall accuracy drops to 60%, and 71 of 140 good applicants are flagged for review. In a lending context, that trade-off is intentional — the cost of approving a defaulter is judged to outweigh the cost of over-flagging safe applicants, who can go through manual review rather than being outright rejected.

This is a business decision, not a purely statistical one, and the threshold/model can be revisited if the cost assumption changes.

## API

### Endpoint

`POST /predict`

**Request body:**

```json
{
  "checking_status": "<0",
  "duration": 24,
  "credit_history": "existing paid",
  "purpose": "radio/tv",
  "credit_amount": 2000,
  "savings_status": "<100",
  "employment": "1<=X<4",
  "installment_commitment": 3,
  "personal_status": "male single",
  "other_parties": "none",
  "residence_since": 2,
  "property_magnitude": "car",
  "age": 30,
  "other_payment_plans": "none",
  "housing": "own",
  "existing_credits": 1,
  "job": "skilled",
  "num_dependents": 1,
  "own_telephone": "yes",
  "foreign_worker": "yes"
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.82,
  "message": "The applicant is likely to default on the loan."
}
```

- `prediction`: `0` = good (likely to repay), `1` = bad (likely to default)
- `probability`: model's estimated probability of the predicted class

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Running with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

The API will be available at `http://localhost:8000/predict`, with interactive docs at `http://localhost:8000/docs`.

## Project structure

```
.
├── data/                   # Training data
├── models/                 # Serialized trained model
├── training.ipynb          # EDA, preprocessing, model training & comparison
├── main.py                 # FastAPI application
├── Dockerfile
├── requirements.txt
└── README.md
```

## Limitations & notes

- The dataset is relatively small (1,000 rows), so test-set metrics carry meaningful variance — a 200-sample test set means single-case swings move recall/precision noticeably. CV scores are a more stable reference than the single test-set split reported above.
- Several features (`age`, `personal_status`, `foreign_worker`) are sensitive attributes in a credit-scoring context. This project does not currently include a fairness audit; before any real-world use, disparate impact across these groups should be evaluated.
- Predicted probabilities are not calibrated (the Decision Tree was tuned with `class_weight='balanced'`), so they should be read as relative risk scores rather than true probabilities.
=======
# credit-risk-classifier
>>>>>>> 3cc4576e977ae3ce43c575fa5345f4f55275fe85

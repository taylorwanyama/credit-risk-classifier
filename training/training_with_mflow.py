import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
# Imported metrics to resolve your F2 and recall placeholders
from sklearn.metrics import recall_score, fbeta_score 

#mlflow.set_tracking_uri('mlruns/')
# Set the experiment name
mlflow.set_experiment('credit-risk-classifier')

with mlflow.start_run():
    # 1. Data Preparation
    df = pd.read_csv('data/credit_risk_data.csv')
    X = df.drop(columns=['class'])
    y = df['class'] 
    y = y.map({'good': 0, 'bad': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Automatically grab numerical and categorical columns
    num_cols = X_train.select_dtypes(include=['int64','float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    # 2. Pipeline Setup
    tree_prep = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    # Define model hyperparameters in a dictionary for easy re-use
    params = {
        'class_weight': 'balanced', 
        'criterion': 'entropy', 
        'max_depth': 3,  # Changed from 3 to 5 to simulate a second run 
        'min_samples_split': 2, 
        'min_samples_leaf': 1, 
        'random_state': 42
    }

    pipeline_dt = Pipeline([
        ('transformer', tree_prep),
        ('classifier', DecisionTreeClassifier(**params)) # Pass parameters cleanly
    ])
    
    # 3. Model Training & Evaluation
    pipeline_dt.fit(X_train, y_train)
    predictions = pipeline_dt.predict(X_test)

    # Compute your actual metrics
    recall = recall_score(y_test, predictions)
    F2 = fbeta_score(y_test, predictions, beta=2)

    # 4. Logging to MLflow
    mlflow.log_params(params)

    mlflow.log_metrics({
        'F2': F2,
        'recall': recall
    })
    
    # Infer the dataset structure for safer deployment later; it documents the expected input/output structure.
    signature = mlflow.models.infer_signature(X_test, predictions)

    # Log the complete workflow
    mlflow.sklearn.log_model(
        sk_model=pipeline_dt,
        artifact_path='credit_risk_model',
        signature=signature,
        serialization_format="pickle",
        registered_model_name="credit-risk-classifier"
    )
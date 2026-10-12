import mlflow
from mlflow import MlflowClient

client = MlflowClient()

model_name = "credit-risk-classifier"
model_version = "1"

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version
)

print(f"{model_name} version {model_version} is now champion.")

champion = client.get_model_version_by_alias(
    name=model_name,
    alias="champion"
)

print("Champion version:", champion.version)
print("Model URI:", f"models:/{model_name}@champion")
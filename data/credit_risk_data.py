import openml
import pandas as pd

dataset = openml.datasets.get_dataset("credit-g")

X, y, categorical_indicator, attribute_names = dataset.get_data(
    target="class"
)

print(X.head())
print(y.head())
print(X.shape)

credit_risk_df = pd.concat([X,y], axis=1)
credit_risk_df.to_csv('credit_risk_data.csv', index=False)
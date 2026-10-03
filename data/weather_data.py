# This file demonstrates how we can retrieve weather data from an api, inspect it and convert it into a pandas dataframe.
#
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": -1.286389,
    "longitude": 36.817223,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "cloud_cover"
    ],
    "timezone": "Africa/Nairobi"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# To see the raw features
print(data.keys())
print(data["hourly"].keys())

# Inspecting the data we retrieved
df = pd.DataFrame(data["hourly"])

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
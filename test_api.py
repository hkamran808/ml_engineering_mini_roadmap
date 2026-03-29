import requests
import pandas as pd
import numpy as np

X_sample = pd.read_csv("application_train.csv").head(1)
sample = X_sample.drop(columns=["SK_ID_CURR", "TARGET"]).to_dict(orient="records")[0]

sample = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in sample.items()} #to avoid error with NaN values in the input, we replace them with None, since our JSON does not support it

response = requests.post("http://127.0.0.1:8000/predict", json=sample)
print(response.status_code)
print(response.text)
print(response.json())
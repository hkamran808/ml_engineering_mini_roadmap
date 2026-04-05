from msgspec import field
import pandas as pd
from fastapi import FastAPI
from predictor import CreditRiskPredictor
from pydantic import BaseModel
from typing import Optional
import csv
import datetime
from evidently import Report
from evidently.presets import DataDriftPreset

import os
log_file = "prediction_log.csv"
file_exists = os.path.exists(log_file)

app = FastAPI()
predictor = CreditRiskPredictor("credit_risk_lgbm.pkl", "label_encoders.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}
##############################################################################################
class ApplicantInput(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float

    class Config:  # we add this to allow extra fields in the input, since our model expects more features, they being in new test api file
        extra = "allow"
@app.post("/predict")
def predict(applicant: ApplicantInput):
    our_result = predictor.predict(applicant.model_dump())  # replacement for .dict() in newer pydantic version v2
    with open(log_file, mode="a") as file:
        writer = csv.DictWriter(file, fieldnames=list(applicant.model_dump().keys()) + ["timestamp", "prediction"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({**applicant.model_dump(), "timestamp": datetime.datetime.now(), "prediction": our_result})
    return {"default_probability": our_result}
import pandas as pd
from fastapi import FastAPI
from predictor import CreditRiskPredictor
from pydantic import BaseModel
from typing import Optional
import csv
import datetime
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

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
    with open("prediction_log.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now(), applicant.model_dump(), our_result])
    return {"default_probability": our_result}
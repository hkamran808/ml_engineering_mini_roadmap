from fastapi import FastAPI
from predictor import CreditRiskPredictor
from pydantic import BaseModel
from typing import Optional

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
    # we may add other features as needed, matching the training data
    class Config:
        extra = "allow"
@app.post("/predict")
def predict(applicant: ApplicantInput):
    our_result = predictor.predict(applicant.model_dump())  # replacement for .dict() in newer pydantic version v2
    return {"default_probability": our_result}
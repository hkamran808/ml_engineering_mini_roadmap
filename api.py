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

report = Report(metrics=[DataDriftPreset()])
column_mapping = ColumnMapping()
# column mapping to be defined based on the dataset...
report.run(reference_data=pd.read_csv("application_train.csv").drop(columns=["SK_ID_CURR", "TARGET"]), 
            current_data=pd.read_csv("application_test.csv").drop(columns=["SK_ID_CURR"]), 
            column_mapping=column_mapping)
#report.show() #optional, to visualize the report in a browser, we can also save it as an HTML file using report.save_html
result_dict = report.as_dict()

drift_results = []
metrics = result_dict["metrics"][0]["result"]["drift_by_column"]
for feature, data in metrics.items():
    drift_results.append({
        "feature": feature,
        "drift_score": data["drift_score"],
        "is_drift": data["drift_detected"]})
    
drift_dataframe = pd.DataFrame(drift_results).sort_values(by="drift_score", ascending=False)
print(drift_dataframe.head())
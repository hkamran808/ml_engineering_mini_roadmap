import pandas as pd
#from evidently.pipeline.column_mapping import ColumnMapping
from evidently import Report
from evidently.presets import DataDriftPreset

reference = pd.read_csv("application_train.csv").drop(columns=["SK_ID_CURR", "TARGET"])
current = pd.read_csv("application_test.csv").drop(columns=["SK_ID_CURR"])

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
#report.to_html("drift_report.html")
print("Drift report saved to drift_report.html")

"""
report = Report(metrics=[DataDriftPreset()])
column_mapping = ColumnMapping()
# column mapping to be defined based on the dataset...
report.run(reference_data=pd.read_csv("application_train.csv").drop(columns=["SK_ID_CURR", "TARGET"]), 
            current_data=pd.read_csv("prediction_log.csv").drop(columns=["timestamp", "prediction"]), 
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
"""
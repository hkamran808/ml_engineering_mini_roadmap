import pandas as pd
#from evidently.pipeline.column_mapping import ColumnMapping
from evidently import Report
from evidently.presets import DataDriftPreset

reference = pd.read_csv("application_train.csv").drop(columns=["SK_ID_CURR", "TARGET"])
current = pd.read_csv("application_test.csv").drop(columns=["SK_ID_CURR"])
#current=pd.read_csv("prediction_log.csv").drop(columns=["timestamp", "prediction"])

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
#report.to_html("drift_report.html")
print("Drift report saved to drift_report.html")
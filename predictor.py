import joblib
import pandas as pd

class CreditRiskPredictor:
    def __init__(self, model_path, encoders_path):
        self.model_path = model_path
        self.encoders_path = encoders_path

        self.model = joblib.load(self.model_path)
        self.encoders = joblib.load(self.encoders_path)  # "label_encoders.pkl" for categorical features

    def engineer_features(self, df):  #our feature engineering function, same as in training file
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
        return df

    def preprocess(self, applicant_data):
        df = pd.DataFrame([applicant_data])
        for col, le in self.encoders.items():
            if col in applicant_data:
                df[col] = le.transform(df[col].astype(str).values)
        df = self.engineer_features(df)
        df = df.apply(pd.to_numeric, errors='ignore')
        return df
    
    def predict(self, applicant_data):
        df1 = self.preprocess(applicant_data)
        return float(self.model.predict_proba(df1)[:, 1][0])

if __name__ == "__main__":
    X_sample = pd.read_csv("application_train.csv").head(1)
    sample_applicant = X_sample.drop(columns=["SK_ID_CURR", "TARGET"]).to_dict(orient="records")[0]

    predictor = CreditRiskPredictor("credit_risk_lgbm.pkl", "label_encoders.pkl") #old model, just for testing
    result = predictor.predict(sample_applicant)
    print(result)
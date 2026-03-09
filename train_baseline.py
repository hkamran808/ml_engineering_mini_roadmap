import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. BASELINE DATA
train = pd.read_csv("application_train.csv")
test = pd.read_csv("application_test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 2. CLEANING
train = train.drop(columns=["SK_ID_CURR"])
test_ids = test["SK_ID_CURR"]
test = test.drop(columns=["SK_ID_CURR"])

y = train["TARGET"]
X = train.drop(columns=["TARGET"])

# 3. HANDLING CATEGORICAL FEATURES
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    
    le = LabelEncoder()
    
    combined = pd.concat([X[col], test[col]], axis=0)
    le.fit(combined.astype(str))
    
    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


# 4. SIMPLE FEATURE ENGINEERING
X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / (X["AMT_INCOME_TOTAL"] + 1)
X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / (X["AMT_INCOME_TOTAL"] + 1)

test["CREDIT_INCOME_RATIO"] = test["AMT_CREDIT"] / (test["AMT_INCOME_TOTAL"] + 1)
test["ANNUITY_INCOME_RATIO"] = test["AMT_ANNUITY"] / (test["AMT_INCOME_TOTAL"] + 1)

# 5. CLEANING FEATURE NAMES
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)


# 6. LIGHTGBM PARAMETERS
lgb_params = {

    "n_estimators": 10000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "random_state": 42,
    "n_jobs": -1}


# 7. CROSS VALIDATION
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
feature_importances = np.zeros(X.shape[1])


# 8. TRAINING LOOP TO GET OOF PREDICTIONS AND FEATURE IMPORTANCES

for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):

    print(f"\nFold {fold+1}")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)])

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    fold_auc = roc_auc_score(y_val, val_preds)
    print("Fold AUC:", fold_auc)

    feature_importances += model.feature_importances_ / skfold.n_splits


# 9. FINAL METRICS
full_auc = roc_auc_score(y, oof_preds)
print("\nOverall OOF ROC AUC:", full_auc)

# 10. TRAIN FINAL MODEL
final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X, y)

# 11. MODEL SAVING
joblib.dump(final_model, "credit_risk_lgbm.pkl")
print("Model saved.")

# 12. TEST PREDICTIONS
test_preds = final_model.predict_proba(test)[:, 1]
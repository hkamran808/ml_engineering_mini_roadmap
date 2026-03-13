import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("application_train.csv")
test = pd.read_csv("application_test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train = train.drop(columns=["SK_ID_CURR"])
test_ids = test["SK_ID_CURR"]
test = test.drop(columns=["SK_ID_CURR"])

y = train["TARGET"]
X = train.drop(columns=["TARGET"])

cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    
    le = LabelEncoder()
    
    combined = pd.concat([X[col], test[col]], axis=0)
    le.fit(combined.astype(str))
    
    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / (X["AMT_INCOME_TOTAL"] + 1)
X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / (X["AMT_INCOME_TOTAL"] + 1)

test["CREDIT_INCOME_RATIO"] = test["AMT_CREDIT"] / (test["AMT_INCOME_TOTAL"] + 1)
test["ANNUITY_INCOME_RATIO"] = test["AMT_ANNUITY"] / (test["AMT_INCOME_TOTAL"] + 1)

X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

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

"""
# day2
from sklearn.model_selection import GridSearchCV
param_grid = {
    "num_leaves": [31, 64, 128],
    "max_depth": [-1, 8, 12],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_samples": [20, 50],
    "feature_fraction": [0.7, 0.8, 0.9]}

base_model = lgb.LGBMClassifier(n_estimators=2000, random_state=1, n_jobs=-1)
grid_search = GridSearchCV(estimator=base_model, 
                           param_grid=param_grid, 
                           cv=3, 
                           scoring="roc_auc", 
                           verbose=2, 
                           n_jobs=-1)
grid_search.fit(X, y)
print(10*"-", "best parameters\n", 10*"-")
print(grid_search.best_params_)
print(10*"-", "best roc-auc\n", 10*"-")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_
joblib.dump(best_model, "best_lgbm_model_fromDAY2.pkl")
"""

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

import optuna
def objective(trial):
    num_leaves = trial.suggest_int("num_leaves", 20, 300)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 12)
    min_child_samples = trial.suggest_int("min_child_samples", 10, 150)
    model = lgb.LGBMClassifier(num_leaves=num_leaves, 
                               learning_rate=learning_rate,
                               max_depth=max_depth,
                               min_child_samples=min_child_samples,
                               n_estimators=300,
                               random_state=1,
                               n_jobs=-1,
                               verbose=-1)
    model.fit(X_train, y_train)
    
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(10*"-", "Data retrieved from this optimization", 10*"-")
print(f"Best Parameters: {study.best_params} - Best Value: {study.best_value}")

import json
with open("best_params_day3.json", "w") as f:
    json.dump(study.best_params, f, indent=2)
joblib.dump(study, "optuna_study_day3.pkl")

"""
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

oof_preds = np.zeros(len(X))
feature_importances = np.zeros(X.shape[1])

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

full_auc = roc_auc_score(y, oof_preds)
print("\nOverall OOF ROC AUC:", full_auc)

final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X, y)

joblib.dump(final_model, "credit_risk_lgbm.pkl")
print("Model saved.")

test_preds = final_model.predict_proba(test)[:, 1]

# day1
print(final_model.get_params())

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importances
}).sort_values("importance", ascending=False)

print(importance_df.head(20))
"""
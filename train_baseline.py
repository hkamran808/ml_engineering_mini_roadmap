import mlflow
mlflow.set_experiment("credit-risk-home_project")

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

flag_cols = [col for col in X.columns if col.startswith("FLAG_")]

X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / (X["AMT_INCOME_TOTAL"] + 1)
X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / (X["AMT_INCOME_TOTAL"] + 1)
"""
X["EMPLOYED_TO_AGE_RATIO"] = X["DAYS_EMPLOYED"].abs() / (X["DAYS_BIRTH"].abs() + 1)
X["ANNUITY_TO_CAR_AGE"] = X["AMT_ANNUITY"] / (X["OWN_CAR_AGE"].fillna(0) + 1)
X["FLAG_COUNT"] = X[flag_cols].sum(axis=1)
"""
test["CREDIT_INCOME_RATIO"] = test["AMT_CREDIT"] / (test["AMT_INCOME_TOTAL"] + 1)
test["ANNUITY_INCOME_RATIO"] = test["AMT_ANNUITY"] / (test["AMT_INCOME_TOTAL"] + 1)
"""
test["EMPLOYED_TO_AGE_RATIO"] = test["DAYS_EMPLOYED"].abs() / (test["DAYS_BIRTH"].abs() + 1)
test["ANNUITY_TO_CAR_AGE"] = test["AMT_ANNUITY"] / (test["OWN_CAR_AGE"].fillna(0) + 1)
test["FLAG_COUNT"] = test[flag_cols].sum(axis=1)
"""
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

import json
RUN_OPTUNA = False
if RUN_OPTUNA:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print(10*"-", "Data retrieved from this optimization", 10*"-")
    print(f"Best Parameters: {study.best_params} - Best Value: {study.best_value}")
    with open("best_params_day3.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    joblib.dump(study, "optuna_study_day3.pkl")

with open("best_params_day3.json", "r") as f:
    best_params = json.load(f)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
"""
oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_OOF, y_train_OOF = X.iloc[train_idx], y.iloc[train_idx]
    X_val_OOF, y_val_OOF = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**best_params, 
                               n_estimators=300, 
                               random_state=1, 
                               n_jobs=-1, 
                               verbosity=-1)
    model.fit(X_train_OOF, y_train_OOF)
    
    oof_preds[val_idx] = model.predict_proba(X_val_OOF)[:, 1]
    print(f"Fold {fold+1} AUC:", roc_auc_score(y_val_OOF, oof_preds[val_idx]))

print("Overall OOF AUC:", roc_auc_score(y, oof_preds))
"""
# stacking: multiple models and OOFs *logistic reg and lgbm in our case, just for now
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=300, n_jobs=-1)

oof_lgbm = np.zeros(len(X))
oof_logreg = np.zeros(len(X))

with mlflow.start_run(run_name="Baseline LGBM with Stacking"):
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train_OOF_logreg, y_train_OOF_logreg = X_scaled.iloc[train_idx], y.iloc[train_idx]
        X_val_OOF_logreg, y_val_OOF_logreg = X_scaled.iloc[val_idx], y.iloc[val_idx]

        logreg.fit(X_train_OOF_logreg, y_train_OOF_logreg)
        oof_logreg[val_idx] = logreg.predict_proba(X_val_OOF_logreg)[:, 1]

        X_train_OOF_lgbm, y_train_OOF_lgbm = X.iloc[train_idx], y.iloc[train_idx]
        X_val_OOF_lgbm, y_val_OOF_lgbm = X.iloc[val_idx], y.iloc[val_idx]
        lgbm_stack = lgb.LGBMClassifier(**best_params, 
                                n_estimators=300, 
                                random_state=1, 
                                n_jobs=-1, 
                                verbosity=-1)
        
        lgbm_stack.fit(X_train_OOF_lgbm, y_train_OOF_lgbm)

        oof_lgbm[val_idx] = lgbm_stack.predict_proba(X_val_OOF_lgbm)[:, 1]
        print(f"Fold {fold+1} LGBM AUC:", roc_auc_score(y_val_OOF_lgbm, oof_lgbm[val_idx]))
        print(f"Fold {fold+1} LogReg AUC:", roc_auc_score(y_val_OOF_logreg, oof_logreg[val_idx]))

    print("Overall OOF LGBM AUC:", roc_auc_score(y, oof_lgbm))
    print("Overall OOF LogReg AUC:", roc_auc_score(y, oof_logreg))

    # stacking up models: creating meta-features and training a meta-model on top of them
    meta_X = np.column_stack((oof_lgbm, oof_logreg))
    meta_model = LogisticRegression(max_iter=300, n_jobs=-1)
    meta_model.fit(meta_X, y)

    from sklearn.model_selection import cross_val_score
    meta_auc = cross_val_score(meta_model, 
                            meta_X, y, 
                            cv=skf, scoring="roc_auc")
    print("Meta-model CV AUC:", meta_auc.mean())

    mlflow.log_params(best_params)
    mlflow.log_metric("OOF_LGBM_AUC", roc_auc_score(y, oof_lgbm))
    mlflow.log_metric("OOF_LogReg_AUC", roc_auc_score(y, oof_logreg))
    mlflow.log_metric("Meta_Model_CV_AUC", meta_auc.mean())
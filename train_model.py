"""
train_model.py — Run this ONCE to train and save the model.
Uses the Kaggle credit card fraud dataset (creditcard.csv).

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place creditcard.csv in this folder, then run:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

print("Loading data...")
if not os.path.exists("creditcard.csv"):
    print("creditcard.csv not found — generating synthetic data for demo.")
    np.random.seed(42)
    n = 5000
    normal = np.random.randn(n, 10)
    fraud  = np.random.randn(200, 10) * 3 + 5
    X = np.vstack([normal, fraud])
    y = np.array([0] * n + [1] * 200)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 11)])
    df["Amount"] = np.abs(np.random.randn(len(df))) * 100
    df["Class"] = y
else:
    df = pd.read_csv("creditcard.csv")

features = [c for c in df.columns if c != "Class"]
X = df[features].values
y = df["Class"].values

print(f"Dataset: {len(df)} rows, {y.sum()} fraud cases ({y.mean()*100:.2f}%)")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MLflow experiment
mlflow.set_experiment("anomaly_detection")

with mlflow.start_run():
    contamination = y.mean()
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    print("Training Isolation Forest...")
    model.fit(X_scaled)

    # Predict: IsolationForest returns -1 for anomaly, 1 for normal
    preds_raw = model.predict(X_scaled)
    preds = (preds_raw == -1).astype(int)

    report = classification_report(y, preds, output_dict=True)
    precision = report["1"]["precision"]
    recall    = report["1"]["recall"]
    f1        = report["1"]["f1-score"]

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("contamination", round(contamination, 4))
    mlflow.log_metric("fraud_precision", round(precision, 4))
    mlflow.log_metric("fraud_recall",    round(recall, 4))
    mlflow.log_metric("fraud_f1",        round(f1, 4))
    mlflow.sklearn.log_model(model, "isolation_forest")

    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=["Normal", "Fraud"]))
    print(f"MLflow run logged — precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f}")

# Save locally for FastAPI
joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(features, "features.joblib")
print("\nSaved: model.joblib, scaler.joblib, features.joblib")
print("Now run: uvicorn api:app --reload")

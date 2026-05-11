"""
api.py — FastAPI app to serve the anomaly detection model.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(
    title="Anomaly Detection API",
    description="Isolation Forest model served via FastAPI. Detects anomalies in transaction data.",
    version="1.0.0",
)

# ── Load model artifacts ───────────────────────────────────────────────────────
MODEL_PATH   = "model.joblib"
SCALER_PATH  = "scaler.joblib"
FEATURE_PATH = "features.joblib"

for path in [MODEL_PATH, SCALER_PATH, FEATURE_PATH]:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing file: {path}. Run train_model.py first.")

model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
features = joblib.load(FEATURE_PATH)

print(f"Model loaded. Expects {len(features)} features: {features}")


# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.1, -1.2, 0.5, 0.3, -0.8, 1.1, 0.0, 0.7, -0.4, 0.2, 50.0]
            }
        }


class PredictResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    label: str
    num_features_expected: int


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Anomaly Detection API is running",
        "endpoints": {
            "POST /predict": "Predict if a transaction is anomalous",
            "GET  /health":  "Health check",
            "GET  /info":    "Model info",
            "GET  /docs":    "Swagger UI",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "IsolationForest", "features": len(features)}


@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "contamination": model.contamination,
        "num_features": len(features),
        "feature_names": features,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    n_expected = len(features)
    if len(request.features) != n_expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {n_expected} features, got {len(request.features)}. "
                   f"Feature names: {features}",
        )

    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # score_samples: lower = more anomalous
    raw_score = float(model.score_samples(X_scaled)[0])
    prediction = int(model.predict(X_scaled)[0])  # -1 = anomaly, 1 = normal
    is_anomaly = prediction == -1

    return PredictResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(raw_score, 6),
        label="ANOMALY" if is_anomaly else "NORMAL",
        num_features_expected=n_expected,
    )


@app.post("/predict/batch")
def predict_batch(requests: List[PredictRequest]):
    """Predict for multiple transactions at once."""
    results = []
    for req in requests:
        try:
            result = predict(req)
            results.append(result)
        except HTTPException as e:
            results.append({"error": e.detail})
    return {"predictions": results, "count": len(results)}

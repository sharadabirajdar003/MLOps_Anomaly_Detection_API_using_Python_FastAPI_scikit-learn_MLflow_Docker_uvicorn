"""
test_api.py — Quick test script to verify the API is working.
Run AFTER starting the API with: uvicorn api:app --reload
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 50)
print("Testing Anomaly Detection API")
print("=" * 50)

# 1. Health check
r = requests.get(f"{BASE_URL}/health")
print(f"\n[GET /health] {r.status_code}")
print(json.dumps(r.json(), indent=2))

# 2. Model info
r = requests.get(f"{BASE_URL}/info")
info = r.json()
n_features = info["num_features"]
print(f"\n[GET /info] Expects {n_features} features")

# 3. Normal transaction (values near 0)
normal_features = [0.0] * n_features
r = requests.post(f"{BASE_URL}/predict", json={"features": normal_features})
print(f"\n[POST /predict] Normal transaction")
print(json.dumps(r.json(), indent=2))

# 4. Anomalous transaction (extreme values)
anomaly_features = [10.0] * n_features
r = requests.post(f"{BASE_URL}/predict", json={"features": anomaly_features})
print(f"\n[POST /predict] Anomalous transaction (extreme values)")
print(json.dumps(r.json(), indent=2))

# 5. Batch predict
batch_payload = [
    {"features": [0.0] * n_features},
    {"features": [10.0] * n_features},
    {"features": [-5.0] * n_features},
]
r = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
print(f"\n[POST /predict/batch] 3 transactions")
result = r.json()
for i, pred in enumerate(result["predictions"]):
    print(f"  Transaction {i+1}: {pred['label']} (score: {pred['anomaly_score']})")

print("\nAll tests passed! Visit http://localhost:8000/docs for Swagger UI")

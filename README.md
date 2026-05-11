# Project 4 — Dockerised Anomaly Detection API (MLOps)

A production-style ML pipeline: train an Isolation Forest model with MLflow tracking, then serve it via FastAPI, then containerise with Docker.

## Files
```
project4_mlops/
├── train_model.py   ← Step 1: train & save the model
├── api.py           ← Step 2: FastAPI serving endpoint
├── test_api.py      ← Step 3: test the running API
├── Dockerfile       ← Step 4: containerise with Docker
├── requirements.txt
└── README.md
```

---

## How to run in VS Code

### Step 1 — Open folder in VS Code
```
File → Open Folder → select project4_mlops
```

### Step 2 — Open terminal
```
Terminal → New Terminal   (Ctrl + `)
```

### Step 3 — Create & activate virtual environment
```bash
python -m venv venv
```
**Windows:**
```bash
venv\Scripts\activate
```
**Mac / Linux:**
```bash
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — (Optional) Download the real dataset
Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download `creditcard.csv` and place it in this folder.

If you skip this, the script auto-generates synthetic demo data — still works fine.

### Step 6 — Train the model
```bash
python train_model.py
```
This creates `model.joblib`, `scaler.joblib`, `features.joblib` in the folder.
MLflow logs the experiment — view it with:
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

### Step 7 — Start the FastAPI server
Open a NEW terminal tab (keep the venv activated):
```bash
uvicorn api:app --reload
```
Server starts at http://localhost:8000

### Step 8 — Test the API
Open a THIRD terminal tab:
```bash
python test_api.py
```
Or open http://localhost:8000/docs for interactive Swagger UI.

---

## Docker (bonus — shows MLOps skill)

Make sure Docker Desktop is installed and running.

```bash
# Build image
docker build -t anomaly-api .

# Run container
docker run -p 8000:8000 anomaly-api

# Test it (same as before)
python test_api.py
```

---

## API Endpoints

| Method | URL | What it does |
|--------|-----|--------------|
| GET | / | Welcome + endpoint list |
| GET | /health | Health check |
| GET | /info | Model parameters |
| POST | /predict | Predict single transaction |
| POST | /predict/batch | Predict multiple transactions |
| GET | /docs | Swagger UI (interactive) |

### Example curl call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.5, 0.3, -0.8, 1.1, 0.0, 0.7, -0.4, 0.2, 50.0]}'
```

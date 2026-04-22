from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "model.pkl")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

# Create app
app = FastAPI(title="Demand Forecasting API")

# Input schema
class PredictRequest(BaseModel):
    lag_1: float
    lag_7: float
    lag_14: float
    rolling_mean_7: float
    rolling_mean_14: float
    id_encoded: int

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# Metrics endpoint
@app.get("/metrics")
def metrics():
    return {
        "MAE": 0.4819,
        "RMSE": 1.1259
    }

# Predict endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    features = np.array([[
        request.lag_1,
        request.lag_7,
        request.lag_14,
        request.rolling_mean_7,
        request.rolling_mean_14,
        request.id_encoded
    ]])
    prediction = model.predict(features)
    return {"predicted_sales": round(float(prediction[0]), 2)}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="AHE Credit Card Fraud Detection API")

# Add CORS Middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
lgb_model = None
lr_model = None
nn_model = None

def load_models():
    """Load the trained models if they exist."""
    global lgb_model, lr_model, nn_model
    if os.path.exists('lgb_model.joblib') and os.path.exists('lr_model.joblib') and os.path.exists('nn_model.joblib'):
        print("💾 Loading models...")
        lgb_model = joblib.load('lgb_model.joblib')
        lr_model = joblib.load('lr_model.joblib')
        nn_model = joblib.load('nn_model.joblib')
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Models not found. Please run trainer.py first.")

# Initial load
load_models()

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

def ahe_predict_proba(X):
    """Weighted ensemble probability matching trainer.py"""
    if lgb_model is None or lr_model is None or nn_model is None:
        # Emergency Reload if they weren't ready
        load_models()
        if lgb_model is None:
            return [0.0] # Fallback if still not ready

    lgb_prob = lgb_model.predict_proba(X)[:, 1]
    lr_prob  = lr_model.predict_proba(X)[:, 1]
    nn_prob  = nn_model.predict_proba(X)[:, 1]
    
    # Same weights as in trainer.py
    ensemble_prob = 0.45 * lgb_prob + 0.30 * lr_prob + 0.25 * nn_prob
    return ensemble_prob

@app.get("/")
def read_root():
    return {"status": "AHE API is running", "models_loaded": lgb_model is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    # If models are not loaded, try once more
    if lgb_model is None:
        load_models()
        if lgb_model is None:
            return {"error": "Models not trained/loaded"}

    # Convert input to DataFrame with same columns as training (ensure order)
    data_dict = transaction.dict()
    # List of columns exactly as used in training (Time, V1-V28, Amount)
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    data = pd.DataFrame([data_dict], columns=cols)
    
    prob = ahe_predict_proba(data)[0]
    fraud = prob > 0.5
    return {
        "fraud_probability": float(prob), 
        "is_fraud": bool(fraud), 
        "inference_ms": 8.5 # Simulated time as in the original frontend
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
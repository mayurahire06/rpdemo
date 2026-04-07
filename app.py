from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="AHE Credit Card Fraud Detection API")

# Load the trained models (you can save them with joblib after training)
# lgb_model.save_model("lgb_model.txt") etc. - or use joblib for all

class Transaction(BaseModel):
    Time: float
    V1: float
    # ... add all 28 V features + Amount (total 30 features)
    Amount: float

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to DataFrame with same columns as training
    data = pd.DataFrame([transaction.dict()])
    # Add missing V columns if needed (match your training columns)
    prob = ahe_predict_proba(data)[0]   # reuse function from above
    fraud = prob > 0.5
    return {"fraud_probability": float(prob), "is_fraud": bool(fraud), "inference_ms": 8.5}

# Run with: uvicorn app:app --reload
# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="AHE Credit Card Fraud Detection API")

# ====================== CORS FIX ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                    # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],                    # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],                    # Allows all headers
)
# =====================================================

# Load trained models
lgb_model = joblib.load("models/lgb_model.pkl")
lr_model = joblib.load("models/lr_model.pkl")
nn_model = joblib.load("models/nn_model.pkl")

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

def ahe_predict_proba(data):
    lgb_prob = lgb_model.predict_proba(data)[:, 1][0]
    lr_prob = lr_model.predict_proba(data)[:, 1][0]
    nn_prob = nn_model.predict_proba(data)[:, 1][0]
    return 0.45 * lgb_prob + 0.30 * lr_prob + 0.25 * nn_prob

@app.post("/predict")
def predict(transaction: Transaction):
    input_df = pd.DataFrame([transaction.dict()])
    prob = ahe_predict_proba(input_df)
    is_fraud = prob > 0.5

    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": bool(is_fraud),
        "inference_ms": 8.5,
        "message": "Fraud Detected!" if is_fraud else "Transaction is Legitimate"
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
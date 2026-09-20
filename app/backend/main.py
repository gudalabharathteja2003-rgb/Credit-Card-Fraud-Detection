from fastapi import FastAPI
from pydantic import BaseModel


# ==========================================
# CREATE FASTAPI APP
# ==========================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="GET and POST APIs",
    version="1.0.0"
)


# ==========================================
# GET API 1 - HOME
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is working",
        "status": "success"
    }


# ==========================================
# GET API 2 - HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "API is running successfully"
    }


# ==========================================
# POST INPUT MODEL
# ==========================================

class Transaction(BaseModel):
    Time: float
    Amount: float


# ==========================================
# POST API 1 - PREDICTION
# ==========================================

@app.post("/predict")
def predict(transaction: Transaction):

    time = transaction.Time
    amount = transaction.Amount

    if amount > 1000:
        result = "HIGH VALUE TRANSACTION"
    else:
        result = "NORMAL TRANSACTION"

    return {
        "status": "success",
        "Time": time,
        "Amount": amount,
        "result": result
    }


# ==========================================
# POST API 2 - RISK ASSESSMENT
# ==========================================

@app.post("/risk")
def risk_prediction(transaction: Transaction):

    time = transaction.Time
    amount = transaction.Amount

    if amount >= 2000:
        risk_level = "HIGH RISK"
        risk_score = 90

    elif amount >= 500:
        risk_level = "MEDIUM RISK"
        risk_score = 60

    else:
        risk_level = "LOW RISK"
        risk_score = 20

    return {
        "status": "success",
        "Time": time,
        "Amount": amount,
        "risk_level": risk_level,
        "risk_score": risk_score
    }
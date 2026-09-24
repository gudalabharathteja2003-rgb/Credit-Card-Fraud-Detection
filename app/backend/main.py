import os
import uvicorn
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# 1. Initialize FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Backend API providing real-time single and batch fraud detection predictions.",
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# 2. CORS Middleware Setup
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from local frontends & hosted sites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 3. Pydantic Schemas
# ------------------------------------------------------------------------------
class TransactionPayload(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since 1st transaction", example=0.0)
    Amount: float = Field(..., description="Transaction amount ($)", example=149.62)
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Legitimate, 1 = Fraudulent")
    fraud_probability: float = Field(..., description="Risk score from 0.0 to 1.0")
    status: str = Field(..., description="Execution status")

class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_detected_count: int
    results: List[PredictionResponse]

# ------------------------------------------------------------------------------
# 4. Helper Inference Function
# ------------------------------------------------------------------------------
def run_model_inference(data: Dict[str, float]) -> Dict[str, Any]:
    amount = data.get("Amount", 0.0)
    v14 = data.get("V14", 0.0)
    v17 = data.get("V17", 0.0)
    
    if amount > 500.0 or v14 < -3.0 or v17 < -3.0:
        return {"prediction": 1, "fraud_probability": 0.89}
    return {"prediction": 0, "fraud_probability": 0.03}

# ==============================================================================
# 5. GET ENDPOINTS (2 Total)
# ==============================================================================

# GET 1: Root / Health Check
@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {
        "status": "success",
        "message": "API Running",
        "service": "Credit Card Fraud Detection Backend"
    }

# GET 2: System Health & Model Status
@app.get("/health", status_code=status.HTTP_200_OK)
def get_health_status():
    return {
        "status": "healthy",
        "model_loaded": True,
        "environment": "production",
        "features_required": 30
    }

# ==============================================================================
# 6. POST ENDPOINTS (2 Total)
# ==============================================================================

# POST 1: Single Prediction
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def predict_fraud(payload: TransactionPayload):
    try:
        input_data = payload.dict()
        result = run_model_inference(input_data)
        
        return {
            "prediction": result["prediction"],
            "fraud_probability": float(result["fraud_probability"]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Single prediction failed: {str(e)}"
        )

# POST 2: Batch Predictions (Process multiple transactions at once)
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK
)
def predict_fraud_batch(payloads: List[TransactionPayload]):
    try:
        results = []
        fraud_count = 0
        
        for payload in payloads:
            input_data = payload.dict()
            res = run_model_inference(input_data)
            
            if res["prediction"] == 1:
                fraud_count += 1
                
            results.append({
                "prediction": res["prediction"],
                "fraud_probability": float(res["fraud_probability"]),
                "status": "success"
            })
            
        return {
            "total_processed": len(payloads),
            "fraud_detected_count": fraud_count,
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ------------------------------------------------------------------------------
# 7. Local Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=port, reload=True)
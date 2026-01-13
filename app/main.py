from fastapi import FastAPI, HTTPException, Request
from app.schemas import Transaction, PredictionResponse
from app.core import AnomalyDetector
import time

app = FastAPI(
    title="Anomaly Detection API",
    description="API for detecting fraudulent transactions using One-Class SVM.",
    version="1.0.0"
)

# Initialize Detector
detector = AnomalyDetector()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.on_event("startup")
def startup_event():
    try:
        detector.load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/health")
def health_check():
    if detector.model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = detector.predict(transaction)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

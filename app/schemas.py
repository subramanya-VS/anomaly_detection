from pydantic import BaseModel, Field, validator
from typing import Literal
from datetime import datetime

class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: Literal['electronics', 'gas', 'grocery', 'jewelry', 'luxury_goods', 'restaurant', 'retail']
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km")
    timestamp: datetime = Field(..., description="Transaction timestamp")

class PredictionResponse(BaseModel):
    is_fraud: bool
    anomaly_score: float
    method: str = "OneClassSVM"
    reasoning: str

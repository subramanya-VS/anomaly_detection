import joblib
import pandas as pd
import numpy as np
import math
import os
from app.schemas import Transaction, PredictionResponse

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        # Define categories in sorted order
        self.categories = ['electronics', 'gas', 'grocery', 'jewelry', 'luxury_goods', 'restaurant', 'retail']
        # 'electronics' is first, so if drop_first=True, it is dropped.
        self.feature_columns = [
            # Numeric
            "amount_log", "distance_log",
            # Cyclical
            "hour_sin", "hour_cos",
            "day_sin", "day_cos",
            "month_sin", "month_cos",
            # Categorical (One-Hot without 'electronics')
            "merchant_category_gas",
            "merchant_category_grocery",
            "merchant_category_jewelry",
            "merchant_category_luxury_goods",
            "merchant_category_restaurant",
            "merchant_category_retail"
        ]

    def load_model(self):
        # Paths
        MODEL_PATH = "model/one_class_svm.pkl"
        SCALER_PATH = "model/robust_scaler.pkl"
        
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
             raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
             raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def cyclical_encoding(self, value, max_val):
        return np.sin(2 * np.pi * value / max_val), np.cos(2 * np.pi * value / max_val)

    def preprocess(self, transaction: Transaction):
        # Extract components
        amount = transaction.amount
        distance = transaction.distance_from_home
        dt = transaction.timestamp
        
        # 1. Log transform
        amount_log = np.log1p(amount)
        distance_log = np.log1p(distance)
        
        # 2. Time extraction
        hour = dt.hour
        day_of_week = dt.weekday() # 0=Monday, 6=Sunday
        month = dt.month
        
        # 3. Cyclical encoding
        hour_sin, hour_cos = self.cyclical_encoding(hour, 24)
        day_sin, day_cos = self.cyclical_encoding(day_of_week, 7)
        month_sin, month_cos = self.cyclical_encoding(month, 12)
        
        # 4. Categorical Encoding (Manual One-Hot)
        cat = transaction.merchant_category
        
        features = {
            "amount_log": amount_log,
            "distance_log": distance_log,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "day_sin": day_sin, "day_cos": day_cos,
            "month_sin": month_sin, "month_cos": month_cos,
            "merchant_category_gas": 1 if cat == 'gas' else 0,
            "merchant_category_grocery": 1 if cat == 'grocery' else 0,
            "merchant_category_jewelry": 1 if cat == 'jewelry' else 0,
            "merchant_category_luxury_goods": 1 if cat == 'luxury_goods' else 0,
            "merchant_category_restaurant": 1 if cat == 'restaurant' else 0,
            "merchant_category_retail": 1 if cat == 'retail' else 0,
        }
        
        # Create DataFrame with exact column order
        df = pd.DataFrame([features], columns=self.feature_columns)
        return df

    def predict(self, transaction: Transaction) -> PredictionResponse:
        df = self.preprocess(transaction)
        
        # Scale
        df_scaled = self.scaler.transform(df.values)
        
        # Predict
        prediction = self.model.predict(df_scaled)[0]
        score = self.model.decision_function(df_scaled)[0]
        
        is_fraud = True if prediction == -1 else False
        
        reasoning = "Transaction fits normal patterns."
        if is_fraud:
            reasoning = f"Anomaly detected. Feature pattern deviation score: {score:.4f}."
            
        return PredictionResponse(
            is_fraud=is_fraud,
            anomaly_score=float(score),
            reasoning=reasoning
        )

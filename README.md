# Anomaly Detection with API 

## 📌 Project Overview

This project involves the generation of a high-fidelity synthetic financial dataset tailored to the Indian fintech ecosystem (INR currency, PAN card regulations, geospatial constraints) and the development of a robust anomaly detection pipeline. The system is designed to simulate realistic legitimate user behavior while injecting specific, detectable fraud vectors for model training.



## 📂 Project Structure

```
├── app/
│   ├── main.py          # FastAPI application entry point, endpoints, and startup logic
│   ├── core.py          # AnomalyDetector class: Preprocessing, model loading, and prediction
│   └── schemas.py       # Pydantic models for request/response validation
├── model/
|   ├── models.ipynb        # python notebook that trains one-class SVM, autoencoders and isolation forest
│   ├── one_class_svm.pkl   # Pre-trained One-Class SVM model
│   └── robust_scaler.pkl   # Pre-trained RobustScaler for feature normalization
├── data/
|   ├──generate_synthetic_dataset.ipynb  # python notebook to generate synthetic dataset
│   └── transactions.csv    # Synthetic transaction dataset
├── tests/
│   └── test_api.py      # Unit tests for the API endpoints
├── DECISION_LOG.md      # Record of key technical decisions and model selection rationale
└── requirements.txt     # Python dependencies
```


## Part 1: Synthetic Data Generation Specs

The dataset is generated using a combination of **stochastic processes** and **agent-based simulation** to create the following behavioral patterns.

### A — Legitimate Baseline

*These patterns establish the "normal" behavior against which fraud is detected.*

1. **Log-normal Amount Distributions**
Transaction amounts follow positive skewed distributions specific to `merchant_category` (e.g., Grocery = low mean/low variance; Jewelry = high mean/high variance).


2. **Geospatial Centroid Mobility**
Users have a `home_lat`/`home_long`. Transactions are sampled using distance decay functions around home/work centroids.


3. **Temporal Seasonality**
Probabilistic modeling of time. Peaks during lunch (12-2 PM) and evening (7-9 PM). Low activity 2-5 AM.



5. **Benford’s Law Compliance**
Legitimate prices (e.g., ₹1,243.50) follow natural digit distributions. Fraud often uses "human-invented" round numbers.



### B — Card Cloning 

6. **Impossible Travel (Velocity Violation)**
* **Rule:** `Haversine(L1, L2) / Δt > 800 km/h`. Physically impossible movement between transactions.


7. **Concurrent Session Usage**
* **Rule:** Same `card_number` used at near-identical timestamps in locations separated by >10km.


## PART2: Preprocessing Pipeline for Financial Anomaly Detection

This document details the data preprocessing pipeline implemented in the project. The pipeline is designed to transform raw transaction data into a format suitable for unsupervised anomaly detection models (**Isolation Forest, One-Class SVM, and Autoencoders**).


##  Pipeline Overview

The pipeline consists of 8 distinct phases, executed sequentially:

1. **Time based Sorting:** Enforcing chronological order to prevent data leakage.
2. **Data Cleaning:** Removing invalid transactions and duplicates.
3. **Feature Selection:** Dropping non-predictive identifiers.
4. **Feature Engineering:** Transforming time and magnitude into learnable geometric features.
5. **Categorical Encoding:** Converting categories into numerical vectors.
6. **Train/Test Splitting:** Performing a strict time-based split (Past vs. Future).
7. **Normalization:** Scaling features using robust statistics.
8. **Model-Specific Preparation:** Filtering training data for unsupervised learning.

---

### 1. Time based Ordering: 
- The dataset is sorted by `timestamp` in ascending order. This step ensures the model learns from past events to predict future anomalies.

### 2. Data Cleaning & Validation
- Remove noise and impossible values that could destabilize the models.

* **Constraints Enforced:**
* `amount > 0` (Transaction value must be positive)
* `hour` $\in [0, 23]$
* `day_of_week` $\in [0, 6]$
* `month` $\in [1, 12]$


* **Deduplication:** Exact duplicate rows are removed to prevent density bias in clustering.

### 3. Target Separation & Feature Selection

- Prevent label leakage and remove high-cardinality identifiers that act as noise.

* **Target Extraction:** The `is_fraud` label is separated into variable `y` and removed from the feature set `X`.
* **Dropped Columns:**
* `transaction_id`, `card_number`, `customer_id`, `merchant_id` (Unique Identifiers)
* `timestamp` (Replaced by engineered time features)
* `fraud_type` (Label leakage)



### 4. Feature Engineering
- Convert raw features into a geometric space understandable by distance-based algorithms (SVM/Autoencoders).

#### A. Cyclical Time Encoding

Time is circular (23:00 is close to 00:00), but raw numbers imply they are far apart. We map time features onto a unit circle using Sine and Cosine transformations.

* **Transformations:**
* `hour` $\rightarrow$ `hour_sin`, `hour_cos`
* `day_of_week` $\rightarrow$ `day_sin`, `day_cos`
* `month` $\rightarrow$ `month_sin`, `month_cos`



#### B. Logarithmic Transformation

Financial transaction amounts and distances often follow a "Power Law" distribution (heavy tails). We compress these distributions to make them closer to a Normal distribution, which stabilizes Gradient Descent.

* **Transformations:**
* `amount` $\rightarrow$ `amount_log` (using `np.log1p`)
* `distance_from_home` $\rightarrow$ `distance_log` (using `np.log1p`)



### 5. Categorical Encoding
- **One-Hot Encoding** is applied to `merchant_category`.

### 6. Train / Test Split (Time-Based)
Simulate a real-world production environment where we train on historical data and predict on future data.

* **Method:** Strict **80/20 Time Split**.
* **Training Set:** First 80% of rows (Historical).
* **Testing Set:** Last 20% of rows (Future).
* *Note: Random shuffling is explicitly disabled to prevent "future leakage".*

### 7. Data Normalization

- Scale all features to a comparable range so that features with large values (like Amount) don't dominate the loss function.

- **Scaler Used:** **RobustScaler**. Unlike Standard or MinMax scalers, RobustScaler uses the Median and Interquartile Range (IQR). It is resistant to the extreme outliers often found in fraud datasets.

### 8. Unsupervised Training Setup

- Enable "Normality Learning" for Autoencoders and One-Class SVM.

- A specific training set `X_train_ae` is created by removing all fraud instances (`y_train == 1`) from the training data. The models are trained **only on legitimate transactions** so they learn to reconstruct "normal" patterns. During testing, they flag anything they cannot reconstruct (fraud) as an anomaly.
* *Note: Isolation Forest is trained on the full, contaminated training set.*

---

## Final Feature Set

The processed data fed into the models consists of:

* `merchant_lat`, `merchant_long` (Geospatial)
* `amount_log`, `distance_log` (Log-transformed Magnitude)
* `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos` (Cyclical Time)
* `merchant_category_*` (One-Hot Encoded Categories)

## PART 3: Anomaly Detection API

This project includes a production-ready FastAPI application to serve the One-Class SVM model.

### ⚙️ Setup & Running

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server**
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

### 📡 API Endpoints

#### `POST /predict`

Detects if a transaction is anomalous (fraudulent).

**Request Body** (JSON):

| Field | Type | Description |
| :--- | :--- | :--- |
| `amount` | float | Transaction amount (must be > 0) |
| `merchant_category` | string | Category: `electronics`, `gas`, `grocery`, `jewelry`, `luxury_goods`, `restaurant`, `retail` |
| `distance_from_home` | float | Distance from user home in km |
| `timestamp` | string (ISO 8601) | Transaction timestamp (e.g., `2025-09-15T14:30:00Z`) |

**Example Request**:
```json
{
  "amount": 5000.0,
  "merchant_category": "electronics",
  "distance_from_home": 12.5,
  "timestamp": "2025-09-15T14:30:00Z"
}
```

**Success Response (200 OK)**:
```json
{
  "is_fraud": false,
  "anomaly_score": 0.1234,
  "method": "OneClassSVM",
  "reasoning": "Transaction fits normal patterns."
}
```

**Error Response**:
- **422 Validation Error**: If fields are missing or invalid (e.g., negative amount).
- **500 Internal Server Error**: If model prediction fails.

#### `GET /health`

Checks if the API and model are loaded correctly.

**Response**:
```json
{
  "status": "healthy"
}
```

### 🧪 Running Tests

Unit tests are included to verify API functionality.

```bash
pytest tests/
```

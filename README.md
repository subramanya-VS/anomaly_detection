# Anomaly Detection with API 

## 📌 Project Overview

This project involves the generation of a high-fidelity synthetic financial dataset tailored to the Indian fintech ecosystem (INR currency, PAN card regulations, geospatial constraints) and the development of a robust anomaly detection pipeline. The system is designed to simulate realistic legitimate user behavior while injecting specific, detectable fraud vectors for model training.


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

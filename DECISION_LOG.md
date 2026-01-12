# ITERATION 1

Experiment: StandardScaler + random split for isolation forest
Action: Replaced RobustScaler with StandardScaler and used train_test_split(shuffle=True).
Result: reported Precision: 0.3569, Recall: 0.8388, F1: 0.5008


# ITERATION 2
Experiment: Implement One-Class SVM trained on removing the fraud samples (cell 14)
Result: metrics change from Precision: 0.1638, Recall: 0.4383, F1: 0.2385 to Precision: 0.2557, Recall: 0.8766, F1: 0.3959. Thus maximum change in recall observed 
Why this happened: One-Class SVM assumes all training samples are normal. Including fraud samples during training contaminated the learned decision boundary, causing the model to treat fraudulent behavior as part of the normal manifold and reducing anomaly detection effectiveness.


# ITERATION 1
I tried StandardScaler + random split for isolation forest, it didn't work because StandardScaler is sensitive to extreme outliers, so I switched to RobustScaler with StandardScaler and used train_test_split(shuffle=True)

# ITERATION 2
I tried to Implement One-Class SVM trained on without removing the fraud samples, it didn't work because One-Class SVM assumes all training samples are normal. Including fraud samples during training contaminated the learned decision boundary, causing the model to treat fraudulent behavior as part of the normal manifold and reducing anomaly detection effectiveness, so I switched to Implement One-Class SVM trained on removing the fraud samples (cell 14)

# ITERATION 3


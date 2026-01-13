# ITERATION 1
I tried StandardScaler + random split for isolation forest, it didn't work because StandardScaler is sensitive to extreme outliers, so I switched to RobustScaler with StandardScaler and used train_test_split(shuffle=True)

# ITERATION 2
I tried to Implement One-Class SVM trained on without removing the fraud samples, it didn't work because One-Class SVM assumes all training samples are normal. Including fraud samples during training contaminated the learned decision boundary, causing the model to treat fraudulent behavior as part of the normal manifold and reducing anomaly detection effectiveness, so I switched to Implement One-Class SVM trained on removing the fraud samples (cell 14) This Resulted in metrics change from Precision: 0.1638, Recall: 0.4383, F1: 0.2385 to Precision: 0.2557, Recall: 0.8766, F1: 0.3959 in positive samples i.e. is_fraud=1

# ITERATION 3
I tried to implement AutoEncoders with one more dense 128 layer hoping to improve any metrics, it didnt work because original (64→32→16→32→64) AE was already expressive enough to model the normal manifold, so adding a 128 layer did not improve detection, so I switched back to the original architecture with early stopping to prevent overfitting


# ITERATION 4: 
Observation :
The Autoencoder achieved near-perfect performance (Precision ≈ 0.99, Recall ≈ 1.00, F1 ≈ 0.996), which was inconsistent with the behavior of Isolation Forest and One-Class SVM and unrealistic for an unsupervised fraud detection task.

I tried to Train the Autoencoder using a clean preprocessing pipeline and evaluated it using a percentile-based threshold computed directly from the test-set reconstruction errors. it didnt work because The target label is_fraud was accidentally retained in the feature matrix during early experiments, allowing the model to implicitly learn the ground truth., so I Explicitly removed is_fraud from the feature matrix immediately after extracting the target variable and Recomputed the Autoencoder anomaly threshold using only reconstruction errors from training-set normal transactions, ensuring no test data was used during threshold selection.
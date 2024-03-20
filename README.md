# Credit Card Fraud Detection

This project investigates various classification models for credit card fraud detection on imbalanced data. Hyperparameter tuning, SMOTE, and ADASYN techniques were employed to optimize model performance.

### Project Goals

* Compare classification models for credit card fraud detection.
* Address class imbalance in the dataset.
* Evaluate models using metrics beyond accuracy (AUC, precision, recall, F1-score).

### Data Analysis and Preprocessing

* Explored a credit card fraud dataset with features transformed using Principal Component Analysis (PCA).
* Identified significant class imbalance (fraudulent transactions: 0.172%).
* Performed exploratory data analysis (EDA) to understand feature relationships.
* Analyzed feature distributions and applied power transformations for normalization.

### Model Building and Evaluation

* Evaluated five classification models: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost.
* Tuned hyperparameters for each model using GridSearchCV with stratified K-fold cross-validation.
* Assessed model performance using AUC and classification reports.

#### Model Performance without Balancing

| Model | AUC | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.98 | 0.88 | 0.62 | 0.73 |
| KNN | 0.90 | 0.96 | 0.74 | 0.84 |
| Decision Tree | 0.89 | 0.86 | 0.77 | 0.81 |
| Random Forest | 0.97 | 0.95 | 0.72 | 0.82 |
| XGBoost | 0.98 | 0.94 | 0.74 | 0.83 |

### Handling Class Imbalance

* Implemented SMOTE and ADASYN to oversample the minority class (fraudulent transactions).
* Re-trained and evaluated all models on the balanced datasets.

**Table: Model Performance with balancing**

| Model | AUC | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---|---|---|---|---|
| Logistic Regression with SMOTE | 0.98 | 0.05 | 0.88 | 0.10 |
| Logistic Regression with ADASYN | 0.98 | 0.02 | 0.91 | 0.04 |
| KNN with SMOTE | 0.92 | 0.65 | 0.83 | 0.73 |
| KNN with ADASYN | 0.92 | 0.65 | 0.83 | 0.73 |
| Decision Tree with SMOTE | 0.96 | 0.02 | 0.87 | 0.04 |
| Decision Tree with ADASYN | 0.92 | 0.02 | 0.90 | 0.04 |
| Random Forest with SMOTE | 0.98 | 0.39 | 0.84 | 0.53 |
| Random Forest with ADASYN | 0.98 | 0.05 | 0.88 | 0.10 |
| XGBoost with SMOTE | 0.98 | 0.72 | 0.83 | 0.77 |
| XGBoost with ADASYN | 0.98 | 0.67 | 0.85 | 0.75 |


#### Key Observations

* kNN initially performed well with a higher f1-score(0.84). However, its AUC score(0.90) revealed a limitation: while it caught more fraud, it also mistakenly classified some legitimate transactions as fraudulent.
* XGBoost with SMOTE achieved a balanced performance (AUC: 0.98, F1: 0.77).This indicates it effectively distinguished fraudulent transactions from legitimate ones, minimizing the misclassification of legitimate transactions as fraudulent.
* Random Forest with SMOTE exhibited a trade-off between AUC (0.98) and F1-score (0.57) due to lower precision.

### Choosing the Best Model

* Prioritized a model with a good balance between precision and recall to minimize false positives.
* XGBoost with SMOTE emerged as the best model (AUC: 0.98, F1: 0.77) due to its balance between precision (0.72) and recall (0.83).
* Random Forest with SMOTE also had a high AUC but a lower F1-score, indicating a trade-off not ideal for this scenario.

### Conclusion

This project identified XGBoost with SMOTE as the most suitable model for credit card fraud detection in this case. It effectively balances identifying fraudulent transactions with maintaining accuracy for legitimate transactions.

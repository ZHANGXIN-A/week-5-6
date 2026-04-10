# Week05-06 HW03 — Tree Ensemble to Boosting

## 1. Dataset Introduction

Dataset: Titanic
Problem Type: Binary Classification
Target Variable: Survived

The Titanic dataset is a tabular dataset used for predicting passenger survival.
It contains both numerical and categorical features, which makes it suitable
for comparing tree-based ensemble methods such as Random Forest and Boosting models.

---

## 2. Experimental Setup

Train / Validation / Test Split: 70% / 15% / 15%
Metrics: Accuracy, F1-score, ROC-AUC

The dataset was split into training, validation, and test sets.
The validation set was used for hyperparameter tuning and early stopping,
while the test set is reserved for final evaluation.

---

## 3. Model Comparison

### Validation Results

| Model         | Accuracy | F1-score | ROC-AUC |
| ------------- | -------- | -------- | ------- |
| Random Forest | 0.8582   | 0.8041   | 0.9206  |
| XGBoost       | 0.8657   | 0.8235   | 0.9145  |
| LightGBM      | 0.8806   | 0.8491   | 0.9219  |

LightGBM achieved the best baseline performance among the three models,
followed by XGBoost and Random Forest.

---

## 4. Hyperparameter Tuning

Hyperparameter tuning was performed on the XGBoost model to improve validation performance.

The following parameters were adjusted:

* n_estimators: 200 → 500
  Increasing the number of trees allows the model to learn more complex patterns.

* learning_rate: 0.05 → 0.03
  A smaller learning rate helps the model learn more gradually and improves generalization.

* max_depth: 3 → 4
  Slightly deeper trees allow the model to capture more feature interactions.

* subsample: default → 0.9
  Subsampling helps reduce overfitting and improves model robustness.

### Validation Results After Tuning

| Model         | Accuracy | F1-score | ROC-AUC |
| ------------- | -------- | -------- | ------- |
| Random Forest | 0.8582   | 0.8041   | 0.9206  |
| XGBoost       | 0.8657   | 0.8235   | 0.9145  |
| LightGBM      | 0.8806   | 0.8491   | 0.9219  |
| XGBoost Tuned | 0.8657   | 0.8269   | 0.9162  |

After tuning, the performance slightly improved.
The F1-score increased from 0.8235 to 0.8269 and ROC-AUC increased from 0.9145 to 0.9162.
This indicates that hyperparameter tuning helped improve model generalization.

---

## 5. Early Stopping

Early stopping was applied to the tuned XGBoost model.

The model was trained with a large number of estimators (1000),
but training stopped automatically when validation performance
did not improve for 30 rounds.

This helps prevent overfitting and reduces unnecessary training.

### Validation Results After Early Stopping

| Model                  | Accuracy | F1-score | ROC-AUC |
| ---------------------- | -------- | -------- | ------- |
| Random Forest          | 0.8582   | 0.8041   | 0.9206  |
| XGBoost                | 0.8657   | 0.8235   | 0.9145  |
| LightGBM               | 0.8806   | 0.8491   | 0.9219  |
| XGBoost Tuned          | 0.8657   | 0.8269   | 0.9162  |
| XGBoost Early Stopping | 0.8731   | 0.8350   | 0.9191  |

After applying early stopping, the model performance improved:

* Accuracy: 0.8657 → 0.8731
* F1-score: 0.8269 → 0.8350
* ROC-AUC: 0.9162 → 0.9191

This shows that early stopping helped the model generalize better.

---

## 6. Final Model Comparison

| Model                  | Accuracy   | F1-score   | ROC-AUC    |
| ---------------------- | ---------- | ---------- | ---------- |
| Random Forest          | 0.8582     | 0.8041     | 0.9206     |
| XGBoost                | 0.8657     | 0.8235     | 0.9145     |
| **LightGBM (Best)**    | **0.8806** | **0.8491** | **0.9219** |
| XGBoost Tuned          | 0.8657     | 0.8269     | 0.9162     |
| XGBoost Early Stopping | 0.8731     | 0.8350     | 0.9191     |

---

## 7. Analysis

Random Forest is based on bagging and builds trees independently,
while boosting models build trees sequentially to correct errors.
In this experiment, boosting models generally performed better.

LightGBM achieved the best baseline performance.
However, after tuning and early stopping, XGBoost improved significantly.

Early stopping helped improve performance and reduce overfitting.

The final selected model is LightGBM because it achieved the highest
validation ROC-AUC and F1-score among baseline models.

---

## 8. Final Model Selection

LightGBM achieved the highest validation performance among baseline models,
with the best F1-score (0.8491) and ROC-AUC (0.9219).

Although XGBoost improved after tuning and early stopping,
LightGBM still maintained the strongest overall performance.

Therefore, LightGBM was selected as the final model.

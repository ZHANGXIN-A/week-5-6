# Week05-06 HW03 — Tree Ensemble to Boosting

## Pre-submission Checklist

[ok] Dataset description included
[ok]Train / validation / test structure clearly defined
[ok]Random Forest, XGBoost, and LightGBM compared
[ok]At least two hyperparameters tuned
[ok]Early stopping applied
[ok]Final model selection rationale provided

---

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
while the test set was reserved for final evaluation.

---

## 3. Model Comparison

### Validation Results

| Model         | Accuracy | F1-score | ROC-AUC |
| ------------- | -------- | -------- | ------- |
| Random Forest | 0.8582   | 0.8041   | 0.9206  |
| XGBoost       | 0.8657   | 0.8235   | 0.9145  |
| LightGBM      | 0.8806   | 0.8491   | 0.9219  |

LightGBM achieved the best baseline validation performance among the three models,
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

This shows that early stopping helped the model generalize better on the validation set.

---

## 6. Final Validation Comparison

| Model                             | Accuracy   | F1-score   | ROC-AUC    |
| --------------------------------- | ---------- | ---------- | ---------- |
| Random Forest                     | 0.8582     | 0.8041     | 0.9206     |
| XGBoost                           | 0.8657     | 0.8235     | 0.9145     |
| **LightGBM (Best on Validation)** | **0.8806** | **0.8491** | **0.9219** |
| XGBoost Tuned                     | 0.8657     | 0.8269     | 0.9162     |
| XGBoost Early Stopping            | 0.8731     | 0.8350     | 0.9191     |

---

## 7. Final Test Evaluation

| Model                      | Accuracy   | F1-score   | ROC-AUC    |
| -------------------------- | ---------- | ---------- | ---------- |
| **Random Forest**          | **0.7761** | 0.6429     | **0.8110** |
| **XGBoost Early Stopping** | 0.7612     | **0.6596** | 0.7777     |
| LightGBM                   | 0.7537     | 0.6526     | 0.7783     |

On the held-out test set, Random Forest achieved the best Accuracy and ROC-AUC,
while XGBoost Early Stopping achieved the best F1-score.
LightGBM, although strongest on the validation set, showed lower generalization performance on the test set.

---

## 8. Analysis

Random Forest is based on bagging and builds trees independently,
while boosting models build trees sequentially to correct previous errors.
In the validation results, boosting models generally performed better.

LightGBM achieved the best validation performance,
and XGBoost improved after tuning and early stopping.

However, the final test results show that the best validation model
did not necessarily produce the best generalization performance.
Random Forest achieved the best test Accuracy and ROC-AUC,
which suggests that it was more stable on unseen data in this experiment.

Early stopping improved XGBoost on the validation set,
and it also achieved the highest test F1-score among the tested final models.

---

## 9. Final Model Selection

Based on the validation set, LightGBM was the strongest model during model comparison.
However, based on the held-out test set, Random Forest achieved the best overall generalization performance,
with the highest Accuracy (0.7761) and ROC-AUC (0.8110).

Therefore, Random Forest can be considered the final model in terms of test performance,
while LightGBM can be regarded as the best validation model.

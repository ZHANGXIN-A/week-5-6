# Week05-06 HW03 — Tree Ensemble to Boosting

## 1. Dataset Introduction
Dataset: Titanic
Problem Type: Binary Classification
Target Variable: Survived

## 2. Experimental Setup
Train / Validation / Test Split: 70% / 15% / 15%
Metrics: Accuracy, F1-score, ROC-AUC

## 3. Model Comparison

### Validation Results

| Model | Accuracy | F1-score | ROC-AUC |
|------|----------|---------|--------|
| Random Forest | 0.8582 | 0.8041 | 0.9206 |
| XGBoost | 0.8657 | 0.8235 | 0.9145 |
| LightGBM | 0.8806 | 0.8491 | 0.9219 |


| 

## 4. Hyperparameter Tuning
### Validation Results

| Model | Accuracy | F1-score | ROC-AUC |
|------|----------|---------|--------|
| Random Forest | 0.8582 | 0.8041 | 0.9206 |
| XGBoost | 0.8657 | 0.8235 | 0.9145 |
| LightGBM | 0.8806 | 0.8491 | 0.9219 |
| XGBoost Tuned | 0.8657 | 0.8269 | 0.9162 |

Hyperparameter tuning was performed on the XGBoost model to improve validation performance.

The following parameters were adjusted:

- n_estimators: 200 → 500  
  Increasing the number of trees allows the model to learn more complex patterns.

- learning_rate: 0.05 → 0.03  
  A smaller learning rate helps the model learn more gradually and improves generalization.

- max_depth: 3 → 4  
  Slightly deeper trees allow the model to capture more feature interactions.

- subsample: default → 0.9  
  Subsampling helps reduce overfitting and improves model robustness.

After tuning, the performance slightly improved.  
The F1-score increased from 0.8235 to 0.8269 and ROC-AUC increased from 0.9145 to 0.9162.  
This indicates that hyperparameter tuning helped improve model generalization.
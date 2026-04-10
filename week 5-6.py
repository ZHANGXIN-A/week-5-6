import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import lightgbm

df = pd.read_csv("titanic.csv")

features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target].copy()

# 类别变量转数字
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 缺失值填补
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])


# 先切出 test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 再从 train_full 里切出 validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
)

# 这样大概就是 70 / 15 / 15
print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("X_test shape:", X_test.shape)


def evaluate_model(model, X_data, y_true, name="Dataset"):
    y_pred = model.predict(X_data)
    y_prob = model.predict_proba(X_data)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)

    print(f"\n[{name}]")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    return acc, f1, roc


from sklearn.ensemble import RandomForestClassifier

# Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# train
rf.fit(X_train, y_train)

# validation evaluation
print("\n===== Random Forest (Validation) =====")
evaluate_model(rf, X_valid, y_valid, "Validation")

# XGBoost baseline
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

print("\n===== XGBoost (Validation) =====")
xgb_valid_result = evaluate_model(xgb, X_valid, y_valid, "Validation")
#import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 1. Load dataset
df = pd.read_csv("titanic.csv")

# 2. Define features and target
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
target = "Survived"

X = df[features].copy()
y = df[target].copy()

# 3. Minimal preprocessing
# categorical -> numeric
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Embarked"] = X["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# missing value handling
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Fare"] = X["Fare"].fillna(X["Fare"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# 4. Train / Validation / Test split
# first split: test = 15%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# second split: validation from remaining 85%
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
)

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("X_test shape:", X_test.shape)


# 5. Evaluation function
def evaluate_model(model, X_data, y_true, name="Dataset"):
    y_pred = model.predict(X_data)
    y_prob = model.predict_proba(X_data)[:, 1]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)

    print(f"\n[{name}]")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    return acc, f1, roc


# 6. Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf.fit(X_train, y_train)

print("\n===== Random Forest (Validation) =====")
rf_valid_result = evaluate_model(rf, X_valid, y_valid, "Validation")


# 7. XGBoost baseline
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

print("\n===== XGBoost (Validation) =====")
xgb_valid_result = evaluate_model(xgb, X_valid, y_valid, "Validation")

# 7. XGBoost tuned
xgb_tuned = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    random_state=42,
    eval_metric="logloss"
)

xgb_tuned.fit(X_train, y_train)

print("\n===== XGBoost Tuned (Validation) =====")
xgb_tuned_valid_result = evaluate_model(xgb_tuned, X_valid, y_valid, "Validation")


# 8. LightGBM baseline
lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=15,
    random_state=42,
    verbose=-1
)

lgbm.fit(X_train, y_train)

print("\n===== LightGBM (Validation) =====")
lgbm_valid_result = evaluate_model(lgbm, X_valid, y_valid, "Validation")
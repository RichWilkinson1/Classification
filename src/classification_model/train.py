import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from classification_model import data_load, features, evaluation

# Central model registry (name → class + params)
MODEL_REGISTRY = {
    "xgb": {
        "class": XGBClassifier,
        "params": {"random_state":42, "use_label_encoder":False, "eval_metric":'logloss'}
    },
    "rf": {
        "class": RandomForestClassifier,
        "params": {"n_estimators":100, "random_state":42}
    },
    "logistic": {
        "class": LogisticRegression,
        "params": {"solver": "lbfgs", "max_iter":1000, "random_state":42}
    },
    "lgbm": {
        "class": LGBMClassifier,
        "params": {"random_state":42}
    },
}

def build_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name='{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    cfg = MODEL_REGISTRY[model_name]
    return cfg["class"](**cfg["params"])

def train_and_evaluate(model_name: str = "xgboost"):
    
    """
    End-to-end training pipeline: load data, preprocess, train model, and evaluate.
    """
    
    # 1. Load raw data
    df = data_load.load_data()

    # 2. Preprocess features and target
    X, y = features.preprocess_data(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Handle class imbalance (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Training size before/after SMOTE: {len(X_train)}/{len(X_train_bal)}")

    # 5. Initialize and train the chosen classifier
    model = build_model(model_name)   # <--- Build selected model from registry
    model.fit(X_train_bal, y_train_bal)

    # 6. Evaluate on test set
    y_pred = model.predict(X_test)
    evaluation.evaluate_model(y_test, y_pred)

    if hasattr(model, "feature_importances_"):
        # For tree-based models
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_train_bal.columns)
        feat_imp = feat_imp.sort_values(ascending=False)
        print("Top features by importance:\n", feat_imp.head(10))
    elif hasattr(model, "coef_"):
        # For linear models like LogisticRegression
        coefs = model.coef_[0]  # get the coefficients array
        feat_imp = pd.Series(np.abs(coefs), index=X_train_bal.columns).sort_values(ascending=False)
        print("Top features by (abs) coefficient:\n", feat_imp.head(10))

    return model, X_test, y_test

if __name__ == "__main__":
    # Run with a specific model, e.g. "logistic" or "random_forest"
    train_and_evaluate(model_name="random_forest")
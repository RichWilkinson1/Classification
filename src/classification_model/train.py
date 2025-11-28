import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Import our modules
from classification_model import data_load, features, evaluation

def train_and_evaluate():
    """
    End-to-end training pipeline: load data, preprocess, train model, and evaluate.
    """
    # 1. Load raw data
    df = data_load.load_data()

    # 2. Preprocess features and target
    X, y = features.preprocess_data(df)

    # 3. Split into train and test sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Using stratify=y to maintain the same churn rate in both train and test:contentReference[oaicite:12]{index=12}.

    # 4. (Optional) Handle class imbalance with SMOTE oversampling on training data
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Training set size before SMOTE: {len(X_train)}, after SMOTE: {len(X_train_bal)}")

    # 5. Initialize and train the classifier (using XGBoost in this example)
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_bal, y_train_bal)

    # 6. Evaluate on the test set
    y_pred = model.predict(X_test)
    # Leverage evaluate module to print metrics
    evaluation.evaluate_model(y_test, y_pred)

    # Return the trained model (and test data if needed)
    return model, X_test, y_test

# If this script is run directly, execute the training pipeline
if __name__ == "__main__":
    train_and_evaluate()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    """
    Compute and display evaluation metrics for model predictions.
    """
    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    # Precision, Recall, F1 for positive class (churn=1)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Positive=Churn): {prec:.4f}")
    print(f"Recall (Positive=Churn): {rec:.4f}")
    print(f"F1-Score (Positive=Churn): {f1:.4f}")
    # Classification report for detailed breakdown
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Actual vs Predicted):")
    print(cm)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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

def plot_roc_curve(y_true, y_probs):
    """
    Plot ROC curve and print AUC score.
    """

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    print(f"AUC Score: {auc_score:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
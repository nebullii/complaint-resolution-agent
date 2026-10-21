from sklearn.metrics import classification_report, f1_score

from src.labels import HIGH_RISK


def evaluate_predictions(y_true, y_pred, output_dict=False):
    """Computes macro F1 and prints the full classification report."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)

    print(f"Macro F1: {macro_f1:.4f}")
    if not output_dict:
        print("\nClassification Report:")
        print(report)

    return macro_f1, report


def summarize(y_true, y_pred):
    """Macro F1, accuracy and high-risk recall as a flat dict, ready for MLflow."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": report["accuracy"],
    }
    for c in HIGH_RISK:
        summary[f"recall_{c}"] = report[c]["recall"] if c in report else 0.0
    return summary, report

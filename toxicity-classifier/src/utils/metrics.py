import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, matthews_corrcoef

def compute_classification_metrics(y_true, y_pred, y_probs=None, num_classes=4):
    """
    Computes all essential classification metrics.
    If y_probs is provided, also computes multi-class ROC-AUC (one-vs-rest).
    """
    acc = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)
    
    roc_auc = None
    if y_probs is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except ValueError:
            # Can happen if a class is completely missing in true labels
            roc_auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_per_class": f1_per_class.tolist(),
        "mcc": mcc,
        "roc_auc_macro": roc_auc,
        "confusion_matrix": cm.tolist()
    }

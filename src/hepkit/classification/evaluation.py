import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate various evaluation metrics for binary classification.
    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - y_prob (array-like, optional): Predicted probabilities for positive class.
    Returns:
    - metrics (dict): Dictionary containing the calculated metrics:
        - accuracy (float): Accuracy score.
        - precision (float): Precision score.
        - recall (float): Recall score.
        - f1_score (float): F1 score.
        - roc_auc (float, optional): ROC AUC score (only if y_prob is provided).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def calculate_hep_metrics(y_true, y_pred, y_proba=None, weights=None):
    """Calculate HEP-specific metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "signal_efficiency": np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1),
        "background_rejection": 1 - np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0),
    }

    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
        metrics["significance"] = calculate_significance(y_true, y_proba, weights)

    return metrics


def calculate_significance(y_true, y_proba, weights=None, threshold=0.5):
    """Calculate significance S/sqrt(B) for given threshold"""
    # Implementation here
    pass


def calculate_roc_curve(y_true, y_scores):
    """Compute ROC curve data without plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, thresholds, auc


def cross_validate_model(model, X, y, cv=5, scoring="accuracy"):
    """
    Cross-validates a machine learning model using the specified number of folds and scoring metric.

    Parameters:
        model (object): The machine learning model to be cross-validated.
        X (array-like): The input features.
        y (array-like): The target variable.
        cv (int, optional): The number of folds for cross-validation. Default is 5.
        scoring (str, optional): The scoring metric to evaluate the model performance. Default is 'accuracy'.

    Returns:
        float: The mean score of the model across all folds.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return np.mean(scores)


def stratified_k_fold_cv(model, X, y, n_splits=5, scoring="accuracy"):
    """
    Perform stratified k-fold cross-validation on a given model.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: The feature matrix.
    - y: The target variable.
    - n_splits: The number of folds. Default is 5.
    - scoring: The scoring metric to be used. Default is 'accuracy'.

    Returns:
    - The mean score of the cross-validation.
    """
    skf = StratifiedKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return np.mean(scores)

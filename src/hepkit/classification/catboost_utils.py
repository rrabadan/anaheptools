import catboost
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, Pool


def train_catboost_model(
    model_params,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    weights=None,
    val_weights=None,
    train_pool=None,
    val_pool=None,
    verbose=100,
    plot=True,
):
    """
    Train the CatBoost model on the training dataset.

    Parameters:
    - model (CatBoostClassifier): The CatBoost model to train.
    - X_train (array-like, optional): Feature matrix for training.
    - y_train (array-like, optional): Target vector for training.
    - X_val (array-like, optional): Feature matrix for validation.
    - y_val (array-like, optional): Target vector for validation.
    - weights (array-like, optional): Sample weights for training.
    - val_weights (array-like, optional): Sample weights for validation.
    - train_pool (Pool, optional): Training pool (alternative to X_train, y_train, weights).
    - val_pool (Pool, optional): Validation pool (alternative to X_val, y_val, val_weights).
    - verbose (int, optional): Verbosity level.
    - plot (bool, optional): Whether to plot training progress.

    Returns:
    - model (CatBoostClassifier): Trained CatBoost model.
    """
    # Handle train data
    if train_pool is None:
        if X_train is None or y_train is None:
            raise ValueError("Either train_pool or both X_train and y_train must be provided")
        train_pool = Pool(data=X_train, label=y_train, weight=weights)

    # Handle validation data
    eval_set = None
    if val_pool is not None:
        eval_set = val_pool
    elif X_val is not None and y_val is not None:
        eval_set = Pool(data=X_val, label=y_val, weight=val_weights)

    model = CatBoostClassifier(**model_params)

    model.fit(train_pool, eval_set=eval_set, verbose=verbose, plot=plot)
    return model


def cross_val_catboost_model(
    model_params,
    X=None,
    y=None,
    weights=None,
    pool=None,
    nfolds: int = 5,
    plot: bool = True,
    stratified: bool = False,
):
    """
    Perform cross-validation for evaluating a CatBoost model.
    Args:
        X (array-like, optional): Feature matrix. Required if pool is None.
        y (array-like, optional): Target vector. Required if pool is None.
        weights (array-like, optional): Sample weights. Default is None.
        pool (Pool, optional): The training dataset as a Pool object. If provided, X, y, and weights are ignored.
        params (dict): The parameters for the CatBoost model.
        nfolds (int, optional): The number of folds for cross-validation. Defaults to 5.
        plot (bool, optional): Whether to plot the cross-validation results. Defaults to True.
        stratified (bool, optional): Whether to use stratified sampling for cross-validation. Defaults to False.
    Returns:
        dict: The cross-validation results.
    """

    if pool is None:
        if X is None or y is None:
            raise ValueError("Either pool must be provided, or both X and y must be provided.")
        pool = Pool(data=X, label=y, weight=weights)

    if model_params is None:
        raise ValueError("params must be provided.")

    cv_catb = catboost.cv(
        params=model_params,
        pool=pool,
        fold_count=nfolds,
        shuffle=True,
        partition_random_seed=0,
        plot=plot,
        stratified=stratified,
        verbose=False,
    )

    return cv_catb


def evaluate_catboost_model(model, X_test, y_test, weights=None, plot=True):
    """
    Evaluate a CatBoost model on a test dataset.

    Parameters:
    - model (CatBoostClassifier): The trained CatBoost model.
    - X_test (array-like): Feature matrix for testing.
    - y_test (array-like): Target vector for testing.
    - weights (array-like, optional): Sample weights for testing.
    - plot (bool, optional): Whether to plot the evaluation results.

    Returns:
    - eval_result (dict): Evaluation results including accuracy, logloss, and AUC.
    """
    test_pool = Pool(data=X_test, label=y_test, weight=weights)

    eval_result = model.eval_metrics(
        data=test_pool,
        metrics=["Accuracy", "Logloss", "AUC"],
        ntree_start=0,
        ntree_end=model.tree_count_,
        thread_count=-1,
    )

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_result["iterations"], eval_result["Logloss"], label="Logloss")
        plt.plot(eval_result["iterations"], eval_result["AUC"], label="AUC")
        plt.xlabel("Iterations")
        plt.ylabel("Metric Value")
        plt.title("Model Evaluation Metrics")
        plt.legend()
        plt.show()

    return eval_result


def select_catboost_features(
    model_params,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    weights=None,
    val_weights=None,
    train_pool=None,
    val_pool=None,
    feature_names=None,
    num_features=10,
    steps=1,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    train_final_model=True,
):
    """
    Select features using CatBoost's built-in feature selection.

    Parameters:
    - model_params (dict): CatBoost parameters
    - X_train, y_train: Training data (primary interface)
    - X_val, y_val: Validation data (primary interface)
    - weights, val_weights: Sample weights
    - train_pool, val_pool: Alternative Pool interface
    - feature_names (list): Feature names
    - num_features (int): Number of features to select
    - steps (int): Feature selection steps
    - algorithm: Feature selection algorithm
    - train_final_model (bool): Whether to train final model

    Returns:
    - summary: Feature selection summary
    - model: Trained model (if train_final_model=True)
    """
    # Handle train data
    if train_pool is None:
        if X_train is None or y_train is None:
            raise ValueError("Either train_pool or both X_train and y_train must be provided")
        train_pool = Pool(data=X_train, label=y_train, weight=weights)

    # Handle validation data
    if val_pool is None:
        if X_val is None or y_val is None:
            raise ValueError("Either val_pool or both X_val and y_val must be provided")
        val_pool = Pool(data=X_val, label=y_val, weight=val_weights)

    if model_params is None:
        model_params = {}

    model = CatBoostClassifier(**model_params)

    summary = model.select_features(
        train_pool,
        eval_set=val_pool,
        features_for_select=feature_names,
        num_features_to_select=num_features,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=train_final_model,
        logging_level="Silent",
        plot=True,
    )

    print("Selected features:", summary["selected_features_names"])
    return summary, model


def plot_catboost_learning_curve(model, metric="Logloss", color="green"):
    """
    Plots the learning curve for a given model.
    Parameters:
    - model: The trained model object.
    - metric: The evaluation metric to plot. Valid options are 'Logloss', 'AUC', and 'Accuracy'. Default is 'Logloss'.
    - color: The color of the plotted lines and points. Default is 'green'.
    Returns:
    None
    """

    assert metric in ["Logloss", "AUC", "Accuracy"], "Invalid metric"

    evals_result = model.get_evals_result()
    if "validation" not in evals_result:
        raise ValueError(
            "No validation results found. Make sure the model was trained with a validation set."
        )
    val_metrics = evals_result["validation"]
    n = len(val_metrics[metric])

    # Plot validation curve
    plt.plot(range(n), val_metrics[metric], label="Validation", c=color)

    # Plot training curve if available
    if "learn" in evals_result:
        train_metrics = evals_result["learn"]
        plt.plot(range(n), train_metrics[metric], label="Training", c=color, linestyle="--")
    else:
        print(
            "Warning: 'learn' metrics not available in evaluation results. Skipping training curve plot."
        )

    # Plot best iteration point
    if metric == "AUC":
        best_iter = np.argmax(val_metrics[metric])
    else:
        best_iter = np.argmin(val_metrics[metric])
    plt.scatter(best_iter, val_metrics[metric][best_iter], c=color)

    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.legend()


def get_catboost_shap_values(model, pool):
    """
    Compute SHAP values for a CatBoost model using a Pool object.

    Note:
    - For binary classification and regression, returns SHAP values for features (last column is expected value).
    - For multiclass classification, returns a 3D array (samples, classes, features+1), where the last feature is expected value per class.

    Parameters:
    - model (CatBoost model): The trained CatBoost model.
    - pool (catboost.Pool): The dataset as a CatBoost Pool object.

    Returns:
    - shap_values: The SHAP values (see note above).
    """
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    if shap_values.ndim == 2:
        # Binary classification or regression: remove last column (expected value)
        return shap_values[:, :-1]
    elif shap_values.ndim == 3:
        # Multiclass: remove last column (expected value) for each class
        return shap_values[:, :, :-1]
    else:
        raise ValueError(f"Unexpected shape for SHAP values: {shap_values.shape}")


def plot_catboost_loss_by_eliminated_features(
    summary: dict, show_names: bool = False, figsize: tuple = (10, 6)
):
    """
    Plots the loss value against the count of removed features.
    Parameters:
    - summary (dict): A dictionary containing the summary information.
    - show_names (bool): Flag indicating whether to show feature names on the plot. Default is False.
    - figsize (tuple): A tuple specifying the figure size. Default is (10, 6).
    Returns:
    None
    """

    loss_graph = summary["loss_graph"]

    loss_values = loss_graph["loss_values"]
    removed_features_count = loss_graph["removed_features_count"]

    # Create the plot
    plt.figure(figsize=figsize)
    plt.scatter(removed_features_count, loss_values, color="blue")
    plt.plot(removed_features_count, loss_values, color="blue", linestyle="--")
    plt.xlabel("Removed Features Count")
    plt.ylabel("Loss Value")
    plt.title("Loss Value vs Removed Features Count")

    if show_names:
        feature_names = summary["eliminated_features_names"]

        # Annotate each point with the feature name
        for i, feature_name in enumerate(feature_names):
            plt.annotate(
                feature_name,
                (removed_features_count[i + 1], loss_values[i + 1]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )


def plot_catboost_feature_importance(model, feature_names, max_num_features=20):
    """
    Plot the feature importance of a model.

    Parameters:
    - model: The trained model.
    - feature_names: List of feature names.
    - max_num_features: Maximum number of features to display (default: 20).

    Returns:
    - None
    """
    feature_importances = model.get_feature_importance()
    indices = np.argsort(feature_importances)[-max_num_features:]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), feature_importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")


def create_catboost_pools(
    X_train, y_train, X_val=None, y_val=None, train_weights=None, val_weights=None
):
    """Create CatBoost Pool objects from arrays."""
    train_pool = Pool(data=X_train, label=y_train, weight=train_weights)
    val_pool = None
    if X_val is not None and y_val is not None:
        val_pool = Pool(data=X_val, label=y_val, weight=val_weights)
    return train_pool, val_pool


def save_catboost_model(model, path, train_pool=None):
    """
    Parameters:
    - model (CatBoostClassifier): The trained CatBoost model.
    - path (str): Path to save the model.

    Returns:
    - None
    """
    model.save_model(path, format="json", pool=train_pool)


def load_catboost_model(path, format="json"):
    """
    Load a trained model from disk.

    Parameters:
    - model_path (str): Path to the saved model.

    Returns:
    - model (CatBoostClassifier): Loaded CatBoost model.
    """
    model = CatBoostClassifier()
    model.load_model(path, format=format)
    return model

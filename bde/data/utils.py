"""Validation helpers shared across estimators and data utilities."""

import warnings

import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from bde.task import TaskType


def validate_fit_data(estimator, X, y):
    """Run sklearn-style checks for training data and update estimator metadata."""
    y_array = np.asarray(y)
    if y_array.ndim > 1 and y_array.shape[1] == 1:
        warnings.warn(
            DataConversionWarning(
                "A column-vector y was passed when a 1d array was expected."
            ),
            stacklevel=3,
        )
        y_array = y_array.reshape(-1)

    feature_names = getattr(X, "columns", None)
    X_checked, y_checked = check_X_y(
        X,
        y_array,
        accept_sparse=False,
        ensure_2d=True,
        ensure_min_samples=1,
        force_all_finite=True,
        y_numeric=(estimator.task == TaskType.REGRESSION),  # only Treu for regression
    )

    if feature_names is not None:
        estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        delattr(estimator, "feature_names_in_")

    estimator.n_features_in_ = X_checked.shape[1]

    if estimator.task == TaskType.CLASSIFICATION:
        check_classification_targets(y_checked)
        estimator.classes_ = np.unique(y_checked)

    if estimator.task == TaskType.REGRESSION:
        y_checked = y_checked[:, None]  # restore column form for the network

    return X_checked, y_checked


def validate_predict_data(estimator, X):
    """Validate inputs for predict-like methods and enforce feature metadata."""

    check_is_fitted(estimator, attributes=["n_features_in_"])

    feature_names = getattr(X, "columns", None)
    X_checked = check_array(
        X,
        accept_sparse=False,
        ensure_2d=True,
        ensure_min_samples=1,
        force_all_finite=True,
    )

    if X_checked.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_checked.shape[1]} features, but {estimator.__class__.__name__} was fitted with {estimator.n_features_in_}."
        )

    if hasattr(estimator, "feature_names_in_"):
        if feature_names is None:
            raise ValueError(
                "X has no feature names, but this estimator was fitted with feature names."
            )
        if list(feature_names) != list(estimator.feature_names_in_):
            raise ValueError("Feature names of X do not match those seen during fit.")

    return X_checked

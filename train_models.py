"""Train and persist the five classifiers required by ML Assignment 2.

Dataset: Breast Cancer Wisconsin (Diagnostic), distributed with scikit-learn
and originally published in the UCI Machine Learning Repository.

The script uses one stratified hold-out split. It exports only the hold-out set
as test_data.csv. All preprocessing is embedded in each saved Pipeline, so the
Streamlit application never retrains on uploaded test data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def clean_feature_name(name: str) -> str:
    """Convert the source feature label into a stable CSV/Python name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    features.columns = [clean_feature_name(c) for c in features.columns]

    # The scikit-learn source uses 0=malignant and 1=benign. For intuitive
    # binary metrics, this project makes malignant the positive class (1).
    target = (dataset.target == 0).astype(int)
    target.name = TARGET_COLUMN

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "UCI Machine Learning Repository via scikit-learn",
        "dataset_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "total_instances": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "target_column": TARGET_COLUMN,
        "positive_class": "Malignant",
        "class_mapping": {"0": "Benign", "1": "Malignant"},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_names": features.columns.tolist(),
    }
    return features, target, metadata


def build_models() -> dict[str, Pipeline]:
    scaled_preprocessing = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    unscaled_preprocessing = [("imputer", SimpleImputer(strategy="median"))]

    return {
        "Logistic Regression": Pipeline(
            scaled_preprocessing
            + [("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))]
        ),
        "Decision Tree": Pipeline(
            unscaled_preprocessing
            + [("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE))]
        ),
        "K-Nearest Neighbor": Pipeline(
            scaled_preprocessing
            + [("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))]
        ),
        "Gaussian Naive Bayes": Pipeline(
            unscaled_preprocessing + [("classifier", GaussianNB())]
        ),
        "Support Vector Machine": Pipeline(
            scaled_preprocessing
            + [("classifier", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))]
        ),
    }


def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y, metadata = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    model_files: dict[str, str] = {}
    metric_rows: list[dict[str, float | str]] = []

    for display_name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = pipeline.predict_proba(X_test)[:, 1]
        safe_name = clean_feature_name(display_name)
        file_name = f"{safe_name}.joblib"
        joblib.dump(pipeline, MODEL_DIR / file_name)
        model_files[display_name] = file_name
        metric_rows.append({"ML Model Name": display_name, **evaluate(y_test, predictions, scores)})

    metrics = pd.DataFrame(metric_rows).sort_values("MCC", ascending=False).reset_index(drop=True)
    metrics.to_csv(ROOT / "metrics.csv", index=False, float_format="%.6f")
    metadata.update(
        {
            "training_instances": int(X_train.shape[0]),
            "test_instances": int(X_test.shape[0]),
            "model_files": model_files,
            "metric_definition": "Binary metrics use Malignant as the positive class; AUC uses malignant probability.",
        }
    )
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {len(model_files)} models and {len(test_data)} test rows in {ROOT}")


if __name__ == "__main__":
    main()

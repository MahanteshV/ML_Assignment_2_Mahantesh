"""Streamlit evaluator for the five saved breast-cancer classifiers.

The app intentionally performs inference only. Training is isolated in
train_models.py, and every saved object is a complete preprocessing + model
Pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "diagnosis"

st.set_page_config(
    page_title="OncoClass | Model Evaluation Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#132238; --blue:#1d4ed8; --cyan:#06b6d4; --paper:#f7fafc; }
    .stApp { background: linear-gradient(135deg, #f8fbff 0%, #eef7fb 55%, #f8f5ff 100%); }
    .hero { padding: 1.35rem 1.55rem; border-radius: 18px; color: white;
            background: linear-gradient(120deg, #14213d 0%, #1d4ed8 55%, #0891b2 100%);
            box-shadow: 0 12px 28px rgba(29,78,216,.18); margin-bottom: 1rem; }
    .hero h1 { margin:0; font-size:2.15rem; letter-spacing:-.03em; }
    .hero p { margin:.35rem 0 0; color:#dbeafe; }
    .info-card { background:rgba(255,255,255,.85); border:1px solid #dbeafe;
                 border-radius:14px; padding:1rem 1.1rem; min-height:112px; }
    .status-ok { color:#047857; font-weight:700; }
    .small-note { color:#526176; font-size:.88rem; }
    .report-table { width:100%; border-collapse:collapse; background:rgba(255,255,255,.9); font-size:.88rem; }
    .report-table th { background:#e8f0ff; color:#14213d; text-align:left; }
    .report-table th, .report-table td { padding:.48rem .58rem; border:1px solid #dce8f5; }
    .report-table tbody tr:nth-child(even) { background:#f8fbff; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.9); border:1px solid #dce8f5;
                                  padding:.7rem; border-radius:14px; }
    section[data-testid="stSidebar"] { background:#f4f8fd; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metadata() -> dict:
    return json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_reference_metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "metrics.csv")


@st.cache_data
def load_bundled_test_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "test_data.csv")


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(ROOT / "model" / file_name)


def validate_input(data: pd.DataFrame, required_features: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in required_features if name not in data.columns]
    return not missing, missing


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


metadata = load_metadata()
reference_metrics = load_reference_metrics()
model_names = list(metadata["model_files"].keys())

st.markdown(
    """
    <div class="hero">
      <h1>OncoClass Evaluation Studio</h1>
      <p>Five saved classifiers. One reproducible hold-out set. Zero retraining during evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Evaluation controls")
    selected_model = st.selectbox("Select a trained model", model_names)
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"], help="Include all 30 feature columns and the diagnosis target.")
    use_demo = st.toggle("Use bundled hold-out test data", value=uploaded_file is None)
    st.divider()
    st.caption("Target convention")
    st.write("**1 = Malignant** · **0 = Benign**")
    with open(ROOT / "test_data.csv", "rb") as sample_file:
        st.download_button(
            "Download valid sample CSV",
            sample_file,
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

if uploaded_file is not None:
    try:
        evaluation_data = pd.read_csv(uploaded_file)
        data_source = f"Uploaded file: {uploaded_file.name}"
    except Exception as exc:
        st.error(f"The uploaded CSV could not be read: {exc}")
        st.stop()
elif use_demo:
    evaluation_data = load_bundled_test_data().copy()
    data_source = "Bundled stratified hold-out set"
else:
    st.info("Upload a CSV or enable the bundled hold-out test data to begin.")
    st.stop()

required_features = metadata["feature_names"]
valid, missing_columns = validate_input(evaluation_data, required_features)
if not valid:
    st.error(f"Missing {len(missing_columns)} required feature column(s): {', '.join(missing_columns)}")
    st.stop()
if TARGET_COLUMN not in evaluation_data.columns:
    st.error(f"The target column '{TARGET_COLUMN}' is required for evaluation metrics.")
    st.stop()

numeric_features = evaluation_data[required_features].apply(pd.to_numeric, errors="coerce")
non_numeric_count = int(numeric_features.isna().sum().sum() - evaluation_data[required_features].isna().sum().sum())
if non_numeric_count:
    st.warning(f"Converted {non_numeric_count} non-numeric feature value(s) to missing values; the saved pipeline will impute them.")

try:
    y_true = pd.to_numeric(evaluation_data[TARGET_COLUMN], errors="raise").astype(int)
except Exception:
    st.error("The diagnosis target must contain numeric 0/1 values.")
    st.stop()
if not set(y_true.unique()).issubset({0, 1}):
    st.error("The diagnosis target may contain only 0 (Benign) and 1 (Malignant).")
    st.stop()
if y_true.nunique() < 2:
    st.error("AUC requires both target classes. Upload test data containing benign and malignant rows.")
    st.stop()

model = load_model(metadata["model_files"][selected_model])
y_pred = model.predict(numeric_features)
y_score = model.predict_proba(numeric_features)[:, 1]
metrics = calculate_metrics(y_true, y_pred, y_score)

summary_cols = st.columns(3)
with summary_cols[0]:
    st.markdown(f'<div class="info-card"><b>Selected model</b><br>{selected_model}<br><span class="status-ok">Ready for inference</span></div>', unsafe_allow_html=True)
with summary_cols[1]:
    st.markdown(f'<div class="info-card"><b>Evaluation data</b><br>{data_source}<br><span class="small-note">{len(evaluation_data)} rows · {len(required_features)} features</span></div>', unsafe_allow_html=True)
with summary_cols[2]:
    malignant_rate = y_true.mean() * 100
    st.markdown(f'<div class="info-card"><b>Class profile</b><br>{int(y_true.sum())} malignant / {int((1-y_true).sum())} benign<br><span class="small-note">{malignant_rate:.1f}% positive class</span></div>', unsafe_allow_html=True)

st.subheader("Evaluation snapshot")
metric_columns = st.columns(6)
for column, (name, value) in zip(metric_columns, metrics.items()):
    column.metric(name, f"{value:.3f}")

overview_tab, diagnostics_tab, comparison_tab, data_tab = st.tabs(
    ["Model diagnostics", "Classification details", "Five-model benchmark", "Data preview"]
)

with overview_tab:
    left, right = st.columns([1, 1])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with left:
        st.markdown("#### Confusion matrix")
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette("#1d4ed8", as_cmap=True),
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with right:
        st.markdown("#### ROC curve")
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(false_positive_rate, true_positive_rate, color="#1d4ed8", linewidth=2.5, label=f"{selected_model} (AUC={metrics['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Random baseline")
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

with diagnostics_tab:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T
    st.markdown(report_frame.round(3).to_html(classes="report-table"), unsafe_allow_html=True)
    tn, fp, fn, tp = matrix.ravel()
    st.markdown(
        f"**Error reading:** the model produced **{fp} false positive(s)** and **{fn} false negative(s)**. "
        "In this educational dataset, false negatives are malignant cases predicted as benign."
    )
    output = evaluation_data.copy()
    output["predicted_diagnosis"] = y_pred
    output["predicted_label"] = np.where(y_pred == 1, "Malignant", "Benign")
    output["malignant_probability"] = y_score
    output["prediction_correct"] = y_pred == y_true.to_numpy()
    st.download_button(
        "Download row-level predictions",
        output.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

with comparison_tab:
    st.caption("Reference results from the fixed stratified 20% hold-out set (random_state=42).")
    display_metrics = reference_metrics.copy()
    numeric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    formatted_metrics = display_metrics.copy()
    for column in numeric_columns:
        formatted_metrics[column] = formatted_metrics[column].map(lambda value: f"{value:.3f}")
    st.markdown(formatted_metrics.to_html(index=False, classes="report-table"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(numeric_columns))
    width = 0.15
    palette = sns.color_palette("crest", n_colors=len(display_metrics))
    for model_index, (_, row) in enumerate(display_metrics.iterrows()):
        offset = (model_index - (len(display_metrics) - 1) / 2) * width
        ax.bar(x + offset, row[numeric_columns].astype(float), width, label=row["ML Model Name"], color=palette[model_index])
    ax.set_xticks(x, numeric_columns)
    ax.set_ylim(0.70, 1.01)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

with data_tab:
    st.markdown(evaluation_data.head(25).round(4).to_html(index=False, classes="report-table"), unsafe_allow_html=True)
    st.caption("Preview is limited to 25 rows. Models receive features in the saved training order.")

st.divider()
st.caption("Educational decision-support demo only. Not a diagnostic medical device. Models are loaded from disk and are not fitted on uploaded data.")

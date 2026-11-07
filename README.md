# OncoClass Evaluation Studio

A reproducible machine-learning classification project for **BITS WILP M.Tech (AIML) - Machine Learning Assignment 2**. The project trains five required classifiers on the same dataset, saves complete preprocessing/model pipelines, and evaluates them in a custom Streamlit interface without retraining on uploaded test data.

> **Submission fields to update**
>
> - GitHub repository: `PASTE_GITHUB_REPOSITORY_URL_HERE`
> - Live Streamlit application: `PASTE_STREAMLIT_APP_URL_HERE`

## a. Problem statement

The objective is to classify a breast tumour as **malignant (1)** or **benign (0)** from diagnostic measurements computed from digitized fine-needle aspirate images. Five classification algorithms are trained on one fixed training split and compared on one untouched, stratified hold-out set. The deployed application accepts a test CSV, loads a selected saved pipeline, generates predictions, and displays evaluation metrics, a confusion matrix, an ROC curve, and a classification report. It does not fit or update a model using uploaded data.

This is an educational model-comparison exercise and **not a medical diagnostic device**.

## b. Dataset description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Original source:** UCI Machine Learning Repository
- **Dataset page:** <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
- **Access used in this project:** `sklearn.datasets.load_breast_cancer`
- **Problem type:** Binary classification
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target:** `diagnosis`
- **Project encoding:** `0 = Benign`, `1 = Malignant`
- **Split:** Stratified 80% training (455 rows) / 20% test (114 rows)
- **Random state:** 42

The 30 predictors summarize radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. For each characteristic, mean, standard-error, and worst/largest measurements are represented.

## c. GitHub Repository Link

**Repository:** `PASTE_GITHUB_REPOSITORY_URL_HERE`

Required repository contents are included:

```text
ML_Assignment_2_Mahantesh/
|-- app.py
|-- train_models.py
|-- smoke_test.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- metrics.csv
|-- metadata.json
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- k_nearest_neighbor.joblib
|   |-- gaussian_naive_bayes.joblib
|   `-- support_vector_machine.joblib
`-- .gitignore
```

## d. Models used and evaluation results

All binary precision, recall, F1, and AUC calculations use **Malignant (1)** as the positive class. Values below are from the bundled 114-row hold-out set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | **0.9954** | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9035 | 0.8980 | 0.8974 | 0.8333 | 0.8642 | 0.7908 |
| K-Nearest Neighbor | 0.9561 | 0.9835 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9825** | 0.9950 | **1.0000** | **0.9524** | **0.9756** | **0.9626** |

### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong and well-balanced baseline. It achieved the highest AUC (0.9954), with high malignant-class precision and recall, showing that a linear boundary performs very well after scaling. |
| Decision Tree | Most interpretable model, but weakest result on the hold-out set. Its lower AUC (0.8980) and MCC (0.7908) suggest that one bounded-depth tree did not capture the smooth decision boundary as effectively as the scaled models. |
| K-Nearest Neighbor | Good local-neighbour performance after standardization. Recall (0.9048) was lower than Logistic Regression and SVM, so it missed more malignant cases on this split. |
| Gaussian Naive Bayes | Produced no false-positive malignant predictions (precision 1.0000), but recall of 0.8333 indicates several malignant cases were predicted as benign. Its high AUC shows useful ranking despite the default decision threshold. |
| Support Vector Machine | Best overall hold-out classifier: highest accuracy, F1, and MCC, tied-best recall, and perfect malignant precision. The scaled RBF boundary separated the classes effectively. |
| **Overall winner** | **Support Vector Machine**, because MCC (0.9626) summarizes the strongest balanced binary performance and it also led accuracy and F1. Logistic Regression remains a close, more interpretable alternative and had a slightly higher AUC. |

The test set is small (114 rows), so differences of a few cases can noticeably change the metrics. A larger external validation set or repeated cross-validation would be appropriate before drawing clinical conclusions.

## Streamlit application features

- CSV upload limited to test/evaluation data
- Dropdown for all five saved models
- Input-schema and binary-target validation
- Six required metrics: Accuracy, AUC, Precision, Recall, F1, and MCC
- Confusion matrix and ROC curve
- Per-class classification report
- Five-model benchmark table and chart
- Row-level prediction download
- Bundled valid sample CSV download
- No retraining on uploaded data

## Run locally

Use Python 3.11 or a compatible version:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Reproduce training artifacts

```bash
python train_models.py
```

This command recreates `test_data.csv`, `metrics.csv`, `metadata.json`, and all five `.joblib` pipelines. It uses a deterministic stratified split with `random_state=42`.

## Test CSV contract

The uploaded CSV must contain:

1. All 30 feature columns listed in `metadata.json`.
2. The target column `diagnosis`.
3. Target values restricted to `0` (Benign) and `1` (Malignant).
4. At least one row from each target class so AUC is defined.

Extra columns are allowed and ignored during inference. Missing numeric values are median-imputed inside each saved pipeline.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and commit this project with meaningful, incremental commits.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Choose **New app**.
4. Select the repository and the `main` branch.
5. Set the application file to `app.py`.
6. Select a compatible Python version if prompted.
7. Deploy and wait for the health check to pass.
8. Test the live app with `test_data.csv` and update both link placeholders in this README and the submission PDF.

## Reproducibility and academic-integrity notes

- The repository should retain the learner's own commit history, comments, and explanations.
- Review and understand every file before submission; adapt the analysis in your own words.
- Do not claim the generated placeholder URLs or a local screenshot as deployed/BITS Lab evidence.
- Run the project in the BITS Virtual Lab and capture one screenshot showing the lab environment and successful execution.

## Citation

Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5DW2B>

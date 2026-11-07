"""Fast deployment smoke test: load every artifact and run inference."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
data = pd.read_csv(ROOT / "test_data.csv")
X = data[metadata["feature_names"]]
y = data[metadata["target_column"]]

assert len(data) >= 1
assert y.nunique() == 2
for name, file_name in metadata["model_files"].items():
    model = joblib.load(ROOT / "model" / file_name)
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    assert len(predictions) == len(data)
    auc = roc_auc_score(y, scores)
    print(f"PASS | {name:24s} | rows={len(predictions):3d} | AUC={auc:.4f}")

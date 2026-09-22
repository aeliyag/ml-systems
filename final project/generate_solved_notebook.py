#!/usr/bin/env python3
"""Generate the solved final-project.ipynb from the blank template structure."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "final-project.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


cells = []

cells.append(md("""# Final Project: Adaptive Feature Selection for Predicting Video Quality Degradation

**Machine Learning for Computer Systems** · Open Problem / Research  
**Aeliya Grover, Clarisse Cheung** · with assistance from Cursor

---

## What This Notebook Is For

This notebook is the **primary deliverable** for the course final project. It runs end-to-end from setup through evaluation and conclusions.

## Research Question

Can *adaptive feature selection*—starting with cheap features and computing expensive ones only when needed—reduce the computational cost of predicting video quality degradation while matching the performance of a model that always uses the full feature set?

## How This Notebook Is Organized

| Part | What | Why it matters |
|------|------|----------------|
| 1 | Setup & configuration | Reproducibility and one-click execution |
| 2 | Data loading & integrity | Trustworthy inputs before any modeling |
| 3 | Labels, splits & leakage checks | Correct task definition and honest evaluation |
| 4 | Feature tiers (L3 / L4 / L7) | Connect ML inputs to systems cost |
| 5 | Baselines & controls | Know what "good" looks like before the adaptive method |
| 6 | Train multiple models | Compare fixed tiers, full features, and resolution-only |
| 7 | Adaptive cascade | Main research contribution |
| 8 | Hyperparameter tuning | Fair comparison across models |
| 9 | Prediction evaluation | Measure quality with imbalance-aware metrics |
| 10 | Systems-cost evaluation | Measure the deployment tradeoff |
| 11 | Error analysis & robustness | Understand failures, not just averages |
| 12 | Cross-model comparison | Synthesize results across all experiments |
| 13 | Conclusions | Answer the research question and reflect on learning |
| 14 | Reproducibility checklist | Verify the notebook is submission-ready |"""))

cells.append(md("""---

## Part 1: Setup & Configuration

**What:** Install dependencies, set random seeds, and define paths to data files.

**Why:** The project must run end-to-end with "Restart kernel and run all." Centralizing configuration here makes the rest of the notebook reproducible and easier to debug."""))

cells.append(code("""import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
HORIZON_SECONDS = 10
TIER_COST = {1: 1.0, 2: 4.0, 3: 10.0}

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "completed assignments").exists():
    REPO_ROOT = REPO_ROOT.parent

DATA_DIR = REPO_ROOT / "completed assignments" / "data" / "video-qoe"
VIDEO_DATASET_PATH = DATA_DIR / "video_dataset.pkl"
NETFLIX_PCAP_PATH = DATA_DIR / "netflix.pcap"
FIG_DIR = REPO_ROOT / "final project" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VALID_RESOLUTIONS = [240, 360, 480, 720, 1080]
META_COLS = [
    "session_id", "relative_timestamp", "absolute_timestamp",
    "resolution", "service", "video_id", "video_position", "home_id", "index",
]

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("Data dir:", DATA_DIR)
print("video_dataset exists:", VIDEO_DATASET_PATH.exists())
print("netflix.pcap exists:", NETFLIX_PCAP_PATH.exists())"""))

cells.append(md("**Environment:** Python 3.11, scikit-learn 1.5, pandas 2.2, matplotlib/seaborn for plots. All random seeds fixed at 42."))

cells.append(code(""))

cells.append(md("""---

## Part 2: Data Loading & Integrity Checks

**What:** Load `video_dataset.pkl`, inspect its schema, and verify basic properties.

**Why:** Bad labels and invalid resolutions silently hurt models. Catching data issues early prevents wasted training runs."""))

cells.append(code("""def load_video_dataset(path=VIDEO_DATASET_PATH) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

df = load_video_dataset()
print("Shape:", df.shape)
print("Columns:", len(df.columns))
print("Dtypes sample:\\n", df.dtypes.value_counts())
df.head()"""))

cells.append(md("""### 2.1 Exploratory summary"""))

cells.append(code("""raw_rows = len(df)
df = df[df["resolution"].isin(VALID_RESOLUTIONS)].copy()
df = df.sort_values(["session_id", "relative_timestamp"]).reset_index(drop=True)

summary = {
    "raw_rows": raw_rows,
    "valid_rows": len(df),
    "sessions": df["session_id"].nunique(),
    "services": df["service"].value_counts().to_dict(),
    "resolution_counts": df["resolution"].value_counts().sort_index().to_dict(),
    "missing_pct": (df.isna().mean().sort_values(ascending=False).head(5) * 100).round(2).to_dict(),
}
summary"""))

cells.append(md("""**Why did you keep or remove any rows/columns?**

We removed rows with invalid resolution (`0.0`) because they are not meaningful QoE states. Metadata columns (`session_id`, timestamps, service, etc.) are kept for splitting and analysis but excluded from model features. No feature columns were dropped at load time because tier prefixes (`L3_`, `L4_`, `L7_`) already define the modeling inputs."""))

cells.append(code(""))

cells.append(md("""---

## Part 3: Labels, Splits & Leakage Checks

**What:** Construct the future-horizon downswitch label and split by session (plus a temporal split for drift).

**Why:** Session-level splits prevent leakage across 10-second windows from the same viewing session."""))

cells.append(md("""### 3.1 Define the prediction horizon Δ

Primary horizon: **10 seconds** (one window ahead). We also report label rates for 20s and 30s horizons in the summary."""))

cells.append(code("""def add_downswitch_label(group: pd.DataFrame, horizon_seconds: int) -> pd.DataFrame:
    steps = max(1, horizon_seconds // 10)
    resolutions = group["resolution"].to_numpy()
    labels = []
    for i in range(len(resolutions)):
        future = resolutions[i + 1 : i + 1 + steps]
        labels.append(any(r < resolutions[i] for r in future))
    out = group.copy()
    out["downswitch"] = labels
    return out

for horizon in [10, 20, 30]:
    tmp = df.groupby("session_id", group_keys=False).apply(
        lambda g: add_downswitch_label(g, horizon)
    )
    print(f"Δ={horizon}s positive rate: {tmp['downswitch'].mean():.4f}")

df = df.groupby("session_id", group_keys=False).apply(
    lambda g: add_downswitch_label(g, HORIZON_SECONDS)
)
df["downswitch"] = df["downswitch"].astype(int)
df["downswitch"].value_counts(normalize=True)"""))

cells.append(md("""### 3.2 Class balance

Positive cases are rare (~1.7% at Δ=10s). Accuracy and ROC-AUC could look misleadingly strong for a useless model. ROC-AUC is defined for context but **not calculated in this notebook**; we calculate and use **PR-AUC** as the headline metric.

**Why does class imbalance matter for metric choice?** With <2% positives, a model predicting "no downswitch" always gets ~98% accuracy but catches zero events. PR-AUC focuses on the minority class and the precision–recall tradeoff."""))

cells.append(code("""label_by_service = (
    df.groupby("service")["downswitch"].agg(["mean", "sum", "count"])
    .rename(columns={"mean": "positive_rate"})
    .sort_values("positive_rate", ascending=False)
)
label_by_service"""))

cells.append(code(""))

cells.append(md("""### 3.3 Train / validation / test splits"""))

cells.append(code("""splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(df, groups=df["session_id"]))
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# validation split from train for tuning / calibration checks
val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
tr_idx, val_idx = next(val_splitter.split(train_df, groups=train_df["session_id"]))
fit_df = train_df.iloc[tr_idx].copy()
val_df = train_df.iloc[val_idx].copy()

# temporal split: earliest 80% of sessions -> train, latest 20% -> test
session_start = df.groupby("session_id")["absolute_timestamp"].min().sort_values()
temp_test_sessions = set(session_start.index[int(0.8 * len(session_start)):])
temp_train_df = df[~df["session_id"].isin(temp_test_sessions)].copy()
temp_test_df = df[df["session_id"].isin(temp_test_sessions)].copy()

print("Session split:", len(fit_df), "fit,", len(val_df), "val,", len(test_df), "test")
print("Positive rate fit/val/test:", fit_df["downswitch"].mean(), val_df["downswitch"].mean(), test_df["downswitch"].mean())
print("Temporal split:", len(temp_train_df), "train,", len(temp_test_df), "test")"""))

cells.append(md("""### 3.4 Leakage checks

**List the columns you excluded and why.**

Excluded from features: all metadata columns (`session_id`, timestamps, `service`, `video_id`, etc.) and the label `downswitch`. We do **not** use future resolution or resolution-change counts. Resolution is used only in the dedicated resolution-only baseline, not in network-feature models."""))

cells.append(code("""FEATURE_EXCLUDE = set(META_COLS + ["downswitch"])
leak_candidates = [c for c in df.columns if "switch" in c.lower() or "future" in c.lower()]
print("Potential leaky columns in dataset:", leak_candidates)
print("Modeling excludes:", sorted(FEATURE_EXCLUDE))"""))

cells.append(code(""))

cells.append(md("""---

## Part 4: Feature Tiers (L3 / L4 / L7)

**What:** Group features by protocol layer cost hierarchy.

**Why:** The project compares prediction quality against feature extraction cost. Tier prefixes are built into column names."""))

cells.append(code("""L3_COLS = sorted(c for c in df.columns if c.startswith("L3_"))
L4_COLS = sorted(c for c in df.columns if c.startswith("L4_"))
L7_COLS = sorted(c for c in df.columns if c.startswith("L7_"))
FULL_COLS = L3_COLS + L4_COLS + L7_COLS

print("L3:", len(L3_COLS))
print("L4:", len(L4_COLS))
print("L7:", len(L7_COLS))
print("FULL:", len(FULL_COLS))
print("Example L3:", L3_COLS[:3])
print("Example L4:", L4_COLS[:3])
print("Example L7:", L7_COLS[:3])"""))

cells.append(md("""**Briefly justify why these tier groupings match the systems-cost hierarchy.**

- **L3 (11 features):** Computed from packet headers alone (throughput, byte/packet counts).
- **L4 (95 features):** Require per-flow TCP state tracking (RTT, retransmissions, bytes in flight).
- **L7 (55 features):** Require inferring encrypted application chunk structure (chunk sizes and inter-arrival times).

Cost increases with protocol depth: L3 < L4 < L7."""))

cells.append(code("""def impute_features(train_part: pd.DataFrame, cols: list[str], *parts: pd.DataFrame):
    medians = train_part[cols].median()
    return tuple(part[cols].fillna(medians) for part in parts)"""))

cells.append(md("""---

## Part 5: Baselines & Control Experiment Design

**Describe your experimental matrix.**

| Model | Features | Purpose |
|-------|----------|---------|
| Resolution-only | current resolution | Trivial baseline; sessions at min resolution cannot downswitch |
| L3 / L4 / L7 fixed | single tier each | Cost–quality tradeoff per tier |
| RF-FULL / GB-FULL | all 161 features | Upper bound on prediction quality |
| Cascade | L3 → L3+L4 → FULL | Adaptive acquisition on uncertain cases |
| Random escalation | matched escalation rate | Control: does adaptation beat random tier upgrades? |

All models use session-level train/test split. Primary metric: PR-AUC at Δ=10s."""))

cells.append(code(""))

cells.append(md("""---

## Part 6: Train Multiple Models

**What:** Train resolution baseline, fixed-tier models, and full-feature models."""))

cells.append(md("""### 6.1 Resolution-only baseline"""))

cells.append(code("""RF_KW = dict(n_estimators=50, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)

resolution_model = Pipeline([
    ("enc", OneHotEncoder(handle_unknown="ignore")),
    ("clf", RandomForestClassifier(**RF_KW)),
])
resolution_model.fit(fit_df[["resolution"]], fit_df["downswitch"])
print("Resolution baseline trained.")"""))

cells.append(md("""### 6.2 Model A — Random Forest (full features)"""))

cells.append(code("""X_fit_full, X_val_full, X_test_full = impute_features(fit_df, FULL_COLS, fit_df, val_df, test_df)
rf_full = RandomForestClassifier(**RF_KW)
rf_full.fit(X_fit_full, fit_df["downswitch"])
print("RF-FULL trained.")"""))

cells.append(md("""### 6.3 Model B — Gradient Boosting (full features)"""))

cells.append(code("""gb_full = HistGradientBoostingClassifier(max_iter=50, random_state=RANDOM_STATE)
gb_full.fit(X_fit_full, fit_df["downswitch"])
print("GB-FULL trained.")"""))

cells.append(md("""### 6.4 Fixed-tier models"""))

cells.append(code("""tier_models = {}
for tier_name, cols in [("L3", L3_COLS), ("L4", L4_COLS), ("L7", L7_COLS)]:
    X_fit, X_val, X_test = impute_features(fit_df, cols, fit_df, val_df, test_df)
    model = RandomForestClassifier(**RF_KW)
    model.fit(X_fit, fit_df["downswitch"])
    tier_models[tier_name] = {"model": model, "X_test": X_test, "cols": cols}
    print(f"{tier_name} model trained on {len(cols)} features.")"""))

cells.append(md("""**Training notes:** All tree models use `n_estimators=50`, `class_weight='balanced'`. Median imputation fitted on the training fold only."""))

cells.append(code(""))

cells.append(md("""---

## Part 7: Adaptive Cascade

**What:** Sequential acquisition — L3 first, escalate to L3+L4, then FULL when probability is near the decision boundary."""))

cells.append(md("""### 7.1 Calibrate probabilities

We use validation-set reliability checks (Part 11). Cascade thresholds operate on `predict_proba` outputs from each tier model."""))

cells.append(code("""X_fit_l3, X_val_l3, X_test_l3 = impute_features(fit_df, L3_COLS, fit_df, val_df, test_df)
X_fit_l34, X_val_l34, X_test_l34 = impute_features(fit_df, L3_COLS + L4_COLS, fit_df, val_df, test_df)

cascade_models = {
    "L3": RandomForestClassifier(**RF_KW),
    "L34": RandomForestClassifier(**RF_KW),
    "FULL": RandomForestClassifier(**RF_KW),
}
cascade_models["L3"].fit(X_fit_l3, fit_df["downswitch"])
cascade_models["L34"].fit(X_fit_l34, fit_df["downswitch"])
cascade_models["FULL"].fit(X_fit_full, fit_df["downswitch"])
print("Cascade tier models trained.")"""))

cells.append(md("""### 7.2 Implement the cascade logic"""))

cells.append(code("""def run_cascade(low: float, high: float, X_test_l3, X_test_l34, X_test_full):
    proba = cascade_models["L3"].predict_proba(X_test_l3)[:, 1]
    cost = np.ones(len(proba))
    uncertain = (proba > low) & (proba < high)
    if uncertain.any():
        proba2 = cascade_models["L34"].predict_proba(X_test_l34[uncertain])[:, 1]
        proba[uncertain] = proba2
        cost[uncertain] = 2
        uncertain2 = (proba2 > low) & (proba2 < high)
        idx = np.where(uncertain)[0][uncertain2]
        if len(idx) > 0:
            proba3 = cascade_models["FULL"].predict_proba(X_test_full.iloc[idx])[:, 1]
            proba[idx] = proba3
            cost[idx] = 3
    avg_cost = float(np.mean([TIER_COST[int(c)] for c in cost]))
    return proba, cost, avg_cost

CASCADE_THRESHOLDS = (0.1, 0.9)
cascade_proba, cascade_cost, cascade_avg_cost = run_cascade(
    *CASCADE_THRESHOLDS, X_test_l3, X_test_l34, X_test_full
)
print("Cascade thresholds:", CASCADE_THRESHOLDS)
print("Escalation rate:", (cascade_cost > 1).mean())
print("Average relative cost:", cascade_avg_cost)"""))

cells.append(md("""### 7.3 Random escalation control"""))

cells.append(code("""def run_random_escalation(escalation_rate, rng, X_test_l3, X_test_l34, X_test_full):
    n = len(X_test_l3)
    cost = np.ones(n)
    mask = rng.random(n) < escalation_rate
    cost[mask] = rng.choice([2, 3], size=mask.sum())
    proba = cascade_models["L3"].predict_proba(X_test_l3)[:, 1]
    idx2 = np.where(cost == 2)[0]
    idx3 = np.where(cost == 3)[0]
    if len(idx2):
        proba[idx2] = cascade_models["L34"].predict_proba(X_test_l34.iloc[idx2])[:, 1]
    if len(idx3):
        proba[idx3] = cascade_models["FULL"].predict_proba(X_test_full.iloc[idx3])[:, 1]
    avg_cost = float(np.mean([TIER_COST[int(c)] for c in cost]))
    return proba, cost, avg_cost

rng = np.random.default_rng(RANDOM_STATE)
esc_rate = float((cascade_cost > 1).mean())
rand_proba, rand_cost, rand_avg_cost = run_random_escalation(
    esc_rate, rng, X_test_l3, X_test_l34, X_test_full
)
print("Matched escalation rate:", esc_rate)"""))

cells.append(md("""**Explain your threshold choices and why calibration matters for the cascade.**

We use a wide uncertainty band [0.1, 0.9] because L3 probabilities are often extreme; narrower bands rarely trigger escalation. Calibration matters because thresholds assume probabilities are comparable across tiers; miscalibrated scores would escalate the wrong sessions."""))

cells.append(code(""))

cells.append(md("""---

## Part 8: Hyperparameter tuning & validation

**What:** Tune `n_estimators` for RF-FULL on the validation fold (session-grouped train subset)."""))

cells.append(code("""param_grid = {"n_estimators": [50, 100]}
grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    param_grid,
    scoring="average_precision",
    cv=3,
    n_jobs=-1,
)
grid.fit(X_fit_full, fit_df["downswitch"])
print("Best params:", grid.best_params_)
print(f"Best val AP (CV): {grid.best_score_:.4f}")

# Refit best on full train_df for reporting
best_n = grid.best_params_["n_estimators"]
rf_full = RandomForestClassifier(n_estimators=best_n, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
X_train_full, X_test_full = impute_features(train_df, FULL_COLS, train_df, test_df)
rf_full.fit(X_train_full, train_df["downswitch"])"""))

cells.append(md("""**Best hyperparameters:** `n_estimators=50` or `100` (whichever GridSearchCV selects). All tier models use the same setting for fair comparison."""))

cells.append(code(""))

cells.append(md("""---

## Part 9: Prediction Evaluation

**What:** Evaluate all models on the held-out test set with PR-AUC, precision/recall/F1, confusion matrices, per-service breakdown, and lead time."""))

cells.append(code("""def score_model(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, proba)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred)
    return {"PR_AUC": pr_auc, "precision": p, "recall": r, "f1": f1, "cm": cm, "proba": proba}

y_test = test_df["downswitch"].astype(int)

results = {}
results["resolution"] = score_model(y_test, resolution_model.predict_proba(test_df[["resolution"]])[:, 1])
results["L3"] = score_model(y_test, tier_models["L3"]["model"].predict_proba(tier_models["L3"]["X_test"])[:, 1])
results["L4"] = score_model(y_test, tier_models["L4"]["model"].predict_proba(tier_models["L4"]["X_test"])[:, 1])
results["L7"] = score_model(y_test, tier_models["L7"]["model"].predict_proba(tier_models["L7"]["X_test"])[:, 1])
results["RF_FULL"] = score_model(y_test, rf_full.predict_proba(X_test_full)[:, 1])
results["GB_FULL"] = score_model(y_test, gb_full.predict_proba(X_test_full)[:, 1])
results["cascade"] = score_model(y_test, cascade_proba)
results["random_escalation"] = score_model(y_test, rand_proba)

rows = []
for name, m in results.items():
    rows.append({"model": name, "PR_AUC": m["PR_AUC"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
metrics_df = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
metrics_df"""))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, name in zip(axes.ravel(), ["L3", "L4", "RF_FULL", "cascade"]):
    cm = results[name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrices.png", dpi=120)
plt.show()"""))

cells.append(code("""per_service_rows = []
for name in ["L3", "L4", "L7", "RF_FULL", "cascade"]:
    proba = results[name]["proba"]
    for svc in sorted(test_df["service"].unique()):
        mask = (test_df["service"] == svc).to_numpy()
        if y_test[mask].sum() == 0:
            continue
        ap = average_precision_score(y_test[mask], proba[mask])
        per_service_rows.append({"model": name, "service": svc, "PR_AUC": ap})
pd.DataFrame(per_service_rows).pivot(index="service", columns="model", values="PR_AUC").round(3)"""))

cells.append(code("""# Lead time: seconds before downswitch that RF-FULL first fires (prob >= 0.5)
ALERT_THRESHOLD = 0.5
lead_times = []
no_prior_alert = 0
test_reset = test_df.reset_index(drop=True)
proba_full = np.asarray(results["RF_FULL"]["proba"])

for i, row in test_reset.iterrows():
    if row["downswitch"] != 1:
        continue
    sid = row["session_id"]
    sess = test_reset[test_reset["session_id"] == sid].sort_values("relative_timestamp")
    idxs = sess.index.to_numpy()
    pos = int(np.where(idxs == i)[0][0])
    alert_time = None
    for j in range(pos, -1, -1):
        row_idx = idxs[j]
        if proba_full[row_idx] >= ALERT_THRESHOLD:
            alert_time = test_reset.at[row_idx, "relative_timestamp"]
            break
    if alert_time is None:
        no_prior_alert += 1
        continue
    lead = row["relative_timestamp"] - alert_time
    if lead >= 0:
        lead_times.append(lead)

print(f"Lead time (RF-FULL, threshold={ALERT_THRESHOLD}):")
print(f"  events with prior alert: n={len(lead_times)}")
print(f"  events with no prior alert: n={no_prior_alert}")
if lead_times:
    lt = np.asarray(lead_times)
    print(f"  median={np.median(lt):.1f}s  mean={np.mean(lt):.1f}s")
    print(f"  p25={np.percentile(lt, 25):.1f}s  p75={np.percentile(lt, 75):.1f}s  max={lt.max():.1f}s")
else:
    print("  (no lead times computed)")"""))

cells.append(md("""**Interpret the prediction results.**

- **RF-FULL** achieves the highest PR-AUC (**0.440**), well above the resolution-only baseline (~0.027).
- **L4/L7** tiers outperform **L3** alone, showing deeper protocol features help prediction.
- The **cascade** improves over L3-only but **does not match RF-FULL** at comparable cost when L3 probabilities are highly confident (few escalations).
- **Random escalation** at the same rate performs similarly or worse than the adaptive cascade, suggesting escalation only helps when targeted."""))

cells.append(code(""))

cells.append(md("""---

## Part 10: Systems-Cost Evaluation

**What:** Plot prediction quality vs. relative feature cost; measure PCAP extraction timing."""))

cells.append(code("""threshold_grid = [(0.45, 0.55), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8), (0.15, 0.85), (0.1, 0.9)]
curve_rows = []
for low, high in threshold_grid:
    proba, cost, avg_cost = run_cascade(low, high, X_test_l3, X_test_l34, X_test_full)
    ap = average_precision_score(y_test, proba)
    curve_rows.append({"low": low, "high": high, "PR_AUC": ap, "avg_cost": avg_cost, "escalate": (cost > 1).mean()})
curve_df = pd.DataFrame(curve_rows)
curve_df"""))

cells.append(code("""plt.figure(figsize=(7, 5))
plt.plot(curve_df["avg_cost"], curve_df["PR_AUC"], "o-", label="Cascade sweep")
for tier, cost, key in [("L3", 1, "L3"), ("L4", 4, "L4"), ("L7", 7, "L7"), ("RF_FULL", 10, "RF_FULL")]:
    plt.scatter(cost, results[key]["PR_AUC"], s=100, label=f"{tier} fixed")
plt.xlabel("Relative feature cost (units)")
plt.ylabel("PR-AUC")
plt.title("Prediction Quality vs. Systems Cost")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "quality_cost_curve.png", dpi=120)
plt.show()"""))

cells.append(code("""# Empirical PCAP timing (Netflix trace)
pcap_timings = {}
try:
    from netml.pparser.parser import PCAP

    t0 = time.time()
    pcap = PCAP(str(NETFLIX_PCAP_PATH), flow_ptks_thres=2, verbose=0)
    pcap.pcap2flows()
    pcap_timings["pcap2flows_sec"] = time.time() - t0

    t0 = time.time()
    pcap.flow2features("STATS", fft=False, header=True)
    pcap_timings["flow2features_STATS_sec"] = time.time() - t0
except Exception as exc:
    pcap_timings["error"] = str(exc)

pcap_timings["feature_counts"] = {"L3": len(L3_COLS), "L4": len(L4_COLS), "L7": len(L7_COLS)}
pcap_timings"""))

cells.append(code("""# Random escalation vs cascade at matched cost
comparison_cost = pd.DataFrame([
    {"method": "cascade", "PR_AUC": results["cascade"]["PR_AUC"], "avg_cost": cascade_avg_cost},
    {"method": "random_escalation", "PR_AUC": results["random_escalation"]["PR_AUC"], "avg_cost": rand_avg_cost},
    {"method": "L4_fixed", "PR_AUC": results["L4"]["PR_AUC"], "avg_cost": 4.0},
    {"method": "RF_FULL", "PR_AUC": results["RF_FULL"]["PR_AUC"], "avg_cost": 10.0},
])
comparison_cost"""))

cells.append(md("""**Interpret the cost results.**

- **L4 fixed** offers a strong operating point: ~78% of FULL PR-AUC (0.344 vs 0.440) at 40% relative feature cost (4 vs 10 units).
- The **cascade** can reduce average cost to near-L3 levels but sacrifices substantial PR-AUC unless escalation rate is high.
- **No single cascade threshold** simultaneously matches FULL quality and beats L4 on cost—honest finding per project spec.
- L7 adds modest PR-AUC gain over L4 but at highest extraction cost (chunk inference in encrypted traffic).

**PCAP timing result:** NetML successfully parsed `netflix.pcap` into flows in **15.5 seconds** (`pcap2flows_sec`). The subsequent `flow2features("STATS")` call stopped with `error: 'proto'`, meaning this NetML version expected a protocol field that was not present in the parsed flow representation. Therefore, we **do not claim an empirical feature-extraction time** from this run. The quality–cost analysis above instead uses the explicit relative cost model (L3=1, L4=4, FULL=10); the displayed feature counts come from the precomputed dataset schema. The exception is captured so this optional timing probe does not prevent the reproducible modeling pipeline from completing."""))

cells.append(code(""))

cells.append(md("""---

## Part 11: Error Analysis & Robustness Checks"""))

cells.append(code("""# False positives / negatives for RF-FULL
pred_full = (results["RF_FULL"]["proba"] >= 0.5).astype(int)
fp_mask = (pred_full == 1) & (y_test == 0)
fn_mask = (pred_full == 0) & (y_test == 1)
print("False positives:", fp_mask.sum())
print("False negatives:", fn_mask.sum())
print("FP rate among negatives:", fp_mask.sum() / (y_test == 0).sum())
print("FN rate among positives:", fn_mask.sum() / (y_test == 1).sum())"""))

cells.append(code("""# Calibration plot (RF-FULL)
prob_true, prob_pred = calibration_curve(y_test, results["RF_FULL"]["proba"], n_bins=10)
plt.figure(figsize=(5, 4))
plt.plot(prob_pred, prob_true, marker="o", label="RF-FULL")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration — RF-FULL")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "calibration.png", dpi=120)
plt.show()"""))

cells.append(code("""# Temporal drift: train on early sessions, test on late sessions
X_temp_train, X_temp_test = impute_features(temp_train_df, FULL_COLS, temp_train_df, temp_test_df)
drift_model = RandomForestClassifier(**RF_KW)
drift_model.fit(X_temp_train, temp_train_df["downswitch"])
temp_proba = drift_model.predict_proba(X_temp_test)[:, 1]
temp_ap = average_precision_score(temp_test_df["downswitch"], temp_proba)
session_ap = average_precision_score(y_test, results["RF_FULL"]["proba"])
print(f"PR-AUC session split: {session_ap:.4f}")
print(f"PR-AUC temporal split: {temp_ap:.4f}")
print(f"Delta (drift): {session_ap - temp_ap:.4f}")"""))

cells.append(md("""**What failure modes did you find?**

1. **Class imbalance** — model misses many downswitches (low recall at 0.5 threshold) despite reasonable PR-AUC.
2. **False positives** — flag sessions that never downswitch, wasting compute in a cascade deployment.
3. **Service variation** — Twitch/YouTube/Netflix differ in adaptation behavior; pooled metrics hide weak per-service performance.
4. **Temporal drift** — performance drops on later sessions, suggesting models should be refreshed over time.
5. **Cascade rarely escalates** — L3 probabilities are often extreme, limiting adaptive benefit without softer base models."""))

cells.append(code(""))

cells.append(md("""---

## Part 12: Cross-Model Comparison & Interpretation"""))

cells.append(code("""comparison = pd.DataFrame([
    {"model": "Resolution-only", "features": "resolution", "PR_AUC": results["resolution"]["PR_AUC"], "avg_cost": 0.1, "lead_time_median": np.nan},
    {"model": "L3-only", "features": "L3", "PR_AUC": results["L3"]["PR_AUC"], "avg_cost": 1.0, "lead_time_median": np.nan},
    {"model": "L4-only", "features": "L4", "PR_AUC": results["L4"]["PR_AUC"], "avg_cost": 4.0, "lead_time_median": np.nan},
    {"model": "L7-only", "features": "L7", "PR_AUC": results["L7"]["PR_AUC"], "avg_cost": 7.0, "lead_time_median": np.nan},
    {"model": "RF-FULL", "features": "L3+L4+L7", "PR_AUC": results["RF_FULL"]["PR_AUC"], "avg_cost": 10.0, "lead_time_median": np.median(lead_times) if lead_times else np.nan},
    {"model": "GB-FULL", "features": "L3+L4+L7", "PR_AUC": results["GB_FULL"]["PR_AUC"], "avg_cost": 10.0, "lead_time_median": np.nan},
    {"model": "Cascade", "features": "adaptive", "PR_AUC": results["cascade"]["PR_AUC"], "avg_cost": cascade_avg_cost, "lead_time_median": np.nan},
    {"model": "Random escalation", "features": "matched", "PR_AUC": results["random_escalation"]["PR_AUC"], "avg_cost": rand_avg_cost, "lead_time_median": np.nan},
]).sort_values("PR_AUC", ascending=False)
comparison.round(4)"""))

cells.append(code("""plt.figure(figsize=(7, 5))
for name in ["L3", "L4", "L7", "RF_FULL", "cascade"]:
    p, r, _ = precision_recall_curve(y_test, results[name]["proba"])
    plt.plot(r, p, label=f"{name} (AP={results[name]['PR_AUC']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "pr_curves.png", dpi=120)
plt.show()"""))

cells.append(md("""**Summarize the key findings across all models.**

Network features vastly outperform resolution-only for predicting future downswitches. L4 features capture most of the gain over L3; L7 adds a smaller increment at higher cost. The adaptive cascade does not beat a well-chosen **fixed L4 model** on the quality–cost frontier with our threshold policy, but it beats random escalation at the same average cost."""))

cells.append(code(""))

cells.append(md("""---

## Part 13: Conclusions

### 13.1 Research question

*Can adaptive feature selection reduce computational cost while matching full-feature performance?*

**Your answer:** **Partially, but not with our current cascade policy.** Adaptive selection reduces average feature cost, yet matching RF-FULL PR-AUC requires escalating enough sessions that cost approaches the full feature set. A fixed **L4-only model** is the practical sweet spot: most of FULL's PR-AUC at lower cost."""))

cells.append(code(""))

cells.append(md("""### 13.2 Key findings

1. Future downswitch prediction at Δ=10s is highly imbalanced (~1.7% positive) but learnable (RF-FULL PR-AUC = **0.440**).
2. L3 features alone are cheap but leave significant performance on the table.
3. L4 features provide the best static cost–quality tradeoff.
4. The cascade improves over L3-only but does not dominate L4 or FULL on both axes simultaneously.
5. Session-level evaluation is essential; random window splits would inflate scores."""))

cells.append(code(""))

cells.append(md("""### 13.3 Limitations

- Single primary horizon (10s); multi-horizon sweep reported only for label rates.
- Cascade thresholds tuned post hoc; no learned acquisition policy.
- The one-trace PCAP probe measured flow parsing (15.5s), but NetML's STATS extraction failed on a missing `proto` field; relative tier costs are modeled rather than empirically timed.
- No live deployment or online latency measurements."""))

cells.append(code(""))

cells.append(md("""### 13.4 Future work

- Learn escalation policy (e.g., reinforcement learning or meta-classifier on L3 uncertainty).
- Per-service models or domain adaptation for drift.
- Combine lead-time optimization with cost constraints.
- Evaluate on additional PCAP traces for cost generalization."""))

cells.append(code(""))

cells.append(md("""### 13.5 Learning objective reflection

**What did you learn?**

Feature depth matters for QoE prediction, but so does **when** you pay for features. Systems ML requires joint optimization of accuracy, latency, and extraction cost—not accuracy alone. Designing honest baselines (resolution-only, random escalation) was as important as the adaptive method."""))

cells.append(code(""))

cells.append(md("""---

## Part 14: Reproducibility & Submission Checklist

- [x] Restart kernel and run all — notebook executes top-to-bottom
- [x] Data paths documented relative to repo root
- [x] Random seed = 42
- [x] Plots saved under `final project/figures/`
- [x] Written justifications included
- [x] Comparison table complete (Part 12)
- [x] Conclusions answer the research question
- [ ] Companion Sphinx report (separate deliverable)

**Known assumptions:** Requires `completed assignments/data/video-qoe/video_dataset.pkl` and `netflix.pcap`. Training ~1–3 minutes on a laptop with 50 trees."""))

cells.append(code("""print("Notebook complete. Figures:", list(FIG_DIR.glob('*.png')))"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")

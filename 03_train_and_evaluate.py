"""
03_train_and_evaluate.py
────────────────────────────────────────────────────────────────────────────────
Weeks 2–3 work: EDA + 3×2 ablation study (Ridge, RandomForest, XGBoost) × (circuit-only, circuit+noise)

Outputs (all saved to results/):
  - fidelity_distribution.png
  - correlation_heatmap.png
  - ablation_table.csv              ← the main result table
  - ablation_table.txt              ← formatted for report copy-paste
  - calibration_scatter_<model>.png
  - feature_importance_<model>.png
  - wilcoxon_test.txt

Run AFTER 02_generate_data.py:
    python 03_train_and_evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import seaborn as sns
from scipy import stats

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_CSV   = Path("data/fidelity_dataset.csv")
RESULTS    = Path("results")
RESULTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_FRAC    = 0.20
CV_FOLDS     = 5

# Feature sets
CIRCUIT_FEATURES = [
    "total_gates", "cx_count", "ecr_count", "rz_count", "sx_count", "x_count",
    "depth", "num_qubits", "two_qubit_fraction", "critical_path",
]
NOISE_FEATURES = [
    "mean_t1", "mean_t2", "mean_sq_error", "mean_tq_error", "mean_ro_error",
]
FEATURE_SETS = {
    "A_circuit_only":  CIRCUIT_FEATURES,
    "B_circuit+noise": CIRCUIT_FEATURES + NOISE_FEATURES,
}

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {DATA_CSV}...")
df = pd.read_csv(DATA_CSV)
print(f"  {len(df)} samples, {df.shape[1]} columns")
print(f"  Fidelity: min={df.fidelity.min():.3f}  max={df.fidelity.max():.3f}  "
      f"mean={df.fidelity.mean():.3f}  std={df.fidelity.std():.3f}")
print()

# ── EDA ───────────────────────────────────────────────────────────────────────
print("Generating EDA plots...")

# 1. Fidelity distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["fidelity"], bins=40, edgecolor="white", color="#4C72B0")
axes[0].set_xlabel("Fidelity")
axes[0].set_ylabel("Count")
axes[0].set_title("Fidelity distribution")
axes[0].axvline(df["fidelity"].mean(), color="red", linestyle="--", label=f"Mean={df['fidelity'].mean():.3f}")
axes[0].legend()

# Fidelity by backend
for bname in df["backend"].unique():
    sub = df[df["backend"] == bname]["fidelity"]
    axes[1].hist(sub, bins=30, alpha=0.5, label=bname, edgecolor="white")
axes[1].set_xlabel("Fidelity")
axes[1].set_ylabel("Count")
axes[1].set_title("Fidelity by backend")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(RESULTS / "fidelity_distribution.png", dpi=150)
plt.close()

# 2. Correlation heatmap
num_cols = CIRCUIT_FEATURES + NOISE_FEATURES + ["fidelity"]
corr = df[[c for c in num_cols if c in df.columns]].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature correlation matrix")
plt.tight_layout()
fig.savefig(RESULTS / "correlation_heatmap.png", dpi=150)
plt.close()

# 3. Physics sanity check: CX count vs fidelity
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["cx_count"], df["fidelity"], alpha=0.3, s=10, c="#4C72B0")
ax.set_xlabel("CX count (post-transpilation)")
ax.set_ylabel("Fidelity")
ax.set_title("Physics sanity check: more CX gates → lower fidelity?")
plt.tight_layout()
fig.savefig(RESULTS / "cx_vs_fidelity.png", dpi=150)
plt.close()

print(f"  EDA plots saved to {RESULTS}/")

# ── Stratified train/test split by algorithm family ───────────────────────────
print("\nSplitting data...")
from sklearn.model_selection import train_test_split

# Encode family for stratification
df["family_code"] = pd.Categorical(df["family"]).codes
X_all = df[CIRCUIT_FEATURES + NOISE_FEATURES].copy().fillna(0)
y_all = df["fidelity"].values

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=TEST_FRAC,
    random_state=RANDOM_STATE,
    stratify=df["family_code"],
)
print(f"  Train: {len(train_idx)}  Test: {len(test_idx)}")

# ── Model definitions ─────────────────────────────────────────────────────────
def make_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge()),
    ])

def make_rf():
    return RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
    )

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        tree_method="hist", device="cuda"
    )

MODEL_FACTORIES = {
    "Ridge":        make_ridge,
    "RandomForest": make_rf,
    "XGBoost":      make_xgb,
}

# Hyperparameter grids for GridSearchCV
PARAM_GRIDS = {
    "Ridge": {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "n_estimators": [200, 400],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
    },
}

# ── Training + evaluation ─────────────────────────────────────────────────────
results_rows = []
per_sample_errors = {}   # model_key → array of abs errors on test set

print("\nTraining 6 models (3 × 2 ablation)...")

for model_name, factory in MODEL_FACTORIES.items():
    for feat_key, feat_cols in FEATURE_SETS.items():
        key = f"{model_name}__{feat_key}"
        print(f"\n  [{key}]")

        cols_available = [c for c in feat_cols if c in X_all.columns]
        X_train = X_all.iloc[train_idx][cols_available].values
        X_test  = X_all.iloc[test_idx][cols_available].values
        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]

        model = factory()
        param_grid = PARAM_GRIDS[model_name]

        # 5-fold CV on training set
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        # Use the family codes for stratification in CV
        train_family = df.iloc[train_idx]["family_code"].values

        gs = GridSearchCV(
            model, param_grid,
            cv=CV_FOLDS, scoring="neg_root_mean_squared_error",
            n_jobs=-1, refit=True, verbose=0,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        print(f"    Best params: {gs.best_params_}")
        print(f"    CV RMSE:     {-gs.best_score_:.4f}")

        # Test set evaluation
        y_pred = np.clip(best_model.predict(X_test), 0, 1)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        pearson_r, _  = stats.pearsonr(y_test, y_pred)
        spearman_r, _ = stats.spearmanr(y_test, y_pred)
        print(f"    Test RMSE={rmse:.4f}  MAE={mae:.4f}  "
              f"Pearson={pearson_r:.3f}  Spearman={spearman_r:.3f}")

        abs_errors = np.abs(y_test - y_pred)
        per_sample_errors[key] = abs_errors

        results_rows.append({
            "model":      model_name,
            "feature_set": feat_key,
            "cv_rmse":    round(-gs.best_score_, 4),
            "test_rmse":  round(rmse, 4),
            "test_mae":   round(mae, 4),
            "pearson_r":  round(pearson_r, 3),
            "spearman_r": round(spearman_r, 3),
            "best_params": str(gs.best_params_),
        })

        # ── Calibration scatter plot ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, y_pred, alpha=0.4, s=12, c="#4C72B0")
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("Actual fidelity")
        ax.set_ylabel("Predicted fidelity")
        ax.set_title(f"{model_name} — {feat_key}\nRMSE={rmse:.4f}  Pearson={pearson_r:.3f}")
        ax.legend()
        plt.tight_layout()
        fname = RESULTS / f"calibration_{model_name}_{feat_key}.png"
        fig.savefig(fname, dpi=150)
        plt.close()

        # ── Feature importance (tree models only) ────────────────────────────
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            fi_df = pd.Series(importances, index=cols_available).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            fi_df.plot(kind="barh", ax=ax, color="#4C72B0")
            ax.set_title(f"Feature importance — {model_name} ({feat_key})")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fig.savefig(RESULTS / f"importance_{model_name}_{feat_key}.png", dpi=150)
            plt.close()

# ── Ablation table ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3×2 ABLATION TABLE")
print("=" * 70)
results_df = pd.DataFrame(results_rows)
print(results_df[["model","feature_set","cv_rmse","test_rmse","test_mae","pearson_r","spearman_r"]].to_string(index=False))
results_df.to_csv(RESULTS / "ablation_table.csv", index=False)

# Formatted text version for report
with open(RESULTS / "ablation_table.txt", "w") as f:
    f.write("3×2 Ablation Results\n")
    f.write("=" * 70 + "\n")
    f.write(results_df[["model","feature_set","cv_rmse","test_rmse","test_mae","pearson_r","spearman_r"]].to_string(index=False))
    f.write("\n\n")
    f.write("Baseline (naive mean predictor):\n")
    mean_pred = np.full_like(y_all[test_idx], y_all[train_idx].mean())
    naive_rmse = float(np.sqrt(mean_squared_error(y_all[test_idx], mean_pred)))
    f.write(f"  RMSE = {naive_rmse:.4f}\n")
    f.write(f"\nPublished baselines:\n")
    f.write(f"  Q-fid (LSTM, real hardware): RMSE = 0.0515\n")
    f.write(f"  QuEst (graph transformer):   RMSE = 0.04\n")

# ── Wilcoxon test: A vs B per model ──────────────────────────────────────────
print("\nWilcoxon signed-rank tests (circuit-only vs circuit+noise):")
wilcoxon_lines = ["Wilcoxon signed-rank test: Feature Set A vs B\n" + "=" * 50 + "\n"]
for model_name in MODEL_FACTORIES:
    key_a = f"{model_name}__A_circuit_only"
    key_b = f"{model_name}__B_circuit+noise"
    if key_a in per_sample_errors and key_b in per_sample_errors:
        stat, pval = stats.wilcoxon(per_sample_errors[key_a], per_sample_errors[key_b])
        direction = "B better" if per_sample_errors[key_b].mean() < per_sample_errors[key_a].mean() else "A better"
        line = (f"  {model_name:14s}: stat={stat:.1f}  p={pval:.4f}  "
                f"({direction})  sig={'YES' if pval < 0.05 else 'NO'}")
        print(line)
        wilcoxon_lines.append(line + "\n")

with open(RESULTS / "wilcoxon_test.txt", "w") as f:
    f.writelines(wilcoxon_lines)

# ── Naive baseline RMSE ───────────────────────────────────────────────────────
print(f"\nNaive mean predictor RMSE: {naive_rmse:.4f}")
print(f"\nAll results saved to {RESULTS}/")
print("Done.")
#!/usr/bin/env python3
"""Interactive ML pipeline: predict ΔE from SMILES.

Run:
  python train.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def ask(prompt: str):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def ask_default(prompt: str, default):
    try:
        ans = input(f"{prompt} [Enter to use default ({default})]: ").strip()
    except EOFError:
        ans = ""
    return type(default)(ans) if ans else default


def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def print_metrics(rows, unit):
    print(f"\n{'Set':>8s} {f'MAE ({unit})':>12s} {f'RMSE ({unit})':>12s} {'R²':>8s}")
    print("   " + "-" * 42)
    for tag, mae, rmse, r2 in rows:
        print(f"{tag:>8s} {mae:12.3f} {rmse:12.3f} {r2:8.4f}")


def build_model(choice, seed):
    """Ask for hyperparameters (with defaults) and return (name, model)."""
    if choice == "1":
        print("   Linear Regression: fits a linear relationship between features and target.")
        return "Linear Regression", make_pipeline(
            StandardScaler(with_mean=False), LinearRegression()
        )

    if choice == "2":
        print("   Ridge Regression: alpha controls the strength of L2 regularization;")
        print("   larger alpha means stronger regularization.")
        alpha = ask_default(">> Ridge alpha", 10.0)
        return "Ridge Regression", make_pipeline(
            StandardScaler(with_mean=False), Ridge(alpha=alpha)
        )

    if choice == "3":
        print("   Random Forest: n_estimators controls the number of trees;")
        print("   max_depth limits the depth of each tree.")
        n_estimators = ask_default(">> Number of trees (n_estimators)", 200)
        max_depth    = ask_default(">> Max tree depth (max_depth)", 20)
        return "Random Forest", RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=seed, n_jobs=-1,
        )

    if choice == "4":
        print("   SVR: C controls the trade-off between model complexity and training errors;")
        print("   epsilon defines the width of the insensitive error tube.")
        C       = ask_default(">> SVR C", 10.0)
        epsilon = ask_default(">> SVR epsilon", 1.0)
        return "SVR", make_pipeline(
            StandardScaler(with_mean=False),
            SVR(kernel="rbf", C=C, epsilon=epsilon),
        )

    print("   * Invalid input — using Linear Regression.")
    return build_model("1", seed)


print("\n" + "=" * 60)
print("  ΔE prediction from SMILES")
print("=" * 60)

# ============================================================
#  Load dataset
# ============================================================
csv_path = Path(ask("\n>> Enter the path to your CSV file: "))

df   = pd.read_csv(csv_path)
cols = df.columns.tolist()

print(f"\n   Loaded {len(df)} rows. Here are the first 5 rows:")
print(df.head().to_string())

print(f"\n   The columns are:")
for i, c in enumerate(cols):
    print(f"     {i}: {c}")

# ============================================================
#  Select columns
# ============================================================
id_ans  = ask("\n>> Which column is the molecule ID? (enter number, or Enter to auto-generate): ")
smi_idx = int(ask(">> Which column contains the SMILES strings? (enter number): "))
de_idx  = int(ask(">> Which column contains the target ΔE values? (enter number): "))

smiles_col = cols[smi_idx]
de_col     = cols[de_idx]

if id_ans == "":
    data = pd.DataFrame({
        "id":      [f"mol_{i}" for i in range(len(df))],
        "smiles":  df[smiles_col].values,
        "delta_E": df[de_col].values,
    })
    print("   Using auto-generated IDs (mol_0, mol_1, ...).")
else:
    data = pd.DataFrame({
        "id":      df[cols[int(id_ans)]].values,
        "smiles":  df[smiles_col].values,
        "delta_E": df[de_col].values,
    })

# Clean missing / non-numeric ΔE
n_before = len(data)
data["delta_E"] = pd.to_numeric(data["delta_E"], errors="coerce")
data = data.dropna(subset=["smiles", "delta_E"]).reset_index(drop=True)
if len(data) < n_before:
    print(f"   * Dropped {n_before - len(data)} rows with missing SMILES or ΔE.")

print(f"\n   Selected {len(data)} molecules:")
print(data.head())

energy_unit = ask_default("\n>> Energy unit for ΔE", "Eh")

# ============================================================
#  Compute molecular fingerprints
# ============================================================
print("\n   Convert each SMILES into a Morgan fingerprint —")
print("   a fixed-length bit vector representing local molecular environments.")
print("   'radius' = the number of bond-expansion iterations around each atom;")
print("   'bits'   = the length of the fingerprint vector.")

fp_radius = ask_default("\n>> Morgan fingerprint radius", 2)
fp_nbits  = ask_default(">> Morgan fingerprint length", 2048)

gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_nbits)
fps, keep = [], []
for i, smi in enumerate(data["smiles"]):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        print(f"   * skipping invalid SMILES: {smi}")
        continue
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((fp_nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr.astype(float))
    keep.append(i)

data = data.iloc[keep].reset_index(drop=True)
X = np.vstack(fps)
y = data["delta_E"].to_numpy()
print(f"\n   Feature matrix X : {X.shape}  (molecules x fingerprint bits)")
print(f"   Target vector  y : {y.shape}  (ΔE in {energy_unit})")

# ============================================================
#  Train / Validation / Test split
# ============================================================
print("\n   Split the data into training, validation, and test sets.")
print("   Use validation to tune hyperparameters; look at test only once at the end.")

seed     = int(ask(">> Random seed for reproducibility: "))
test_pct = int(ask(">> Percentage of data for the test set (%): "))
val_pct  = int(ask(">> Percentage of data for the validation set (%): "))
train_pct = 100 - test_pct - val_pct
if train_pct <= 0:
    print("   * test + val >= 100%, resetting to 15/15.")
    test_pct, val_pct, train_pct = 15, 15, 70
print(f"   --> train {train_pct}% / val {val_pct}% / test {test_pct}%")

idx = np.arange(len(X))
idx_tv, idx_test = train_test_split(idx, test_size=test_pct / 100, random_state=seed)
idx_train, idx_val = train_test_split(
    idx_tv, test_size=val_pct / (100 - test_pct), random_state=seed
)
X_train, y_train = X[idx_train], y[idx_train]
X_val,   y_val   = X[idx_val],   y[idx_val]
X_test,  y_test  = X[idx_test],  y[idx_test]
print(f"       train = {len(idx_train)},  val = {len(idx_val)},  test = {len(idx_test)}")

# ============================================================
#  Model selection + hyperparameter tuning on validation
# ============================================================
print("\n   Choose a model to train:")
print("     1 = Linear Regression")
print("     2 = Ridge Regression")
print("     3 = Random Forest")
print("     4 = SVR")
choice = ask(">> Your choice: ")
if choice not in ("1", "2", "3", "4"):
    choice = "1"

# Tune using train / val only
while True:
    name, model = build_model(choice, seed)

    print(f"\n   Fitting {name} on the training set ...")
    model.fit(X_train, y_train)

    train_m = metrics(y_train, model.predict(X_train))
    val_m   = metrics(y_val,   model.predict(X_val))
    print("\n   Train / validation metrics (use these to decide hyperparameters):")
    print_metrics([("train", *train_m), ("val", *val_m)], energy_unit)

    again = ask("\n>> Adjust hyperparameters and retrain? [y/N]: ")
    if again.lower() != "y":
        break
    print("   Keeping the same data split. Re-enter hyperparameters.\n")

# ============================================================
#  Final evaluation on the held-out test set
# ============================================================
print("\n   Evaluating once on the held-out test set ...")
y_pred_test = model.predict(X_test)
test_m = metrics(y_test, y_pred_test)
print_metrics([("train", *train_m), ("val", *val_m), ("test", *test_m)], energy_unit)
test_info = {
    "y_true": y_test, "y_pred": y_pred_test,
    "MAE": test_m[0], "RMSE": test_m[1], "R2": test_m[2],
}

# ============================================================
#  Save results
# ============================================================
out_dir = Path(ask("\n>> Where to save results? (directory path): "))
out_dir.mkdir(parents=True, exist_ok=True)

pred_df = data.iloc[idx_test][["id", "smiles", "delta_E"]].copy()
pred_df = pred_df.rename(columns={"delta_E": "y_true"})
pred_df["y_pred"] = test_info["y_pred"]
pred_path = out_dir / "test_predictions.csv"
pred_df.to_csv(pred_path, index=False)
print(f"   Predictions saved --> {pred_path}")

# ============================================================
#  Parity plot
# ============================================================
do_plot = ask("\n>> Make a parity plot? [Y/n]: ")
if do_plot.lower() != "n":
    yt, yp = test_info["y_true"], test_info["y_pred"]
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.scatter(yt, yp, s=12, alpha=0.65, c="#750F6D")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"True $\Delta E$ ({energy_unit})")
    ax.set_ylabel(rf"Predicted $\Delta E$ ({energy_unit})")
    ax.set_title(name)
    ax.text(0.05, 0.95,
            f"MAE = {test_info['MAE']:.2f} {energy_unit}\n"
            f"RMSE = {test_info['RMSE']:.2f} {energy_unit}\n"
            f"$R^2$ = {test_info['R2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", pad=3))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / f"parity_{name.lower().replace(' ', '_')}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   Plot saved --> {plot_path}")

print("\n   All done!")

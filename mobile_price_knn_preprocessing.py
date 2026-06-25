"""
Mobile Price KNN Classification - Data Processing Script
=========================================================
Dataset: Mobile price dataset (21 features, 1000 rows)
Target:  price_range (0=low, 1=medium, 2=high, 3=very high)
         NOTE: If using the test set (no target column), the script
         will skip train/test split and produce X_scaled for prediction.

Steps covered:
  1. Load & inspect
  2. Drop irrelevant columns (id)
  3. Handle missing values (none found here, but handled generically)
  4. Feature engineering (screen area, pixel density)
  5. Separate features / target
  6. Outlier detection (IQR-based capping)
  7. Feature scaling (StandardScaler — required for KNN)
  8. Train/test split (skipped when no target column)
  9. Find optimal K via cross-validation (training data only)
 10. Save processed arrays
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ── 0. Configuration ──────────────────────────────────────────────────────────
DATA_PATH   = "Train.csv"          # ← change to your file path
OUTPUT_DIR  = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL  = "price_range"       # present in train set, absent in test set
DROP_COLS   = ["id"]              # columns to discard before modelling
TEST_SIZE   = 0.2
RANDOM_SEED = 42
K_RANGE     = range(1, 31)        # range of K values to evaluate

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Shape      : {df.shape}")
print(f"Columns    : {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ── 2. Drop irrelevant columns ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Dropping irrelevant columns")
print("=" * 60)

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
print(f"Dropped    : {DROP_COLS}")
print(f"Remaining  : {df.shape[1]} columns")

# ── 3. Missing value handling ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Missing value analysis")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values found — dataset is clean.")
else:
    print(f"Missing values detected:\n{missing}")
    for col in missing.index:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)   # categorical → mode
        else:
            df[col].fillna(df[col].median(), inplace=True)     # numeric → median
    print("Imputation complete (numeric → median, categorical → mode).")

# ── 4. Feature engineering ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Feature engineering")
print("=" * 60)

# Screen area (cm²)
if "sc_h" in df.columns and "sc_w" in df.columns:
    df["screen_area"] = df["sc_h"] * df["sc_w"]
    print("Created 'screen_area'  = sc_h × sc_w")

# Pixel density proxy (total pixels)
if "px_height" in df.columns and "px_width" in df.columns:
    df["total_pixels"] = df["px_height"] * df["px_width"]
    print("Created 'total_pixels' = px_height × px_width")

# Camera gap (rear - front camera)
if "pc" in df.columns and "fc" in df.columns:
    df["camera_gap"] = df["pc"] - df["fc"]
    print("Created 'camera_gap'   = pc - fc")

print(f"\nNew shape  : {df.shape}")

# ── 5. Separate features / target ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Separating features and target")
print("=" * 60)

has_target = TARGET_COL in df.columns

if has_target:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"Features   : {X.shape[1]}")
    print(f"Target     : '{TARGET_COL}'")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")
else:
    X = df.copy()
    y = None
    print(f"⚠  Target column '{TARGET_COL}' not found — inference mode (no labels).")
    print(f"Features   : {X.shape[1]}")

feature_names = list(X.columns)

# ── 6. Outlier capping (IQR method) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Outlier capping (IQR, factor=1.5)")
print("=" * 60)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
capped_count = 0

for col in numeric_cols:
    Q1  = X[col].quantile(0.25)
    Q3  = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    out = ((X[col] < lo) | (X[col] > hi)).sum()
    if out > 0:
        X[col] = X[col].clip(lower=lo, upper=hi)
        capped_count += out
        print(f"  {col:<20} — {out} outliers capped to [{lo:.2f}, {hi:.2f}]")

if capped_count == 0:
    print("No outliers detected in any numeric column.")
else:
    print(f"\nTotal values capped: {capped_count}")

# ── 7. Feature scaling ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Feature scaling (StandardScaler)")
print("=" * 60)

scaler   = StandardScaler()

if has_target:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train size : {X_train_raw.shape[0]} rows")
    print(f"Test size  : {X_test_raw.shape[0]}  rows")

    X_train = scaler.fit_transform(X_train_raw)   # fit only on train
    X_test  = scaler.transform(X_test_raw)         # transform test with same scaler
    print("Scaling applied (fit on train, transform on both splits).")
else:
    X_scaled = scaler.fit_transform(X)
    print(f"Scaling applied to all {X_scaled.shape[0]} rows (no train/test split — no labels).")

# ── 8. Optimal K selection (only if labels available) ─────────────────────────
if has_target:
    print("\n" + "=" * 60)
    print("STEP 8 — Finding optimal K via 5-fold cross-validation")
    print("=" * 60)

    cv_scores = {}
    for k in K_RANGE:
        knn    = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        cv_scores[k] = scores.mean()

    best_k    = max(cv_scores, key=cv_scores.get)
    best_acc  = cv_scores[best_k]
    print(f"Best K     : {best_k}  (CV accuracy = {best_acc:.4f})")

    # Plot K vs accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(list(cv_scores.keys()), list(cv_scores.values()), marker="o", color="steelblue")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    plt.xlabel("Number of Neighbours (K)")
    plt.ylabel("CV Accuracy")
    plt.title("KNN — Cross-Validation Accuracy vs K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_cv_accuracy.png", dpi=150)
    plt.close()
    print("Plot saved → output/knn_cv_accuracy.png")

    # ── 9. Train final model & evaluate ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 9 — Training KNN (K={best_k}) & evaluating on test set")
    print("=" * 60)

    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    y_pred    = knn_final.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Low", "Medium", "High", "Very High"]))

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Med", "High", "V.High"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — KNN (K={best_k})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Plot saved → output/confusion_matrix.png")

# ── 10. Save processed data ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10 — Saving processed data")
print("=" * 60)

if has_target:
    pd.DataFrame(X_train, columns=feature_names).to_csv(
        OUTPUT_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test,  columns=feature_names).to_csv(
        OUTPUT_DIR / "X_test.csv",  index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv( OUTPUT_DIR / "y_test.csv",  index=False)
    print("Saved → output/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
else:
    pd.DataFrame(X_scaled, columns=feature_names).to_csv(
        OUTPUT_DIR / "X_scaled.csv", index=False)
    print("Saved → output/X_scaled.csv")

# Feature importance proxy (variance after scaling)
feature_var = pd.Series(
    np.var(X_train if has_target else X_scaled, axis=0),
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 features by variance (post-scaling):")
print(feature_var.head(10).to_string())

feature_var.to_csv(OUTPUT_DIR / "feature_variance.csv", header=["variance"])
print("Saved → output/feature_variance.csv")

print("\n" + "=" * 60)
print("✅  Data processing complete!")
print("=" * 60)

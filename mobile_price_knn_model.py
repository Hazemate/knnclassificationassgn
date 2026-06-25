"""
Mobile Price KNN — Model Training, Prediction & Visualisation
==============================================================
Inputs :
  - Train.csv          (labelled training data)
  - X_scaled.csv       (pre-processed test features from preprocessing script)

Outputs (saved to  output/ folder):
  - predictions.csv
  - 01_class_distribution.png
  - 02_feature_importance.png
  - 03_knn_cv_accuracy.png
  - 04_confusion_matrix.png
  - 05_classification_report_heatmap.png
  - 06_pca_decision_boundary.png
  - 07_prediction_distribution.png
  - 08_top_features_boxplot.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH   = r"C:\Users\Ethnotech\Desktop\joyal\mobileprice\Train.csv"
TEST_SCALED  = r"C:\Users\Ethnotech\Desktop\joyal\mobileprice\output\X_scaled.csv"          # output from preprocessing script
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET       = "price_range"
DROP_COLS    = ["id"]
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
K_RANGE      = range(1, 31)
CLASS_NAMES  = ["Low", "Medium", "High", "Very High"]
PALETTE      = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]   # one colour per class

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Load & preprocess training data
# ═══════════════════════════════════════════════════════════════════════════════
print("── Loading training data ──────────────────────────────────")
train_df = pd.read_csv(TRAIN_PATH)
train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True)

# Same feature engineering as preprocessing script
if "sc_h" in train_df.columns and "sc_w" in train_df.columns:
    train_df["screen_area"]  = train_df["sc_h"] * train_df["sc_w"]
if "px_height" in train_df.columns and "px_width" in train_df.columns:
    train_df["total_pixels"] = train_df["px_height"] * train_df["px_width"]
if "pc" in train_df.columns and "fc" in train_df.columns:
    train_df["camera_gap"]   = train_df["pc"] - train_df["fc"]

# Handle missing values
for col in train_df.select_dtypes(include=[np.number]).columns:
    train_df[col].fillna(train_df[col].median(), inplace=True)

# Outlier capping
feature_cols = [c for c in train_df.columns if c != TARGET]
for col in feature_cols:
    Q1, Q3 = train_df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    train_df[col] = train_df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

X = train_df[feature_cols]
y = train_df[TARGET]
print(f"Training data: {X.shape[0]} rows × {X.shape[1]} features")
print(f"Classes      : {sorted(y.unique())}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Train / validation split  +  scaling
# ═══════════════════════════════════════════════════════════════════════════════
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Class distribution in training set
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 1: Class distribution ─────────────────────────────")
counts = y_train.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(CLASS_NAMES, counts.values, color=PALETTE, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Training Set — Class Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_class_distribution.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Feature importance (mutual info / correlation with target)
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 2: Feature importance ─────────────────────────────")
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 8))
colors = ["#C44E52" if v == mi_series.max() else "#4C72B0" for v in mi_series.values]
bars = ax.barh(mi_series.index, mi_series.values, color=colors, edgecolor="white")
ax.set_title("Feature Importance — Mutual Information Score", fontsize=14,
             fontweight="bold", pad=12)
ax.set_xlabel("Mutual Information")
ax.axvline(mi_series.mean(), color="grey", linestyle="--", alpha=0.7, label="Mean")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_feature_importance.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Cross-validation accuracy vs K
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 3: CV accuracy vs K ───────────────────────────────")
cv_means, cv_stds = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for k in K_RANGE:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_train, y_train, cv=skf, scoring="accuracy")
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

cv_means = np.array(cv_means)
cv_stds  = np.array(cv_stds)
best_k   = list(K_RANGE)[np.argmax(cv_means)]
print(f"Best K = {best_k}  (CV acc = {cv_means.max():.4f})")

fig, ax = plt.subplots(figsize=(10, 4))
ks = list(K_RANGE)
ax.plot(ks, cv_means, marker="o", color="#4C72B0", linewidth=2, markersize=5)
ax.fill_between(ks, cv_means - cv_stds, cv_means + cv_stds,
                alpha=0.2, color="#4C72B0", label="±1 std")
ax.axvline(best_k, color="#C44E52", linestyle="--", linewidth=1.8,
           label=f"Best K = {best_k}  ({cv_means.max():.3f})")
ax.set_xlabel("K (Number of Neighbours)")
ax.set_ylabel("5-Fold CV Accuracy")
ax.set_title("KNN — Cross-Validation Accuracy vs K", fontsize=14,
             fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_knn_cv_accuracy.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Train final model & validate
# ═══════════════════════════════════════════════════════════════════════════════
print(f"── Training final KNN (K={best_k}) ────────────────────────")
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(X_train, y_train)

y_pred_val = knn.predict(X_val)
val_acc    = accuracy_score(y_val, y_pred_val)
print(f"Validation accuracy: {val_acc:.4f}")
print(classification_report(y_val, y_pred_val, target_names=CLASS_NAMES))

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 4: Confusion matrix ───────────────────────────────")
cm = confusion_matrix(y_val, y_pred_val)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — KNN (K={best_k})\nValidation Accuracy: {val_acc:.2%}",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_confusion_matrix.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Classification report heatmap (precision / recall / f1)
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 5: Classification report heatmap ──────────────────")
report = classification_report(y_val, y_pred_val,
                               target_names=CLASS_NAMES, output_dict=True)
report_df = pd.DataFrame(report).T.loc[CLASS_NAMES, ["precision", "recall", "f1-score"]]

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(report_df.astype(float), annot=True, fmt=".3f", cmap="YlGn",
            vmin=0, vmax=1, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Per-Class Metrics — Precision / Recall / F1",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_classification_report_heatmap.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — PCA 2D decision boundary
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 6: PCA decision boundary ──────────────────────────")
pca      = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca    = pca.fit_transform(X_train)
var_exp  = pca.explained_variance_ratio_ * 100

knn_pca  = KNeighborsClassifier(n_neighbors=best_k)
knn_pca.fit(X_pca, y_train)

h   = 0.3
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_bg = plt.matplotlib.colors.ListedColormap(
    ["#AEC6E8", "#A8D5B5", "#F2AEAE", "#C8B8E8"])
cmap_pt = plt.matplotlib.colors.ListedColormap(PALETTE)

fig, ax = plt.subplots(figsize=(9, 6))
ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=y_train, cmap=cmap_pt,
                     s=18, edgecolors="white", linewidth=0.4, alpha=0.85)
patches = [mpatches.Patch(color=PALETTE[i], label=CLASS_NAMES[i]) for i in range(4)]
ax.legend(handles=patches, loc="upper right", fontsize=9)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
ax.set_title(f"KNN Decision Boundary (K={best_k}) — PCA Projection",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_pca_decision_boundary.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Predict on pre-processed test set (X_scaled.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("── Predicting on X_scaled.csv ─────────────────────────────")
X_test_scaled = pd.read_csv(TEST_SCALED)

# Align columns (same order as training)
X_test_aligned = X_test_scaled[feature_cols]

test_preds      = knn.predict(X_test_aligned.values)
test_pred_proba = knn.predict_proba(X_test_aligned.values)

pred_df = pd.DataFrame(test_pred_proba,
                       columns=[f"prob_{c.lower().replace(' ','_')}" for c in CLASS_NAMES])
pred_df.insert(0, "predicted_class",   test_preds)
pred_df.insert(1, "predicted_label",
               pd.Series(test_preds).map({i: c for i, c in enumerate(CLASS_NAMES)}))
pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
print(f"Predictions saved → output/predictions.csv  ({len(pred_df)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Prediction distribution on test set
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 7: Prediction distribution ───────────────────────")
pred_counts = pred_df["predicted_label"].value_counts().reindex(CLASS_NAMES)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
bars = axes[0].bar(CLASS_NAMES, pred_counts.values, color=PALETTE,
                   edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, pred_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].set_title("Test Set — Predicted Class Counts", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, pred_counts.max() * 1.18)

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    pred_counts.values, labels=CLASS_NAMES,
    autopct="%1.1f%%", colors=PALETTE,
    startangle=140, pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5})
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
axes[1].set_title("Test Set — Predicted Class Share", fontsize=13, fontweight="bold")

plt.suptitle("KNN Predictions on Test Set", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_prediction_distribution.png", bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Top 6 features: box plots by class (training data)
# ═══════════════════════════════════════════════════════════════════════════════
print("── Plot 8: Top features box plots ─────────────────────────")
mi_top6  = mi_series.sort_values(ascending=False).head(6).index.tolist()
plot_df  = X_train_raw.copy()
plot_df[TARGET] = y_train.values

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(mi_top6):
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        data = plot_df[plot_df[TARGET] == cls_idx][feat]
        bp = axes[i].boxplot(data, positions=[cls_idx], widths=0.5,
                             patch_artist=True, notch=False,
                             boxprops=dict(facecolor=PALETTE[cls_idx], alpha=0.7),
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(color=PALETTE[cls_idx]),
                             capprops=dict(color=PALETTE[cls_idx]),
                             flierprops=dict(marker="o", markerfacecolor=PALETTE[cls_idx],
                                             markersize=2, alpha=0.4))
    axes[i].set_xticks(range(4))
    axes[i].set_xticklabels(["Low", "Med", "High", "V.Hi"], fontsize=9)
    axes[i].set_title(feat, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Value")

fig.suptitle("Top 6 Features — Distribution by Price Class (Training Set)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_top_features_boxplot.png", bbox_inches="tight")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("✅  All done!")
print(f"   Best K              : {best_k}")
print(f"   CV Accuracy         : {cv_means.max():.4f}")
print(f"   Validation Accuracy : {val_acc:.4f}")
print(f"   Test predictions    : {len(pred_df)} rows  → output/predictions.csv")
print("   Plots saved in      : output/")
print("═" * 60)

# knnclassificationassgn
# 📱 Mobile Price Classification — KNN Model

A K-Nearest Neighbours classifier that predicts a mobile phone's price range based on hardware specifications.

---

## 🗂️ Project Structure

```
mobileprice/
├── Train.csv                              ← labelled training data (2000 rows)
├── test.csv                               ← unlabelled test data  (1000 rows)
└── output/
    ├── mobile_price_knn_preprocessing.py  ← Step 1: clean & scale data
    ├── mobile_price_knn_model.py          ← Step 2: train KNN, save predictions
    ├── test_predictions.py                ← Step 3: inspect results + Q&A
    ├── predictions.csv                    ← model output
    ├── X_scaled.csv                       ← scaled test features
    ├── feature_variance.csv               ← feature variance after scaling
    └── *.png                              ← 8 visualisation plots
```

---

## 🎯 Target Classes

| Class | Label      | Description     |
|-------|-----------|-----------------|
| 0     | Low        | Budget phones   |
| 1     | Medium     | Mid-range       |
| 2     | High       | Premium         |
| 3     | Very High  | Flagship        |

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

> ⚠️ Always run from the `mobileprice/` folder, **not** from inside `output/`.

### Step 1 — Preprocess the data
```bash
cd C:\...\mobileprice
py output\mobile_price_knn_preprocessing.py
```
Outputs: `X_scaled.csv`, `feature_variance.csv`

### Step 2 — Train the model & generate predictions
```bash
py output\mobile_price_knn_model.py
```
Outputs: `predictions.csv`, 8 PNG plots

### Step 3 — Inspect predictions
```bash
py output\test_predictions.py               # interactive menu
py output\test_predictions.py --summary     # overall stats
py output\test_predictions.py --row 42      # inspect row 42
py output\test_predictions.py --filter High # all High predictions
py output\test_predictions.py --ask         # Q&A about the model
```

---

## 🧠 How KNN Works

For each new phone, KNN:
1. Computes **Euclidean distance** to every training sample
2. Finds the **K closest neighbours**
3. Takes a **majority vote** of their price classes
4. Assigns that class as the prediction

> **Why scaling is required:** KNN is distance-based. Without `StandardScaler`, large-range features like RAM (256–3998) dominate tiny binary features (0/1), and the model effectively ignores them.

---

## ⚙️ Features Used

**Original (20):** `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`

**Engineered (3 new):**

| Feature        | Formula              |
|----------------|----------------------|
| `screen_area`  | `sc_h × sc_w`        |
| `total_pixels` | `px_height × px_width` |
| `camera_gap`   | `pc - fc`            |

**Total fed into KNN: 23 features.** Top predictor: `ram`.

---

## 🔢 Finding the Best K

K was selected using **Stratified 5-Fold Cross-Validation** over K = 1 to 30:

```python
for k in range(1, 31):
    scores = cross_val_score(KNeighborsClassifier(k), X_train, y_train, cv=5)
    # pick K with highest mean accuracy
```

Result visualised in `03_knn_cv_accuracy.png`.

---

## 📊 Output Visualisations

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Training class counts |
| `02_feature_importance.png` | Mutual information score per feature |
| `03_knn_cv_accuracy.png` | CV accuracy vs K (with ±1 std band) |
| `04_confusion_matrix.png` | True vs predicted labels |
| `05_classification_report_heatmap.png` | Per-class precision / recall / F1 |
| `06_pca_decision_boundary.png` | 2D decision boundary via PCA |
| `07_prediction_distribution.png` | Bar + pie chart of test predictions |
| `08_top_features_boxplot.png` | Top 6 features by price class |

---

## 🔍 Understanding Confidence Scores

KNN confidence = fraction of K neighbours that voted for the predicted class.

**Example (K=13):**
```
9 voted High   → prob_high   = 9/13 = 69.2%  ← predicted
3 voted Medium → prob_medium = 3/13 = 23.1%
1 voted Low    → prob_low    = 1/13 =  7.7%
```

| Confidence | Rating     | Meaning             |
|-----------|-----------|---------------------|
| ≥ 75%     | VERY HIGH ✔✔ | Very reliable     |
| 55–74%    | HIGH ✔     | Reliable           |
| 40–54%    | MODERATE ~ | Uncertain          |
| 25–39%    | LOW ✗      | Treat with caution |
| < 25%     | VERY LOW ✗✗ | Likely wrong      |

---

## 🐛 Errors Encountered & Fixes

### ❌ Error 1 — `SyntaxError: (unicode error) 'unicodeescape'`

**Cause:** Windows backslashes in file paths are treated as escape sequences.
```python
# BAD — \U is a unicode escape, \D is invalid
DATA_PATH = "C:\Users\Desktop\Train.csv"
```
**Fix:** Use a raw string, forward slashes, or a relative path:
```python
DATA_PATH = r"C:\Users\Desktop\Train.csv"   # raw string
DATA_PATH = "C:/Users/Desktop/Train.csv"    # forward slashes
DATA_PATH = "Train.csv"                     # relative (easiest)
```

---

### ❌ Error 2 — `FileNotFoundError: Train.csv not found`

**Cause:** Script was run from inside the `output\` subfolder.

**Fix:** `cd` back to the parent folder before running:
```bash
cd C:\...\mobileprice       # go up one level
py output\mobile_price_knn_model.py
```
The script uses `pathlib.__file__` internally to find sibling files automatically.

---

### ❌ Error 3 — `KeyError: 'price_range'`

**Cause:** Both `Train.csv` and `test.csv` were the **unlabelled** Kaggle test set — neither had the target column.

**Fix:** Download the correct file from Kaggle:
> 🔗 [kaggle.com/datasets/iabhishekofficial/mobile-price-classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

The download contains:
- `train.csv` — **2000 rows with `price_range`** ✅ (use this for training)
- `test.csv` — 1000 rows, no label (use for prediction only)

---

## 🤖 CLI Q&A Assistant

The inspector includes a built-in Q&A assistant about the model:

```bash
py output\test_predictions.py --ask
```

Or choose option **6** in the interactive menu. Example questions:

```
what is this project
how does knn work
what is the best k
what features are used
how do i fix the backslash error
what errors did i get
how do i run the project
what do confidence scores mean
what plots are generated
why do we scale features
tell me about the dataset
what is in predictions.csv
how is model accuracy measured
```

---

## 📁 Dataset

- **Source:** [Kaggle — Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- **Training set:** 2000 rows, 20 features + `price_range` label
- **Test set:** 1000 rows, 20 features, no label

---

## 📄 License

MIT

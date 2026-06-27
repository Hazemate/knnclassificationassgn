import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load training data
df = pd.read_csv("Train.csv")

# Remove id column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Feature engineering
df["screen_area"] = df["sc_h"] * df["sc_w"]
df["total_pixels"] = df["px_height"] * df["px_width"]
df["camera_gap"] = df["pc"] - df["fc"]

# Separate features and target
X = df.drop(columns=["price_range"])
y = df["price_range"]

feature_cols = X.columns

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_scaled, y)

print("\nEnter specifications of the new phone:\n")

data = {}

for col in feature_cols:
    if col in ["screen_area", "total_pixels", "camera_gap"]:
        continue

    value = float(input(f"{col}: "))
    data[col] = value

# Create engineered features
data["screen_area"] = data["sc_h"] * data["sc_w"]
data["total_pixels"] = data["px_height"] * data["px_width"]
data["camera_gap"] = data["pc"] - data["fc"]

new_phone = pd.DataFrame([data])
new_phone = new_phone[feature_cols]

# Scale
new_scaled = scaler.transform(new_phone)

# Predict
prediction = knn.predict(new_scaled)[0]
prob = knn.predict_proba(new_scaled)[0]

classes = ["Low", "Medium", "High", "Very High"]

print("\nPrediction Results")
print("------------------")
print("Predicted Price Category:", classes[prediction])
print()

for c, p in zip(classes, prob):
    print(f"{c:<10}: {p*100:.2f}%")

print(f"\nConfidence: {max(prob)*100:.2f}%")
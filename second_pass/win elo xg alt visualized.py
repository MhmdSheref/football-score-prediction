import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# CONFIGURATION & AESTHETICS
# =====================================================

# Set seaborn theme for professional, presentation-ready graphics
sns.set_theme(style="whitegrid", context="talk", palette="deep")

FILES = ["2022_xg_alt.csv", "2018_xg_alt.csv"]
TARGET = "win"

# Features negated during mirroring
DIFF_FEATURES = ["elo_diff"]

# Features swapped during mirroring
SWAP_PAIRS = [("alt 1", "alt 2"), ("team 1 xg for", "team 2 xg for")]

# The final list of features to train the model on
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for"]

TEST_SIZE = 0.30
VAL_SIZE = 0.50
RANDOM_SEED = 42

# =====================================================
# LOAD & CLEAN
# =====================================================

try:
    dfs = [pd.read_csv(file) for file in FILES]
    df = pd.concat(dfs, ignore_index=True)
except FileNotFoundError:
    print("Files not found. Please ensure the CSV files are in the directory.")
    exit()

df = df.dropna(subset=[TARGET] + ["team 1 elo", "team 2 elo"] + [col for pair in SWAP_PAIRS for col in pair])
df[TARGET] = df[TARGET].astype(int)

df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

# =====================================================
# DATA AUGMENTATION
# =====================================================

flipped = df.copy()

for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

for col1, col2 in SWAP_PAIRS:
    flipped[col1] = df[col2]
    flipped[col2] = df[col1]

flipped[TARGET] = 1 - flipped[TARGET]

augmented = pd.concat([df, flipped], ignore_index=True)

# =====================================================
# FEATURES & SPLIT
# =====================================================

X = augmented[MODEL_FEATURES]
y = augmented[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp
)

# =====================================================
# TRAIN (WITH SCALER PIPELINE)
# =====================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

# =====================================================
# VISUALIZATION 1: FEATURE IMPORTANCE
# =====================================================
# Because we used a StandardScaler, these coefficients are standardized
# and directly comparable to each other.

classifier = model.named_steps["classifier"]
scaled_coefs = classifier.coef_[0]

feature_importance_df = pd.DataFrame({
    "Feature": MODEL_FEATURES,
    "Coefficient (Impact on Odds)": scaled_coefs
}).sort_values(by="Coefficient (Impact on Odds)", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance_df,
    x="Coefficient (Impact on Odds)",
    y="Feature",
    palette="coolwarm"
)
plt.title("Feature Importance\n(Standardized Logistic Coefficients)")
plt.xlabel("Impact on Log-Odds of Winning")
plt.ylabel("")
plt.axvline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 2: WIN PROBABILITY CURVE VS ELO
# =====================================================

elo_range = np.linspace(X["elo_diff"].min(), X["elo_diff"].max(), 1000)

plot_df = pd.DataFrame({
    "elo_diff": elo_range,
    "alt 1": np.zeros(1000), "alt 2": np.zeros(1000),
    "team 1 xg for": np.zeros(1000), "team 2 xg for": np.zeros(1000)
})[MODEL_FEATURES]

probs = model.predict_proba(plot_df)[:, 1]

plt.figure(figsize=(12, 7))

# Plot actual outcomes (jittered slightly on the Y axis for visibility)
jittered_y = y_train + np.random.uniform(-0.03, 0.03, size=len(y_train))
sns.scatterplot(
    x=X_train["elo_diff"],
    y=jittered_y,
    hue=y_train,
    palette={0: "crimson", 1: "dodgerblue"},
    alpha=0.4,
    legend=False
)

# Plot the learned probability curve
plt.plot(elo_range, probs, linewidth=4, color="darkorange", label="Learned Win Probability")

plt.axvline(0, linestyle="--", color="gray", alpha=0.7)
plt.axhline(0.5, linestyle=":", color="gray", alpha=0.7)

plt.xlabel("Elo Difference")
plt.ylabel("Probability of Team 1 Winning")
plt.title("Learned Logistic Curve vs Actual Outcomes\n(Secondary features held at 0)")
plt.legend(loc="lower right")
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.tight_layout()
plt.show()

# =====================================================
# EVALUATION & VISUALIZATION 3: CONFUSION MATRICES
# =====================================================

def plot_custom_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Loss", "Predicted Win"],
                yticklabels=["Actual Loss", "Actual Win"])
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def evaluate_model(X_data, y_data, dataset_name):
    preds = model.predict(X_data)
    print(f"\n--- {dataset_name.upper()} EVALUATION ---")
    print(f"Accuracy: {accuracy_score(y_data, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_data, preds))
    plot_custom_confusion_matrix(y_data, preds, dataset_name)

evaluate_model(X_val, y_val, "Validation")
evaluate_model(X_test, y_test, "Test")

# =====================================================
# VISUALIZATION 4: STANDARD ELO VS LEARNED MODEL
# =====================================================

standard_elo_range = np.linspace(-600, 600, 1000)
elo_curve = 1 / (1 + 10 ** (-standard_elo_range / 400))

plot_df_comparison = pd.DataFrame({
    "elo_diff": standard_elo_range,
    "alt 1": np.zeros(1000), "alt 2": np.zeros(1000),
    "team 1 xg for": np.zeros(1000), "team 2 xg for": np.zeros(1000)
})[MODEL_FEATURES]

lr_curve = model.predict_proba(plot_df_comparison)[:, 1]

plt.figure(figsize=(12, 7))
plt.plot(standard_elo_range, elo_curve, label="Standard Elo Formula", linewidth=3, linestyle="--", color="steelblue")
plt.plot(standard_elo_range, lr_curve, label="Learned Model (Secondary Features=0)", linewidth=4, color="darkorange")

# Fill the area between the curves to highlight where the model diverges from standard Elo
plt.fill_between(standard_elo_range, elo_curve, lr_curve, color="orange", alpha=0.1)

plt.axvline(0, linestyle=":", color="gray")
plt.axhline(0.5, linestyle=":", color="gray")

plt.xlabel("Elo Difference")
plt.ylabel("Win Probability")
plt.title("Standard Elo Theory vs. Learned Real-World Model")
plt.legend()
plt.tight_layout()
plt.show()
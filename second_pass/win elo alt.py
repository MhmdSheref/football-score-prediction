import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# CONFIGURATION
# Modify these variables to change the script's behavior
# =====================================================

FILES = ["2022_alt.csv", "2018_alt.csv"]
TARGET = "win"

# Features that are a "difference" (Team 1 - Team 2). Flipped by multiplying by -1.
DIFF_FEATURES = ["elo_diff"]

# Features that are raw individual stats. Flipped by swapping Team 1's and Team 2's values.
SWAP_PAIRS = [("alt 1", "alt 2")]

# The final list of features to train the model on
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2"]

TEST_SIZE = 0.30
VAL_SIZE = 0.50  # This is 50% of the temp set (which is 30% of total)
RANDOM_SEED = 42

# =====================================================
# LOAD & CLEAN
# =====================================================

# Load all files dynamically based on CONFIG
dfs = [pd.read_csv(file) for file in FILES]
df = pd.concat(dfs, ignore_index=True)

print(f"Combined rows: {len(df)}")

df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

print(f"Rows after cleaning: {len(df)}")

# Create elo_diff
df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

# =====================================================
# DATA AUGMENTATION
# =====================================================

flipped = df.copy()

# 1. Dynamically flip difference-based features
for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

# 2. Dynamically swap individual team metrics
for col1, col2 in SWAP_PAIRS:
    # Pulling directly from the original 'df' prevents overwriting issues
    # (e.g., if we set flipped[col1] = flipped[col2], the original col1 is lost)
    flipped[col1] = df[col2]
    flipped[col2] = df[col1]

# 3. Swap the result
flipped[TARGET] = 1 - flipped[TARGET]

augmented = pd.concat([df, flipped], ignore_index=True)

print(f"\nOriginal dataset: {len(df)}")
print(f"Augmented dataset: {len(augmented)}")

# =====================================================
# VISUALIZE AUGMENTATION (Elo Diff only)
# =====================================================

plt.figure(figsize=(10, 6))
plt.scatter(df["elo_diff"], df[TARGET], alpha=0.4, label="Original")
plt.scatter(flipped["elo_diff"], flipped[TARGET], alpha=0.2, label="Flipped")

plt.xlabel("Elo Difference")
plt.ylabel("Win")
plt.title("Original + Flipped Matches")
plt.legend()
plt.show()

# =====================================================
# FEATURES & SPLIT
# =====================================================

X = augmented[MODEL_FEATURES]
y = augmented[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =====================================================
# TRAIN
# =====================================================

model = LogisticRegression()
model.fit(X_train, y_train)

print("\nCoefficients:")
for feature, coef in zip(MODEL_FEATURES, model.coef_[0]):
    print(f"  {feature}: {coef:.5f}")
print(f"Intercept: {model.intercept_[0]:.5f}")

# =====================================================
# LEARNED CURVE (Plotting Elo Diff while holding Alts at 0)
# =====================================================

elo_range = np.linspace(X["elo_diff"].min(), X["elo_diff"].max(), 1000)

plot_df = pd.DataFrame({
    "elo_diff": elo_range,
    "alt 1": np.zeros(1000),
    "alt 2": np.zeros(1000)
})

# Reorder columns to ensure they match model training exactly
plot_df = plot_df[MODEL_FEATURES]

probs = model.predict_proba(plot_df)[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_train["elo_diff"], y_train, alpha=0.15)
plt.plot(elo_range, probs, linewidth=3, color="orange")
plt.axvline(0, linestyle="--", color="black", alpha=0.5)

plt.xlabel("Elo Difference (with Alt features held at 0)")
plt.ylabel("P(Team 1 Wins)")
plt.title("Win Probability vs Elo Difference")
plt.show()


# =====================================================
# EVALUATION (Validation & Test)
# =====================================================

def evaluate_model(X_data, y_data, dataset_name):
    preds = model.predict(X_data)
    print(f"\n--- {dataset_name.upper()} EVALUATION ---")
    print(f"Accuracy: {accuracy_score(y_data, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_data, preds))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_data, preds))
    disp.plot(cmap="Blues")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.show()


evaluate_model(X_val, y_val, "Validation")
evaluate_model(X_test, y_test, "Test")

# =====================================================
# EXAMPLE ODDS & ELO COMPARISON
# =====================================================

print("\nExample Predictions (assuming neutral alt 1 & alt 2)")
example_diffs = [-400, -200, 0, 200, 400]

example_df = pd.DataFrame({
    "elo_diff": example_diffs,
    "alt 1": [0] * len(example_diffs),
    "alt 2": [0] * len(example_diffs)
})[MODEL_FEATURES]

example_probs = model.predict_proba(example_df)[:, 1]

for diff, prob in zip(example_diffs, example_probs):
    print(f"Elo diff {diff:+4d} -> {prob:.3f}")

# Compare to Standard Elo
standard_elo_range = np.linspace(-600, 600, 1000)
elo_curve = 1 / (1 + 10 ** (-standard_elo_range / 400))

plot_df_comparison = pd.DataFrame({
    "elo_diff": standard_elo_range,
    "alt 1": np.zeros(1000),
    "alt 2": np.zeros(1000)
})[MODEL_FEATURES]

lr_curve = model.predict_proba(plot_df_comparison)[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(standard_elo_range, elo_curve, label="Standard Elo Formula", linewidth=3)
plt.plot(standard_elo_range, lr_curve, label="Learned Logistic (Alts=0)", linewidth=3)
plt.axvline(0, linestyle="--", color="black", alpha=0.5)

plt.xlabel("Elo Difference")
plt.ylabel("Win Probability")
plt.title("Standard Elo vs Learned Logistic")
plt.legend()
plt.show()
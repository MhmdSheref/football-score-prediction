import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================================================
# CONFIGURATION
# =====================================================

FILES = ["2022_humid_clean.csv", "2018_humid_clean.csv"]
TARGET = "win"

# Features that are a "difference" (Team 1 - Team 2). Flipped by multiplying by -1.
DIFF_FEATURES = ["elo_diff"]

# Features that are raw individual stats. Flipped by swapping Team 1's and Team 2's values.
SWAP_PAIRS = [("alt 1", "alt 2"), ("team 1 xg for", "team 2 xg for"), ("humid 1", "humid 2")]

# The final list of features to train the model on
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for", "humid 1", "humid 2"]

TEST_SIZE = 0.30
VAL_SIZE = 0.5  # This is 50% of the temp set (which is 30% of total)

# Define the number of random seeds to try
NUM_RUNS = 50
# Generate a list of random seeds for reproducibility
SEEDS = [42 + i for i in range(NUM_RUNS)]

# =====================================================
# LOAD & CLEAN (Only needs to happen once)
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
# DATA AUGMENTATION (Only needs to happen once)
# =====================================================

flipped = df.copy()

# 1. Dynamically flip difference-based features
for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

# 2. Dynamically swap individual team metrics
for col1, col2 in SWAP_PAIRS:
    flipped[col1] = df[col2]
    flipped[col2] = df[col1]

# 3. Swap the result
flipped[TARGET] = 1 - flipped[TARGET]

augmented = pd.concat([df, flipped], ignore_index=True)

print(f"\nOriginal dataset: {len(df)}")
print(f"Augmented dataset: {len(augmented)}\n")

# =====================================================
# MULTI-SEED TRAINING & EVALUATION
# =====================================================

X = augmented[MODEL_FEATURES]
y = augmented[TARGET]

val_accuracies = []
test_accuracies = []
coefficients_list = []
intercepts_list = []

# Variables to store Seed 42 specific results
seed_42_val_acc = 0
seed_42_test_acc = 0
seed_42_coefs = None
seed_42_intercept = None

print(f"Running Logistic Regression across {NUM_RUNS} random seeds...")

for seed in SEEDS:
    # 1. Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=seed, stratify=y_temp
    )

    # 2. Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 3. Store Coefficients
    coefficients_list.append(model.coef_[0])
    intercepts_list.append(model.intercept_[0])

    # 4. Evaluate & Store Accuracies
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    v_acc = accuracy_score(y_val, val_preds)
    t_acc = accuracy_score(y_test, test_preds)

    val_accuracies.append(v_acc)
    test_accuracies.append(t_acc)

    # Capture Seed 42 specifically
    if seed == 42:
        seed_42_val_acc = v_acc
        seed_42_test_acc = t_acc
        seed_42_coefs = model.coef_[0]
        seed_42_intercept = model.intercept_[0]

# =====================================================
# AGGREGATE STATISTICS
# =====================================================

print("\n=====================================================")
print("RESULTS FOR SEED 42")
print("=====================================================")
print(f"Validation Accuracy: {seed_42_val_acc:.4f}")
print(f"Test Accuracy:       {seed_42_test_acc:.4f}")
print("Coefficients:")
for feature, coef in zip(MODEL_FEATURES, seed_42_coefs):
    print(f"  {feature}: {coef:.5f}")
print(f"Intercept: {seed_42_intercept:.5f}")

print("\n=====================================================")
print(f"RESULTS ACROSS {NUM_RUNS} SEEDS")
print("=====================================================")

print("\n--- VALIDATION ACCURACY ---")
print(f"Max:     {np.max(val_accuracies):.4f}")
print(f"Min:     {np.min(val_accuracies):.4f}")
print(f"Average: {np.mean(val_accuracies):.4f}")

print("\n--- TEST ACCURACY ---")
print(f"Max:     {np.max(test_accuracies):.4f}")
print(f"Min:     {np.min(test_accuracies):.4f}")
print(f"Average: {np.mean(test_accuracies):.4f}")

# Calculate average coefficients to build a "mean" model for the plots
avg_coefs = np.mean(coefficients_list, axis=0)
avg_intercept = np.mean(intercepts_list)

print("\n--- AVERAGE COEFFICIENTS ---")
for feature, coef in zip(MODEL_FEATURES, avg_coefs):
    print(f"  {feature}: {coef:.5f}")
print(f"Intercept: {avg_intercept:.5f}")

# Rebuild the scikit-learn model using the average weights for plotting
mean_model = LogisticRegression()
mean_model.classes_ = np.array([0, 1])
mean_model.coef_ = np.array([avg_coefs])
mean_model.intercept_ = np.array([avg_intercept])

# =====================================================
# EXAMPLE ODDS & ELO COMPARISON (Using Average Weights)
# =====================================================

print("\nExample Predictions (assuming neutral alts, xG, and humidity 65%)")
example_diffs = [-400, -200, 0, 200, 400]
n_examples = len(example_diffs)

example_df = pd.DataFrame({
    "elo_diff": example_diffs,
    "alt 1": [0] * n_examples,
    "alt 2": [0] * n_examples,
    "team 1 xg for": [0] * n_examples,
    "team 2 xg for": [0] * n_examples,
    "humid 1": [65] * n_examples,
    "humid 2": [65] * n_examples
})[MODEL_FEATURES]

example_probs = mean_model.predict_proba(example_df)[:, 1]

for diff, prob in zip(example_diffs, example_probs):
    print(f"Elo diff {diff:+4d} -> {prob:.3f}")

# Compare to Standard Elo
standard_elo_range = np.linspace(-600, 600, 1000)
elo_curve = 1 / (1 + 10 ** (-standard_elo_range / 400))

plot_df_comparison = pd.DataFrame({
    "elo_diff": standard_elo_range,
    "alt 1": np.zeros(1000),
    "alt 2": np.zeros(1000),
    "team 1 xg for": np.zeros(1000),
    "team 2 xg for": np.zeros(1000),
    "humid 1": np.full(1000, 65),
    "humid 2": np.full(1000, 65)
})[MODEL_FEATURES]

lr_curve = mean_model.predict_proba(plot_df_comparison)[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(standard_elo_range, elo_curve, label="Standard Elo Formula", linewidth=3)
plt.plot(standard_elo_range, lr_curve, label="Average Learned Logistic (Neutral Secondary Features)", linewidth=3)
plt.axvline(0, linestyle="--", color="black", alpha=0.5)

plt.xlabel("Elo Difference")
plt.ylabel("Win Probability")
plt.title(f"Standard Elo vs Average Learned Logistic ({NUM_RUNS} runs)")
plt.legend()
plt.show()
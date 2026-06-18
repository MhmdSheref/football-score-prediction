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

# All features are represented as Team1 - Team2 differences
DIFF_FEATURES = [
    "elo_diff",
    "alt_diff",
    "xg_diff",
    "humid_diff"
]

MODEL_FEATURES = [
    "elo_diff",
    "alt_diff",
    "xg_diff",
    "humid_diff"
]

TEST_SIZE = 0.30
VAL_SIZE = 0.5  # 50% of temp set

# Define the number of random seeds to try
NUM_RUNS = 50

# Generate a list of random seeds for reproducibility
SEEDS = [42 + i for i in range(NUM_RUNS)]

# =====================================================
# LOAD & CLEAN
# =====================================================

dfs = [pd.read_csv(file) for file in FILES]
df = pd.concat(dfs, ignore_index=True)

print(f"Combined rows: {len(df)}")

df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

print(f"Rows after cleaning: {len(df)}")

# =====================================================
# CREATE DIFFERENCE FEATURES
# =====================================================

df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]
df["alt_diff"] = df["alt 1"] - df["alt 2"]
df["xg_diff"] = df["team 1 xg for"] - df["team 2 xg for"]
df["humid_diff"] = df["humid 1"] - df["humid 2"]

# Remove rows with missing feature values
df = df.dropna(subset=MODEL_FEATURES)

print(f"Rows after feature cleaning: {len(df)}")

# =====================================================
# DATA AUGMENTATION
# =====================================================

flipped = df.copy()

for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

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

    # =================================================
    # SPLIT
    # =================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=y_temp
    )

    # =================================================
    # TRAIN
    # =================================================

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # Store coefficients
    coefficients_list.append(model.coef_[0])
    intercepts_list.append(model.intercept_[0])

    # =================================================
    # EVALUATE
    # =================================================

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    v_acc = accuracy_score(y_val, val_preds)
    t_acc = accuracy_score(y_test, test_preds)

    val_accuracies.append(v_acc)
    test_accuracies.append(t_acc)

    # Capture seed 42
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

print("\nCoefficients:")

for feature, coef in zip(MODEL_FEATURES, seed_42_coefs):
    print(f"  {feature}: {coef:.6f}")

print(f"Intercept: {seed_42_intercept:.6f}")

print("\n=====================================================")
print(f"RESULTS ACROSS {NUM_RUNS} SEEDS")
print("=====================================================")

print("\n--- VALIDATION ACCURACY ---")
print(f"Max:     {np.max(val_accuracies):.4f}")
print(f"Min:     {np.min(val_accuracies):.4f}")
print(f"Average: {np.mean(val_accuracies):.4f}")
print(f"Std Dev: {np.std(val_accuracies):.4f}")

print("\n--- TEST ACCURACY ---")
print(f"Max:     {np.max(test_accuracies):.4f}")
print(f"Min:     {np.min(test_accuracies):.4f}")
print(f"Average: {np.mean(test_accuracies):.4f}")
print(f"Std Dev: {np.std(test_accuracies):.4f}")

# =====================================================
# AVERAGE MODEL
# =====================================================

avg_coefs = np.mean(coefficients_list, axis=0)
avg_intercept = np.mean(intercepts_list)

print("\n--- AVERAGE COEFFICIENTS ---")

for feature, coef in zip(MODEL_FEATURES, avg_coefs):
    print(f"  {feature}: {coef:.6f}")

print(f"Intercept: {avg_intercept:.6f}")

# Build mean model manually
mean_model = LogisticRegression()

mean_model.classes_ = np.array([0, 1])
mean_model.coef_ = np.array([avg_coefs])
mean_model.intercept_ = np.array([avg_intercept])

# =====================================================
# EXAMPLE PREDICTIONS
# =====================================================

print("\nExample Predictions (neutral secondary differences)")

example_diffs = [-400, -200, 0, 200, 400]
n_examples = len(example_diffs)

example_df = pd.DataFrame({
    "elo_diff": example_diffs,
    "alt_diff": [0] * n_examples,
    "xg_diff": [0] * n_examples,
    "humid_diff": [0] * n_examples
})[MODEL_FEATURES]

example_probs = mean_model.predict_proba(example_df)[:, 1]

for diff, prob in zip(example_diffs, example_probs):
    print(f"Elo diff {diff:+4d} -> {prob:.3f}")

# =====================================================
# COMPARE TO STANDARD ELO
# =====================================================

standard_elo_range = np.linspace(-600, 600, 1000)

elo_curve = 1 / (1 + 10 ** (-standard_elo_range / 400))

plot_df_comparison = pd.DataFrame({
    "elo_diff": standard_elo_range,
    "alt_diff": np.zeros(1000),
    "xg_diff": np.zeros(1000),
    "humid_diff": np.zeros(1000)
})[MODEL_FEATURES]

lr_curve = mean_model.predict_proba(plot_df_comparison)[:, 1]

plt.figure(figsize=(10, 6))

plt.plot(
    standard_elo_range,
    elo_curve,
    label="Standard Elo Formula",
    linewidth=3
)

plt.plot(
    standard_elo_range,
    lr_curve,
    label=f"Average Logistic ({NUM_RUNS} runs)",
    linewidth=3
)

plt.axvline(
    0,
    linestyle="--",
    color="black",
    alpha=0.5
)

plt.xlabel("Elo Difference")
plt.ylabel("Win Probability")
plt.title("Standard Elo vs Learned Logistic")
plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# FEATURE IMPORTANCE SUMMARY
# =====================================================

print("\n=====================================================")
print("AVERAGE MODEL EQUATION")
print("=====================================================")

equation = f"logit(p) = {avg_intercept:.6f}"

for feature, coef in zip(MODEL_FEATURES, avg_coefs):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.6f}*{feature}"

print(equation)

print("\nProbability:")
print("p = 1 / (1 + exp(-logit(p)))")
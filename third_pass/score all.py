import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# CONFIGURATION
# Modify these variables to change the script's behavior
# =====================================================

FILES = ["2022_cleaned.csv", "2018_cleaned.csv"]
TARGET = "score advantage"

# Features negated during mirroring (Team 1 - Team 2)
DIFF_FEATURES = ["elo_diff"]

# Raw individual stats swapped during mirroring
SWAP_PAIRS = [("alt 1", "alt 2"), ("team 1 xg for", "team 2 xg for")]

# The final list of features to train the model on
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for"]

TEST_SIZE = 0.30
VAL_SIZE = 0.50
RANDOM_SEED = 42

# =====================================================
# LOAD DATA
# =====================================================

dfs = [pd.read_csv(file) for file in FILES]
df = pd.concat(dfs, ignore_index=True)

# =====================================================
# CLEAN
# =====================================================

# Ensure all necessary columns are present before dropping NAs
columns_to_check = ["team 1 elo", "team 2 elo", TARGET] + [col for pair in SWAP_PAIRS for col in pair]
df = df.dropna(subset=columns_to_check)

# =====================================================
# CREATE ELO DIFFERENCE
# =====================================================

df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

# =====================================================
# AUGMENT DATA
# =====================================================

flipped = df.copy()

# 1. Flip difference-based features
for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

# 2. Swap individual team metrics
for col1, col2 in SWAP_PAIRS:
    flipped[col1] = df[col2]
    flipped[col2] = df[col1]

# 3. Flip the target score advantage
flipped[TARGET] = -df[TARGET]

df_augmented = pd.concat([df, flipped], ignore_index=True)

print("Original rows:", len(df))
print("Augmented rows:", len(df_augmented))

# =====================================================
# VISUALIZE RAW DATA (Elo Diff only)
# =====================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df_augmented["elo_diff"],
    df_augmented[TARGET],
    alpha=0.5
)

plt.xlabel("Elo Difference")
plt.ylabel("Score Advantage")
plt.title("Raw Data")

plt.grid(True)
plt.show()

# =====================================================
# FEATURES / TARGET
# =====================================================

X = df_augmented[MODEL_FEATURES]
y = df_augmented[TARGET]

# =====================================================
# SPLIT
# =====================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED
)

# =====================================================
# TRAIN
# =====================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# =====================================================
# VISUALIZE FITTED LINE (Holding secondary features at 0)
# =====================================================

x_line = np.linspace(X["elo_diff"].min(), X["elo_diff"].max(), 1000)

# Create a DataFrame for prediction matching feature names and keeping secondary features neutral
plot_df = pd.DataFrame({
    "elo_diff": x_line,
    "alt 1": np.zeros(1000),
    "alt 2": np.zeros(1000),
    "team 1 xg for": np.zeros(1000),
    "team 2 xg for": np.zeros(1000)
})[MODEL_FEATURES]

y_line = model.predict(plot_df)

plt.figure(figsize=(12, 7))

plt.scatter(
    X_train["elo_diff"],
    y_train,
    alpha=0.4,
    label="Training Matches"
)

plt.plot(
    x_line,
    y_line,
    linewidth=3,
    color="orange",
    label="Regression Line (Secondary Features=0)"
)

plt.axhline(0, linestyle="--", alpha=0.5, color="black")
plt.axvline(0, linestyle="--", alpha=0.5, color="black")

plt.xlabel("Elo Difference (with Alt and xG features held at 0)")
plt.ylabel("Score Advantage")
plt.title("Score Advantage vs Elo Difference")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# EVALUATION METRICS HELPER
# =====================================================

def evaluate_regression(X_data, y_data, dataset_name):
    preds = model.predict(X_data)
    print(f"\n{dataset_name.upper()}")
    print(f"MAE:  {mean_absolute_error(y_data, preds):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_data, preds)):.3f}")
    print(f"R²:   {r2_score(y_data, preds):.3f}")
    return preds

val_preds = evaluate_regression(X_val, y_val, "Validation")
test_preds = evaluate_regression(X_test, y_test, "Test")

# =====================================================
# ACTUAL VS PREDICTED (Test Set)
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    test_preds,
    alpha=0.6
)

minimum = min(y_test.min(), test_preds.min())
maximum = max(y_test.max(), test_preds.max())

plt.plot([minimum, maximum], [minimum, maximum], "--", color="black")

plt.xlabel("Actual Score Advantage")
plt.ylabel("Predicted Score Advantage")
plt.title("Actual vs Predicted (Test Set)")

plt.grid(True)
plt.show()

# =====================================================
# MODEL EQUATION (Unscaling dynamically)
# =====================================================

regressor = model.named_steps["regressor"]
scaler = model.named_steps["scaler"]

# Unscale the coefficients for all features dynamically
unscaled_coefs = regressor.coef_ / scaler.scale_

# Unscale the intercept
unscaled_intercept = regressor.intercept_ - np.sum(unscaled_coefs * scaler.mean_)

print("\nModel Equation")
equation = f"score_advantage = {unscaled_intercept:.6f}"
for feat, coef in zip(MODEL_FEATURES, unscaled_coefs):
    equation += f"\n    {coef:+.6f} * {feat}"

print(equation)

# =====================================================
# EXAMPLE PREDICTIONS
# =====================================================

print("\nExample Predictions (assuming neutral alts and xG)")
example_diffs = [-400, -200, 0, 200, 400]
n_examples = len(example_diffs)

example_df = pd.DataFrame({
    "elo_diff": example_diffs,
    "alt 1": [0]*n_examples,
    "alt 2": [0]*n_examples,
    "team 1 xg for": [0]*n_examples,
    "team 2 xg for": [0]*n_examples
})[MODEL_FEATURES]

example_preds = model.predict(example_df)

for diff, pred in zip(example_diffs, example_preds):
    print(f"Elo diff {diff:+4d} -> Predicted score advantage = {pred:.2f}")
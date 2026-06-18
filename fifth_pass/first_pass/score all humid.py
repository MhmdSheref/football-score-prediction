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
# =====================================================

FILES = ["2022_humid_clean.csv", "2018_humid_clean.csv"]
TARGET = "score advantage"

# All features represented as Team 1 - Team 2
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
VAL_SIZE = 0.50

# Define the number of random seeds to try
NUM_RUNS = 50

# Generate a list of random seeds for reproducibility
SEEDS = [42 + i for i in range(NUM_RUNS)]

# =====================================================
# LOAD & CLEAN
# =====================================================

dfs = [pd.read_csv(file) for file in FILES]
df = pd.concat(dfs, ignore_index=True)

columns_to_check = [
    "team 1 elo",
    "team 2 elo",
    "alt 1",
    "alt 2",
    "team 1 xg for",
    "team 2 xg for",
    "humid 1",
    "humid 2",
    TARGET
]

df = df.dropna(subset=columns_to_check)

# =====================================================
# CREATE DIFFERENCE FEATURES
# =====================================================

df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]
df["alt_diff"] = df["alt 1"] - df["alt 2"]
df["xg_diff"] = df["team 1 xg for"] - df["team 2 xg for"]
df["humid_diff"] = df["humid 1"] - df["humid 2"]

# =====================================================
# AUGMENT DATA
# =====================================================

flipped = df.copy()

for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

# Score advantage changes sign when teams are swapped
flipped[TARGET] = -df[TARGET]

df_augmented = pd.concat([df, flipped], ignore_index=True)

print("Original rows:", len(df))
print("Augmented rows:", len(df_augmented))

# =====================================================
# MULTI-SEED TRAINING & EVALUATION
# =====================================================

X = df_augmented[MODEL_FEATURES]
y = df_augmented[TARGET]

val_maes = []
val_rmses = []
val_r2s = []

test_maes = []
test_rmses = []
test_r2s = []

unscaled_coefs_list = []
unscaled_intercepts_list = []

# Variables to store Seed 42 specific results
seed_42_metrics = {}
seed_42_coefs = None
seed_42_intercept = None

print(f"\nRunning Linear Regression across {NUM_RUNS} random seeds...")

for seed in SEEDS:

    # =================================================
    # SPLIT
    # =================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE,
        random_state=seed
    )

    # =================================================
    # TRAIN
    # =================================================

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)

    # =================================================
    # EVALUATE
    # =================================================

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    v_mae = mean_absolute_error(y_val, val_preds)
    v_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    v_r2 = r2_score(y_val, val_preds)

    t_mae = mean_absolute_error(y_test, test_preds)
    t_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    t_r2 = r2_score(y_test, test_preds)

    val_maes.append(v_mae)
    val_rmses.append(v_rmse)
    val_r2s.append(v_r2)

    test_maes.append(t_mae)
    test_rmses.append(t_rmse)
    test_r2s.append(t_r2)

    # =================================================
    # UN-SCALE COEFFICIENTS
    # =================================================

    regressor = model.named_steps["regressor"]
    scaler = model.named_steps["scaler"]

    unscaled_coef = regressor.coef_ / scaler.scale_
    unscaled_intercept = (
        regressor.intercept_
        - np.sum(unscaled_coef * scaler.mean_)
    )

    unscaled_coefs_list.append(unscaled_coef)
    unscaled_intercepts_list.append(unscaled_intercept)

    # =================================================
    # STORE SEED 42 RESULTS
    # =================================================

    if seed == 42:
        seed_42_metrics = {
            "val_mae": v_mae,
            "val_rmse": v_rmse,
            "val_r2": v_r2,
            "test_mae": t_mae,
            "test_rmse": t_rmse,
            "test_r2": t_r2
        }

        seed_42_coefs = unscaled_coef
        seed_42_intercept = unscaled_intercept

        X_train_42 = X_train.copy()
        y_train_42 = y_train.copy()

# =====================================================
# AGGREGATE STATISTICS
# =====================================================

print("\n=====================================================")
print("RESULTS FOR SEED 42")
print("=====================================================")

print(
    f"Validation -> MAE: {seed_42_metrics['val_mae']:.3f} | "
    f"RMSE: {seed_42_metrics['val_rmse']:.3f} | "
    f"R²: {seed_42_metrics['val_r2']:.3f}"
)

print(
    f"Test       -> MAE: {seed_42_metrics['test_mae']:.3f} | "
    f"RMSE: {seed_42_metrics['test_rmse']:.3f} | "
    f"R²: {seed_42_metrics['test_r2']:.3f}"
)

print("\nEquation:")

print(f"score_advantage = {seed_42_intercept:.6f}")

for feat, coef in zip(MODEL_FEATURES, seed_42_coefs):
    print(f"    {coef:+.6f} * {feat}")

print("\n=====================================================")
print(f"RESULTS ACROSS {NUM_RUNS} SEEDS")
print("=====================================================")

print(
    f"Validation MAE: {np.mean(val_maes):.3f} "
    f"(Min: {np.min(val_maes):.3f}, Max: {np.max(val_maes):.3f})"
)

print(
    f"Validation RMSE: {np.mean(val_rmses):.3f}"
)

print(
    f"Validation R²: {np.mean(val_r2s):.3f}"
)

print(
    f"Test MAE: {np.mean(test_maes):.3f} "
    f"(Min: {np.min(test_maes):.3f}, Max: {np.max(test_maes):.3f})"
)

print(
    f"Test RMSE: {np.mean(test_rmses):.3f}"
)

print(
    f"Test R²: {np.mean(test_r2s):.3f}"
)

# =====================================================
# AVERAGE MODEL
# =====================================================

avg_coefs = np.mean(unscaled_coefs_list, axis=0)
avg_intercept = np.mean(unscaled_intercepts_list)

print("\nAverage Model Equation:")
print(f"score_advantage = {avg_intercept:.6f}")

for feat, coef in zip(MODEL_FEATURES, avg_coefs):
    print(f"    {coef:+.6f} * {feat}")

# =====================================================
# HELPER FUNCTION
# =====================================================

def predict_avg_model(df_input):
    return avg_intercept + np.dot(df_input[MODEL_FEATURES], avg_coefs)

# =====================================================
# VISUALIZE FITTED LINE
# =====================================================

x_line = np.linspace(
    X["elo_diff"].min(),
    X["elo_diff"].max(),
    1000
)

plot_df = pd.DataFrame({
    "elo_diff": x_line,
    "alt_diff": np.zeros(1000),
    "xg_diff": np.zeros(1000),
    "humid_diff": np.zeros(1000)
})[MODEL_FEATURES]

y_line = predict_avg_model(plot_df)

plt.figure(figsize=(12, 7))

plt.scatter(
    X_train_42["elo_diff"],
    y_train_42,
    alpha=0.4,
    label="Training Matches (Seed 42 Split)"
)

plt.plot(
    x_line,
    y_line,
    linewidth=3,
    color="orange",
    label="Average Regression Line"
)

plt.axhline(
    0,
    linestyle="--",
    alpha=0.5,
    color="black"
)

plt.axvline(
    0,
    linestyle="--",
    alpha=0.5,
    color="black"
)

plt.xlabel(
    "Elo Difference (altitude, xG and humidity differences held at 0)"
)

plt.ylabel("Score Advantage")

plt.title(
    f"Score Advantage vs Elo Difference ({NUM_RUNS} runs averaged)"
)

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# EXAMPLE PREDICTIONS
# =====================================================

print(
    "\nExample Predictions "
    "(using Average Model, all secondary differences = 0)"
)

example_diffs = [-400, -200, 0, 200, 400]

example_df = pd.DataFrame({
    "elo_diff": example_diffs,
    "alt_diff": [0] * len(example_diffs),
    "xg_diff": [0] * len(example_diffs),
    "humid_diff": [0] * len(example_diffs)
})[MODEL_FEATURES]

example_preds = predict_avg_model(example_df)

for diff, pred in zip(example_diffs, example_preds):
    print(
        f"Elo diff {diff:+4d} -> "
        f"Predicted score advantage = {pred:.2f}"
    )
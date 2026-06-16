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

FILES = ["2022_xg_alt.csv", "2018_xg_alt.csv"]
TARGET = "score advantage"

# Features negated during mirroring (Team 1 - Team 2)
DIFF_FEATURES = ["elo_diff", "alt_diff", "xg_diff"]

# The final list of features to train the model on
MODEL_FEATURES = ["elo_diff", "alt_diff", "xg_diff"]

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

# Ensure all necessary base columns are present before dropping NAs
columns_to_check = [
    "team 1 elo", "team 2 elo",
    "alt 1", "alt 2",
    "team 1 xg for", "team 2 xg for",
    TARGET
]
df = df.dropna(subset=columns_to_check)

# =====================================================
# CREATE DIFFERENCE FEATURES
# =====================================================

df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]
df["alt_diff"] = df["alt 1"] - df["alt 2"]
df["xg_diff"]  = df["team 1 xg for"] - df["team 2 xg for"]

# =====================================================
# AUGMENT DATA
# =====================================================

flipped = df.copy()

# 1. Flip difference-based features
for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]

# 2. Flip the target score advantage
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

# Create a DataFrame for prediction matching feature names and keeping secondary diffs neutral
plot_df = pd.DataFrame({
    "elo_diff": x_line,
    "alt_diff": np.zeros(1000),
    "xg_diff": np.zeros(1000)
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
    label="Regression Line (Secondary Diff Features=0)"
)

plt.axhline(0, linestyle="--", alpha=0.5, color="black")
plt.axvline(0, linestyle="--", alpha=0.5, color="black")

plt.xlabel("Elo Difference (with Alt and xG diffs held at 0)")
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

print("\nExample Predictions (assuming neutral alt and xG diffs)")
example_diffs = [-400, -200, 0, 200, 400]
n_examples = len(example_diffs)

example_df = pd.DataFrame({
    "elo_diff": example_diffs,
    "alt_diff": [0]*n_examples,
    "xg_diff": [0]*n_examples
})[MODEL_FEATURES]

example_preds = model.predict(example_df)

for diff, pred in zip(example_diffs, example_preds):
    print(f"Elo diff {diff:+4d} -> Predicted score advantage = {pred:.2f}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# =====================================================
# 1. VISUALIZE 3D INPUTS + OUTPUT (4D Mapping)
# =====================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z are the inputs, Color is the target output
sc = ax.scatter(
    X_train["elo_diff"],
    X_train["alt_diff"],
    X_train["xg_diff"],
    c=y_train,  # Color mapped to Score Advantage
    cmap='coolwarm',  # Blue for negative, Red for positive
    alpha=0.6,
    s=30
)

ax.set_xlabel('Elo Difference')
ax.set_ylabel('Alt Difference')
ax.set_zlabel('xG Difference')
plt.title('3D Input Features vs Score Advantage (Color)')

# Add a color bar to indicate the score advantage scale
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Score Advantage')

plt.show()

# =====================================================
# 2. INDIVIDUAL GRAPHS (Data vs Projected Line)
# =====================================================
# Set up a figure with 3 side-by-side subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Elo Difference", "Altitude Difference", "xG Difference"]

for i, feature in enumerate(MODEL_FEATURES):
    ax = axes[i]

    # 1. Create a dynamic line space for the current feature
    x_line = np.linspace(X[feature].min(), X[feature].max(), 500)

    # 2. Create a DataFrame keeping all features at 0
    plot_df = pd.DataFrame(np.zeros((500, len(MODEL_FEATURES))), columns=MODEL_FEATURES)

    # 3. Inject the varying current feature
    plot_df[feature] = x_line

    # 4. Predict the line
    y_line = model.predict(plot_df)

    # Plot training data
    ax.scatter(
        X_train[feature],
        y_train,
        alpha=0.3,
        color='gray',
        label="Training Matches"
    )

    # Plot the projected regression line
    ax.plot(
        x_line,
        y_line,
        linewidth=3,
        color="orange",
        label="Projected Line (Others=0)"
    )

    # Formatting
    ax.axhline(0, linestyle="--", alpha=0.5, color="black")
    ax.axvline(0, linestyle="--", alpha=0.5, color="black")
    ax.set_xlabel(f"{titles[i]} (with others held at 0)")

    # Only show the Y label on the first graph to keep it clean
    if i == 0:
        ax.set_ylabel("Score Advantage")

    ax.set_title(f"Score Adv vs {titles[i]}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
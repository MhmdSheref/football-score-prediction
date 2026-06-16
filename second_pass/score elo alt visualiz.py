import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# CONFIGURATION
# Modify these variables to change the script's behavior
# =====================================================

FILES = ["2022_alt.csv", "2018_alt.csv"]
TARGET = "score advantage"

# Features negated during mirroring (Team 1 - Team 2)
DIFF_FEATURES = ["elo_diff"]

# Raw individual stats swapped during mirroring
SWAP_PAIRS = [("alt 1", "alt 2")]

# The final list of features to train the model on
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2"]

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

print(f"Original rows: {len(df)}")
print(f"Augmented rows: {len(df_augmented)}")

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
# VISUALIZE: MULTI-DIMENSIONAL BAND
# =====================================================
# Instead of a single line holding values at 0, we plot the
# model's true predictions for the actual training data.

y_train_preds = model.predict(X_train)

plt.figure(figsize=(12, 7))

# Actual Target values
plt.scatter(
    X_train["elo_diff"],
    y_train,
    alpha=0.3,
    color="blue",
    label="Actual Match Outcomes"
)

# Model Predictions (Creates a band instead of a line due to Alts)
plt.scatter(
    X_train["elo_diff"],
    y_train_preds,
    alpha=0.8,
    color="orange",
    s=15,
    label="Model Predictions (Multi-dimensional plane)"
)

plt.axhline(0, linestyle="--", alpha=0.5, color="black")
plt.axvline(0, linestyle="--", alpha=0.5, color="black")

plt.xlabel("Elo Difference")
plt.ylabel("Score Advantage")
plt.title("Score Advantage vs Elo Difference (True Multi-dimensional Variance)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =====================================================
# VISUALIZE: PCA MAPPING (All dimensions mapped to 2D)
# =====================================================
# Here we compress all 3 features down to 2 mathematical axes
# so we can view the entire feature space at once.

# Extract the scaled features from the pipeline
scaler = model.named_steps["scaler"]
X_train_scaled = scaler.transform(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(10, 8))

# Scatter plot of PC1 vs PC2, colored by the model's predicted advantage
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_train_preds,
    cmap="coolwarm",
    alpha=0.8,
    edgecolor="k"
)

cbar = plt.colorbar(scatter)
cbar.set_label("Predicted Score Advantage", rotation=270, labelpad=15)

plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA Projection: All Features Mapped to 2D Space")

plt.grid(True, alpha=0.3)
plt.show()

# =====================================================
# EVALUATION METRICS HELPER
# =====================================================

def evaluate_regression(X_data, y_data, dataset_name):
    preds = model.predict(X_data)
    print(f"\n--- {dataset_name.upper()} ---")
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
    alpha=0.6,
    color="purple"
)

minimum = min(y_test.min(), test_preds.min())
maximum = max(y_test.max(), test_preds.max())

plt.plot([minimum, maximum], [minimum, maximum], "--", color="black", linewidth=2)

plt.xlabel("Actual Score Advantage")
plt.ylabel("Predicted Score Advantage")
plt.title("Actual vs Predicted (Test Set)")

plt.grid(True, alpha=0.3)
plt.show()

# =====================================================
# MODEL EQUATION
# =====================================================

regressor = model.named_steps["regressor"]

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
# EXAMPLE PREDICTIONS (Using Real Sample Data)
# =====================================================

print("\nExample Predictions (Using real rows from Test set instead of zeros)")

# Take 5 random samples from the test set to show how it handles all dimensions natively
sample_X = X_test.sample(5, random_state=RANDOM_SEED)
sample_y = y_test.loc[sample_X.index]
sample_preds = model.predict(sample_X)

for idx, (index, row) in enumerate(sample_X.iterrows()):
    print(f"\nSample {idx+1}:")
    print(f"  Elo Diff: {row['elo_diff']:+6.1f} | Alt 1: {row['alt 1']:6.1f} | Alt 2: {row['alt 2']:6.1f}")
    print(f"  Predicted Advantage: {sample_preds[idx]:+.2f} (Actual: {sample_y.loc[index]:+.2f})")
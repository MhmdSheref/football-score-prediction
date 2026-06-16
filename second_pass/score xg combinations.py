import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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

FILES = ["2022_xg.csv", "2018_xg.csv"]  # Update with your actual file names
TARGET = "score advantage"
TEST_SIZE = 0.30
VAL_SIZE = 0.50
RANDOM_SEED = 42

# Toggle this to False if you want to skip straight to the final results
SHOW_INDIVIDUAL_PLOTS = True

FEATURE_GROUPS = {
    "Elo": {
        "diff": ["elo_diff"],
        "swap": []
    },
    "xG_For": {
        "diff": [],
        "swap": [("team 1 xg for", "team 2 xg for")]
    },
    "xG_Against": {
        "diff": [],
        "swap": [("team 1 xg against", "team 2 xg against")]
    },
    "xG_Diff": {
        "diff": [],
        "swap": [("team 1 xg diff", "team 2 xg diff")]
    }
}

# =====================================================
# LOAD & INITIAL CLEANING
# =====================================================

print("Loading data...")
dfs = [pd.read_csv(file) for file in FILES]
df = pd.concat(dfs, ignore_index=True)

df = df.dropna(subset=[TARGET])

# Create the standard elo_diff
df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

print(f"Total valid rows loaded: {len(df)}")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def augment_dataset(data, diff_feats, swap_pairs):
    flipped = data.copy()

    # 1. Flip difference-based features
    for feature in diff_feats:
        flipped[feature] = -flipped[feature]

    # 2. Swap individual team metrics
    for col1, col2 in swap_pairs:
        flipped[col1] = data[col2]
        flipped[col2] = data[col1]

    # 3. Flip the target score advantage (Team A winning by 2 means Team B lost by 2)
    flipped[TARGET] = -data[TARGET]

    return pd.concat([data, flipped], ignore_index=True)


def evaluate_regression(model, X_data, y_data, dataset_name, combo_name):
    preds = model.predict(X_data)
    mae = mean_absolute_error(y_data, preds)
    rmse = np.sqrt(mean_squared_error(y_data, preds))
    r2 = r2_score(y_data, preds)

    print(f"--- {dataset_name.upper()} EVALUATION ---")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    if SHOW_INDIVIDUAL_PLOTS:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_data, preds, alpha=0.4)

        # Draw the perfect prediction diagonal line
        min_val = min(y_data.min(), preds.min())
        max_val = max(y_data.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], "--", color="black", label="Perfect Prediction")

        plt.xlabel("Actual Score Advantage")
        plt.ylabel("Predicted Score Advantage")
        plt.title(f"{combo_name}\nActual vs Predicted ({dataset_name})")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.show()

    return mae, rmse, r2


def print_unscaled_equation(model, features):
    regressor = model.named_steps["regressor"]
    scaler = model.named_steps["scaler"]

    unscaled_coefs = regressor.coef_ / scaler.scale_
    unscaled_intercept = regressor.intercept_ - np.sum(unscaled_coefs * scaler.mean_)

    equation = f"Score Advantage = {unscaled_intercept:.4f}"
    for feat, coef in zip(features, unscaled_coefs):
        equation += f"\n    {coef:+.4f} * {feat}"
    print("\nModel Equation (Unscaled):")
    print(equation + "\n")


# =====================================================
# MAIN LOOP: ITERATING COMBINATIONS
# =====================================================

group_names = list(FEATURE_GROUPS.keys())
results = []

for r in range(1, len(group_names) + 1):
    for combo in itertools.combinations(group_names, r):
        combo_name = " + ".join(combo)
        print("\n" + "=" * 70)
        print(f"TESTING COMBINATION: {combo_name}")
        print("=" * 70)

        # 1. Compile the features for this combination
        current_diff = []
        current_swap = []

        for group in combo:
            current_diff.extend(FEATURE_GROUPS[group]["diff"])
            current_swap.extend(FEATURE_GROUPS[group]["swap"])

        current_features = current_diff.copy()
        for c1, c2 in current_swap:
            current_features.extend([c1, c2])

        # Drop NAs dynamically for the current feature set
        columns_to_check = current_features + [TARGET]
        df_clean = df.dropna(subset=columns_to_check)

        # 2. Augment Data
        augmented = augment_dataset(df_clean, current_diff, current_swap)

        # 3. Train/Test Split
        X = augmented[current_features]
        y = augmented[TARGET]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED
        )

        # 4. Train Pipeline (Scaler + Regressor)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        model.fit(X_train, y_train)

        # Print Equation
        print_unscaled_equation(model, current_features)

        # 5. Evaluate
        val_mae, val_rmse, val_r2 = evaluate_regression(model, X_val, y_val, "Validation", combo_name)
        test_mae, test_rmse, test_r2 = evaluate_regression(model, X_test, y_test, "Test", combo_name)

        # 6. Plot Regression Line vs Elo Diff (Only if Elo is in the feature set)
        if "elo_diff" in current_features and SHOW_INDIVIDUAL_PLOTS:
            x_line = np.linspace(X["elo_diff"].min(), X["elo_diff"].max(), 1000)
            plot_df = pd.DataFrame(0, index=np.arange(1000), columns=current_features)
            plot_df["elo_diff"] = x_line

            y_line = model.predict(plot_df)

            plt.figure(figsize=(8, 5))
            plt.scatter(X_train["elo_diff"], y_train, alpha=0.15, label="Training Matches")
            plt.plot(x_line, y_line, linewidth=3, color="orange", label="Regression Line (Other features = 0)")

            plt.axhline(0, linestyle="--", alpha=0.5, color="black")
            plt.axvline(0, linestyle="--", alpha=0.5, color="black")
            plt.xlabel("Elo Difference")
            plt.ylabel("Score Advantage")
            plt.title(f"Score Advantage vs Elo Difference: {combo_name}")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Save results for final comparison
        results.append({
            "Combination": combo_name,
            "Features": len(current_features),
            "Val_MAE": val_mae,
            "Val_RMSE": val_rmse,
            "Val_R2": val_r2,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2
        })

# =====================================================
# FINAL COMPARISON GRAPHS
# =====================================================

print("\n" + "=" * 70)
print("FINAL RESULTS COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
# Sort by Test MAE ascending (Lower error is better)
results_df = results_df.sort_values(by="Test_MAE", ascending=True)

# Graph 1: Test MAE Bar Chart (Sorted)
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(results_df))
plt.barh(y_pos, results_df["Test_MAE"], color="salmon", edgecolor="black")
plt.yticks(y_pos, results_df["Combination"])
plt.xlabel("Test Mean Absolute Error (Goals)")
plt.title("Model Error by Feature Combination (Lower is Better)")

# Add text labels inside the bars
for i, mae in enumerate(results_df["Test_MAE"]):
    plt.text(mae - 0.05, i, f"{mae:.3f}", va="center", ha="right", color="white", fontweight="bold")

plt.tight_layout()
plt.show()

# Graph 2: Validation vs Test MAE Scatter Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    results_df["Val_MAE"],
    results_df["Test_MAE"],
    c=results_df["Features"],
    cmap="coolwarm",
    s=120,
    edgecolors="black"
)

# Plot a generic diagonal line to check for overfitting/underfitting
min_val = min(results_df["Val_MAE"].min(), results_df["Test_MAE"].min())
max_val = max(results_df["Val_MAE"].max(), results_df["Test_MAE"].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Perfect Generalization")

# Annotate points
for i, row in results_df.iterrows():
    plt.annotate(
        row["Combination"],
        (row["Val_MAE"], row["Test_MAE"]),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=8,
        alpha=0.8
    )

plt.xlabel("Validation MAE")
plt.ylabel("Test MAE")
plt.title("Validation vs. Test MAE (Color = Feature Count)")
plt.colorbar(scatter, label="Number of Model Features")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Print the final leaderboard
print(results_df[[
    "Combination",
    "Val_MAE",
    "Val_RMSE",
    "Val_R2",
    "Test_MAE",
    "Test_RMSE",
    "Test_R2"
]].to_string(index=False))
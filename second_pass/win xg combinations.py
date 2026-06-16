import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
# =====================================================

FILES = ["2022_xg.csv", "2018_xg.csv"]  # Update with your actual file names
TARGET = "win"
TEST_SIZE = 0.30
VAL_SIZE = 0.50
RANDOM_SEED = 42

# Toggle this to False if you don't want 30+ plots pausing your script
SHOW_INDIVIDUAL_PLOTS = True

# Define the feature groupings.
# "diff" = features we flip by multiplying by -1
# "swap" = features we flip by trading Team 1 and Team 2 values
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
df[TARGET] = df[TARGET].astype(int)

# Create the standard elo_diff
df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

print(f"Total valid rows loaded: {len(df)}")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def augment_dataset(data, diff_feats, swap_pairs):
    flipped = data.copy()

    for feature in diff_feats:
        flipped[feature] = -flipped[feature]

    for col1, col2 in swap_pairs:
        flipped[col1] = data[col2]
        flipped[col2] = data[col1]

    flipped[TARGET] = 1 - flipped[TARGET]

    return pd.concat([data, flipped], ignore_index=True)


def evaluate_model(model, X_data, y_data, dataset_name, combo_name):
    preds = model.predict(X_data)
    acc = accuracy_score(y_data, preds)

    print(f"--- {dataset_name.upper()} EVALUATION ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_data, preds))

    if SHOW_INDIVIDUAL_PLOTS:
        disp = ConfusionMatrixDisplay(confusion_matrix(y_data, preds))
        disp.plot(cmap="Blues")
        plt.title(f"{combo_name} - {dataset_name} Confusion Matrix")
        plt.show()

    return acc


# =====================================================
# MAIN LOOP: ITERATING COMBINATIONS
# =====================================================

group_names = list(FEATURE_GROUPS.keys())
results = []

# Loop through lengths 1 to 4 (all single features, pairs, triplets, and the whole set)
for r in range(1, len(group_names) + 1):
    for combo in itertools.combinations(group_names, r):
        combo_name = " + ".join(combo)
        print("\n" + "=" * 60)
        print(f"TESTING COMBINATION: {combo_name}")
        print("=" * 60)

        # 1. Compile the features for this combination
        current_diff = []
        current_swap = []

        for group in combo:
            current_diff.extend(FEATURE_GROUPS[group]["diff"])
            current_swap.extend(FEATURE_GROUPS[group]["swap"])

        current_features = current_diff.copy()
        for c1, c2 in current_swap:
            current_features.extend([c1, c2])

        # 2. Augment Data
        augmented = augment_dataset(df, current_diff, current_swap)

        # 3. Train/Test Split
        X = augmented[current_features]
        y = augmented[TARGET]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp
        )

        # 4. Train Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        print("\nCoefficients:")
        for feature, coef in zip(current_features, model.coef_[0]):
            print(f"  {feature}: {coef:.5f}")
        print(f"Intercept: {model.intercept_[0]:.5f}\n")

        # 5. Evaluate
        val_acc = evaluate_model(model, X_val, y_val, "Validation", combo_name)
        test_acc = evaluate_model(model, X_test, y_test, "Test", combo_name)

        # 6. Plot Probability Curve (Only if Elo is in the feature set)
        if "elo_diff" in current_features and SHOW_INDIVIDUAL_PLOTS:
            elo_range = np.linspace(X["elo_diff"].min(), X["elo_diff"].max(), 1000)
            plot_df = pd.DataFrame(0, index=np.arange(1000), columns=current_features)
            plot_df["elo_diff"] = elo_range

            probs = model.predict_proba(plot_df)[:, 1]

            plt.figure(figsize=(8, 5))
            plt.plot(elo_range, probs, linewidth=3, color="orange")
            plt.axvline(0, linestyle="--", color="black", alpha=0.5)
            plt.xlabel("Elo Difference (Other features held at 0)")
            plt.ylabel("P(Team 1 Wins)")
            plt.title(f"Win Probability Curve: {combo_name}")
            plt.show()

        # Save results for final comparison
        results.append({
            "Combination": combo_name,
            "Features": len(current_features),
            "Val_Accuracy": val_acc,
            "Test_Accuracy": test_acc
        })

# =====================================================
# FINAL COMPARISON GRAPHS
# =====================================================

print("\n" + "=" * 60)
print("FINAL RESULTS COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Test_Accuracy", ascending=True)

# Graph 1: Test Accuracy Bar Chart (Sorted)
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(results_df))
plt.barh(y_pos, results_df["Test_Accuracy"], color="skyblue", edgecolor="black")
plt.yticks(y_pos, results_df["Combination"])
plt.xlabel("Test Accuracy")
plt.title("Model Accuracy by Feature Combination")

# Add text labels inside the bars
for i, acc in enumerate(results_df["Test_Accuracy"]):
    plt.text(acc - 0.05, i, f"{acc:.4f}", va="center", color="black", fontweight="bold")

plt.xlim(0.5, 1.0)  # Zoom in on the relevant accuracy range (adjust if needed)
plt.tight_layout()
plt.show()

# Graph 2: Validation vs Test Accuracy Scatter Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    results_df["Val_Accuracy"],
    results_df["Test_Accuracy"],
    c=results_df["Features"],
    cmap="viridis",
    s=100,
    edgecolors="black"
)
plt.plot([0.5, 1], [0.5, 1], linestyle="--", color="gray", label="Perfect Generalization")

# Annotate points
for i, row in results_df.iterrows():
    plt.annotate(
        row["Combination"],
        (row["Val_Accuracy"], row["Test_Accuracy"]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        alpha=0.7
    )

plt.xlabel("Validation Accuracy")
plt.ylabel("Test Accuracy")
plt.title("Validation vs. Test Accuracy (Color = Feature Count)")
plt.colorbar(scatter, label="Number of Features in Model")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Print the final leaderboard
print(results_df[["Combination", "Val_Accuracy", "Test_Accuracy"]].sort_values(by="Test_Accuracy",
                                                                               ascending=False).to_string(index=False))
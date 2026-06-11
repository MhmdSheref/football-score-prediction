import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# 1. LOAD DATA
# =====================================================

print("=" * 50)
print("LOADING DATA")
print("=" * 50)

df1 = pd.read_csv("2022.csv")
df2 = pd.read_csv("2018.csv")

df = pd.concat([df1, df2], ignore_index=True)

print(f"Combined rows: {len(df)}")

# =====================================================
# 2. CLEAN DATA
# =====================================================

print("\n" + "=" * 50)
print("CLEANING DATA")
print("=" * 50)

df = df.dropna(subset=["win"])

df["win"] = df["win"].astype(int)

print(f"Rows after removing missing results: {len(df)}")

print("\nClass Distribution:")
print(df["win"].value_counts())

# =====================================================
# 3. DATA AUGMENTATION
# =====================================================

print("\n" + "=" * 50)
print("AUGMENTING DATA")
print("=" * 50)

flipped = df.copy()

flipped["team 1 elo"] = df["team 2 elo"]
flipped["team 2 elo"] = df["team 1 elo"]

# Flip result
flipped["win"] = 1 - df["win"]

df_augmented = pd.concat(
    [df, flipped],
    ignore_index=True
)

print(f"Original samples: {len(df)}")
print(f"Augmented samples: {len(df_augmented)}")

# =====================================================
# 4. VISUALIZE ORIGINAL DATA
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    df[df["win"] == 1]["team 1 elo"],
    df[df["win"] == 1]["team 2 elo"],
    alpha=0.7,
    label="Team 1 Won"
)

plt.scatter(
    df[df["win"] == 0]["team 1 elo"],
    df[df["win"] == 0]["team 2 elo"],
    alpha=0.7,
    label="Team 1 Lost"
)

plt.xlabel("Team 1 Elo")
plt.ylabel("Team 2 Elo")
plt.title("Original Dataset")
plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# 5. VISUALIZE AUGMENTED DATA
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    df_augmented[df_augmented["win"] == 1]["team 1 elo"],
    df_augmented[df_augmented["win"] == 1]["team 2 elo"],
    alpha=0.5,
    label="Team 1 Won"
)

plt.scatter(
    df_augmented[df_augmented["win"] == 0]["team 1 elo"],
    df_augmented[df_augmented["win"] == 0]["team 2 elo"],
    alpha=0.5,
    label="Team 1 Lost"
)

plt.xlabel("Team 1 Elo")
plt.ylabel("Team 2 Elo")
plt.title("Augmented Dataset")
plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# 6. FEATURES AND TARGET
# =====================================================

X = df_augmented[
    ["team 1 elo", "team 2 elo"]
]

y = df_augmented["win"]

# =====================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

print("\n" + "=" * 50)
print("DATA SPLITS")
print("=" * 50)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"Train size:      {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size:       {len(X_test)}")

# =====================================================
# 8. VISUALIZE SPLITS
# =====================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [
    (X_train, y_train, "Training Set"),
    (X_val, y_val, "Validation Set"),
    (X_test, y_test, "Test Set")
]

for ax, (X_set, y_set, title) in zip(axes, datasets):

    ax.scatter(
        X_set[y_set == 1]["team 1 elo"],
        X_set[y_set == 1]["team 2 elo"],
        alpha=0.6,
        label="Win"
    )

    ax.scatter(
        X_set[y_set == 0]["team 1 elo"],
        X_set[y_set == 0]["team 2 elo"],
        alpha=0.6,
        label="Loss"
    )

    ax.set_title(title)
    ax.set_xlabel("Team 1 Elo")
    ax.set_ylabel("Team 2 Elo")
    ax.grid(True)

plt.tight_layout()
plt.show()

# =====================================================
# 9. TRAIN MODEL
# =====================================================

print("\n" + "=" * 50)
print("TRAINING MODEL")
print("=" * 50)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Training complete.")

# =====================================================
# 10. VALIDATION EVALUATION
# =====================================================

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

val_preds = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_preds
)

print(f"Validation Accuracy: {val_accuracy}")

print("\nClassification Report:")
print(classification_report(
    y_val,
    val_preds
))

cm = confusion_matrix(
    y_val,
    val_preds
)

ConfusionMatrixDisplay(cm).plot()

plt.title("Validation Confusion Matrix")
plt.show()

# =====================================================
# 11. TEST EVALUATION
# =====================================================

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)

test_preds = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_preds
)

print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_preds
))

cm = confusion_matrix(
    y_test,
    test_preds
)

ConfusionMatrixDisplay(cm).plot()

plt.title("Test Confusion Matrix")
plt.show()

# =====================================================
# 12. DECISION SURFACE
# =====================================================

print("\nGenerating decision surface...")

x_min = X["team 1 elo"].min() - 50
x_max = X["team 1 elo"].max() + 50

y_min = X["team 2 elo"].min() - 50
y_max = X["team 2 elo"].max() + 50

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

probs = model.predict_proba(grid)[:, 1]
probs = probs.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(
    xx,
    yy,
    probs,
    levels=30,
    alpha=0.5
)

plt.colorbar(
    label="Probability Team 1 Wins"
)

plt.scatter(
    X_train[y_train == 1]["team 1 elo"],
    X_train[y_train == 1]["team 2 elo"],
    alpha=0.7,
    label="Win"
)

plt.scatter(
    X_train[y_train == 0]["team 1 elo"],
    X_train[y_train == 0]["team 2 elo"],
    alpha=0.7,
    label="Loss"
)

plt.xlabel("Team 1 Elo")
plt.ylabel("Team 2 Elo")
plt.title("Learned Decision Surface")
plt.legend()

plt.show()

# =====================================================
# 13. PREDICTION EXAMPLES
# =====================================================

print("\n" + "=" * 50)
print("EXAMPLE PREDICTIONS")
print("=" * 50)

examples = [
    (2200, 1800),
    (2000, 1900),
    (1900, 1900),
    (1800, 2000),
    (1700, 2100)
]

for elo1, elo2 in examples:

    prob = model.predict_proba(
        [[elo1, elo2]]
    )[0][1]

    prediction = model.predict(
        [[elo1, elo2]]
    )[0]

    print(
        f"Team1 Elo={elo1:4d}, "
        f"Team2 Elo={elo2:4d} | "
        f"P(Win)={prob:.3f} | "
        f"Prediction={prediction}"
    )

# =====================================================
# 14. FEATURE IMPORTANCE
# =====================================================

classifier = model.named_steps["classifier"]

print("\n" + "=" * 50)
print("MODEL COEFFICIENTS")
print("=" * 50)

print(
    f"Team 1 Elo coefficient: "
    f"{classifier.coef_[0][0]:.4f}"
)

print(
    f"Team 2 Elo coefficient: "
    f"{classifier.coef_[0][1]:.4f}"
)

print(
    f"Intercept: "
    f"{classifier.intercept_[0]:.4f}"
)
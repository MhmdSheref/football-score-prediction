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
# LOAD BOTH WORLD CUPS
# =====================================================

df1 = pd.read_csv("2022.csv")
df2 = pd.read_csv("2018.csv")

df = pd.concat([df1, df2], ignore_index=True)

print("Combined rows:", len(df))

# =====================================================
# CLEAN
# =====================================================

df = df.dropna(subset=["win"])

df["win"] = df["win"].astype(int)

print("Rows after cleaning:", len(df))

# =====================================================
# CREATE ELO DIFFERENCE
# =====================================================

df["elo_diff"] = (
    df["team 1 elo"] -
    df["team 2 elo"]
)

# =====================================================
# DATA AUGMENTATION
# =====================================================

flipped = df.copy()

flipped["elo_diff"] = -flipped["elo_diff"]

# swap result
flipped["win"] = 1 - flipped["win"]

augmented = pd.concat(
    [df, flipped],
    ignore_index=True
)

print("\nOriginal dataset:", len(df))
print("Augmented dataset:", len(augmented))

# =====================================================
# VISUALIZE AUGMENTATION
# =====================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["elo_diff"],
    df["win"],
    alpha=0.4,
    label="Original"
)

plt.scatter(
    flipped["elo_diff"],
    flipped["win"],
    alpha=0.2,
    label="Flipped"
)

plt.xlabel("Elo Difference")
plt.ylabel("Win")
plt.title("Original + Flipped Matches")

plt.legend()
plt.show()

# =====================================================
# FEATURES
# =====================================================

X = augmented[["elo_diff"]]
y = augmented["win"]

# =====================================================
# SPLIT
# =====================================================

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

print("\nTrain:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =====================================================
# TRAIN
# =====================================================

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nCoefficient:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# =====================================================
# LEARNED CURVE
# =====================================================

elo_range = np.linspace(
    X["elo_diff"].min(),
    X["elo_diff"].max(),
    1000
)

probs = model.predict_proba(
    elo_range.reshape(-1, 1)
)[:, 1]

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train["elo_diff"],
    y_train,
    alpha=0.15
)

plt.plot(
    elo_range,
    probs,
    linewidth=3
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Elo Difference")
plt.ylabel("P(Team 1 Wins)")
plt.title("Win Probability vs Elo Difference")

plt.show()

# =====================================================
# VALIDATION
# =====================================================

val_preds = model.predict(X_val)

print("\nValidation Accuracy")
print(accuracy_score(y_val, val_preds))

print("\nValidation Report")
print(classification_report(y_val, val_preds))

ConfusionMatrixDisplay(
    confusion_matrix(y_val, val_preds)
).plot()

plt.title("Validation Confusion Matrix")
plt.show()

# =====================================================
# TEST
# =====================================================

test_preds = model.predict(X_test)

print("\nTest Accuracy")
print(accuracy_score(y_test, test_preds))

print("\nTest Report")
print(classification_report(y_test, test_preds))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, test_preds)
).plot()

plt.title("Test Confusion Matrix")
plt.show()

# =====================================================
# EXAMPLE ODDS
# =====================================================

print("\nExample Predictions")

for diff in [-400, -200, 0, 200, 400]:

    prob = model.predict_proba([[diff]])[0][1]

    print(
        f"Elo diff {diff:+4d} -> "
        f"{prob:.3f}"
    )



elo_range = np.linspace(-600, 600, 1000)

elo_curve = 1 / (
    1 + 10 ** (-elo_range / 400)
)

lr_curve = model.predict_proba(
    elo_range.reshape(-1, 1)
)[:, 1]

plt.figure(figsize=(10,6))

plt.plot(
    elo_range,
    elo_curve,
    label="Standard Elo",
    linewidth=3
)

plt.plot(
    elo_range,
    lr_curve,
    label="Learned Logistic",
    linewidth=3
)

plt.axvline(0, linestyle="--")

plt.xlabel("Elo Difference")
plt.ylabel("Win Probability")
plt.title("Standard Elo vs Learned Logistic")

plt.legend()
plt.show()
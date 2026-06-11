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
# LOAD DATA
# =====================================================

df1 = pd.read_csv("2022.csv")
df2 = pd.read_csv("2018.csv")

df = pd.concat([df1, df2], ignore_index=True)

# =====================================================
# CLEAN
# =====================================================

df = df.dropna(
    subset=[
        "team 1 elo",
        "team 2 elo",
        "score advantage"
    ]
)

# =====================================================
# CREATE ELO DIFFERENCE
# =====================================================

df["elo_diff"] = (
        df["team 1 elo"] -
        df["team 2 elo"]
)

# =====================================================
# AUGMENT DATA
# =====================================================

flipped = df.copy()

flipped["elo_diff"] = -df["elo_diff"]

flipped["score advantage"] = (
    -df["score advantage"]
)

df_augmented = pd.concat(
    [df, flipped],
    ignore_index=True
)

print("Original rows:", len(df))
print("Augmented rows:", len(df_augmented))

# =====================================================
# VISUALIZE RAW DATA
# =====================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df_augmented["elo_diff"],
    df_augmented["score advantage"],
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

X = df_augmented[["elo_diff"]]

y = df_augmented["score advantage"]

# =====================================================
# SPLIT
# =====================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
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
# VISUALIZE FITTED LINE
# =====================================================

x_line = np.linspace(
    X["elo_diff"].min(),
    X["elo_diff"].max(),
    1000
)

y_line = model.predict(
    x_line.reshape(-1, 1)
)

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
    label="Regression Line"
)

plt.axhline(
    0,
    linestyle="--",
    alpha=0.5
)

plt.axvline(
    0,
    linestyle="--",
    alpha=0.5
)

plt.xlabel("Elo Difference")
plt.ylabel("Score Advantage")
plt.title("Score Advantage vs Elo Difference")

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# VALIDATION METRICS
# =====================================================

val_preds = model.predict(X_val)

print("\nVALIDATION")

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_val,
            val_preds
        ),
        3
    )
)

print(
    "RMSE:",
    round(
        np.sqrt(
            mean_squared_error(
                y_val,
                val_preds
            )
        ),
        3
    )
)

print(
    "R²:",
    round(
        r2_score(
            y_val,
            val_preds
        ),
        3
    )
)

# =====================================================
# TEST METRICS
# =====================================================

test_preds = model.predict(X_test)

print("\nTEST")

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_test,
            test_preds
        ),
        3
    )
)

print(
    "RMSE:",
    round(
        np.sqrt(
            mean_squared_error(
                y_test,
                test_preds
            )
        ),
        3
    )
)

print(
    "R²:",
    round(
        r2_score(
            y_test,
            test_preds
        ),
        3
    )
)

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    test_preds,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    test_preds.min()
)

maximum = max(
    y_test.max(),
    test_preds.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.xlabel("Actual Score Advantage")
plt.ylabel("Predicted Score Advantage")
plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()

# =====================================================
# MODEL EQUATION
# =====================================================

regressor = model.named_steps["regressor"]

scaler = model.named_steps["scaler"]

coef = (
        regressor.coef_[0]
        /
        scaler.scale_[0]
)

intercept = (
        regressor.intercept_
        -
        coef * scaler.mean_[0]
)

print("\nModel Equation")
print(
    f"score_advantage = "
    f"{coef:.6f} * elo_diff + "
    f"{intercept:.6f}"
)

# =====================================================
# EXAMPLE PREDICTIONS
# =====================================================

for diff in [-400, -200, 0, 200, 400]:
    pred = model.predict([[diff]])[0]

    print(
        f"Elo diff {diff:+4d} -> "
        f"Predicted score advantage = "
        f"{pred:.2f}"
    )

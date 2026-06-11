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
# 1. LOAD DATA
# =====================================================

print("=" * 50)
print("LOADING DATA")
print("=" * 50)

df1 = pd.read_csv("2022.csv")
df2 = pd.read_csv("2018.csv")

df = pd.concat([df1, df2], ignore_index=True)

print("Total rows:", len(df))

# =====================================================
# 2. CLEAN DATA
# =====================================================

df = df.dropna(
    subset=[
        "team 1 elo",
        "team 2 elo",
        "score advantage"
    ]
)

print("Rows after cleaning:", len(df))

# =====================================================
# 3. DATA AUGMENTATION
# =====================================================

# If Team A beat Team B by +3
# then Team B lost to Team A by -3

flipped = df.copy()

flipped["team 1 elo"] = df["team 2 elo"]
flipped["team 2 elo"] = df["team 1 elo"]

flipped["score advantage"] = -df["score advantage"]

df_augmented = pd.concat(
    [df, flipped],
    ignore_index=True
)

print("Augmented rows:", len(df_augmented))

# =====================================================
# 4. VISUALIZE TARGET
# =====================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df_augmented["score advantage"],
    bins=15
)

plt.xlabel("Score Advantage")
plt.ylabel("Count")
plt.title("Distribution of Score Advantage")

plt.show()

# =====================================================
# 5. FEATURES / TARGET
# =====================================================

X = df_augmented[
    ["team 1 elo", "team 2 elo"]
]

y = df_augmented["score advantage"]

# =====================================================
# 6. SPLIT DATA
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

print("\nTrain:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =====================================================
# 7. TRAIN REGRESSION MODEL
# =====================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

print("\nTraining complete.")

# =====================================================
# 8. VALIDATION
# =====================================================

val_preds = model.predict(X_val)

mae = mean_absolute_error(
    y_val,
    val_preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_preds
    )
)

r2 = r2_score(
    y_val,
    val_preds
)

print("\nValidation Metrics")
print("------------------")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²  :", round(r2, 3))

# =====================================================
# 9. PREDICTED VS ACTUAL
# =====================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_val,
    val_preds,
    alpha=0.7
)

min_val = min(y_val.min(), val_preds.min())
max_val = max(y_val.max(), val_preds.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Score Advantage")
plt.ylabel("Predicted Score Advantage")

plt.title("Validation Predictions")

plt.show()

# =====================================================
# 10. TEST SET
# =====================================================

test_preds = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    test_preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_preds
    )
)

r2 = r2_score(
    y_test,
    test_preds
)

print("\nTest Metrics")
print("------------")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²  :", round(r2, 3))

# =====================================================
# 11. RESIDUAL PLOT
# =====================================================

residuals = y_test - test_preds

plt.figure(figsize=(8, 6))

plt.scatter(
    test_preds,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.show()

# =====================================================
# 12. COEFFICIENTS
# =====================================================

regressor = model.named_steps["regressor"]

print("\nCoefficients")
print("------------")
print(
    "Team 1 Elo:",
    regressor.coef_[0]
)

print(
    "Team 2 Elo:",
    regressor.coef_[1]
)

print(
    "Intercept:",
    regressor.intercept_
)

# =====================================================
# 13. EXAMPLE PREDICTIONS
# =====================================================

examples = [
    (2200, 1800),
    (2000, 1900),
    (1900, 1900),
    (1800, 2000),
    (1700, 2100)
]

print("\nExample Predictions")
print("-------------------")

for elo1, elo2 in examples:

    pred = model.predict(
        [[elo1, elo2]]
    )[0]

    print(
        f"{elo1} vs {elo2}"
        f" -> predicted score advantage = "
        f"{pred:.2f}"
    )

from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# VISUALIZE REGRESSION PLANE
# =====================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Actual data
ax.scatter(
    X_train["team 1 elo"],
    X_train["team 2 elo"],
    y_train,
    alpha=0.6
)

# Create plane grid
elo1_range = np.linspace(
    X["team 1 elo"].min(),
    X["team 1 elo"].max(),
    30
)

elo2_range = np.linspace(
    X["team 2 elo"].min(),
    X["team 2 elo"].max(),
    30
)

elo1_grid, elo2_grid = np.meshgrid(
    elo1_range,
    elo2_range
)

grid_points = np.c_[
    elo1_grid.ravel(),
    elo2_grid.ravel()
]

predicted_surface = model.predict(
    grid_points
).reshape(elo1_grid.shape)

# Plot regression plane
ax.plot_surface(
    elo1_grid,
    elo2_grid,
    predicted_surface,
    alpha=0.5
)

ax.set_xlabel("Team 1 Elo")
ax.set_ylabel("Team 2 Elo")
ax.set_zlabel("Score Advantage")

ax.set_title("Linear Regression Plane")

plt.show()
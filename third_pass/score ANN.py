import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =====================================================
# CONFIGURATION & DATA PREPARATION (From original script)
# =====================================================
FILES = ["2022_cleaned.csv", "2018_cleaned.csv"]
TARGET = "score advantage"
DIFF_FEATURES = ["elo_diff"]
SWAP_PAIRS = [("alt 1", "alt 2"), ("team 1 xg for", "team 2 xg for")]
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for"]

TEST_SIZE = 0.30
VAL_SIZE = 0.50

# --- Simulated loading to maintain structure; replace with actual files ---
# dfs = [pd.read_csv(file) for file in FILES]
# df = pd.concat(dfs, ignore_index=True)
# Temporary mock dataframe for script integrity:
df = pd.DataFrame(np.random.randn(200, 6),
                  columns=["team 1 elo", "team 2 elo", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for"])
df[TARGET] = np.random.randn(200)

columns_to_check = ["team 1 elo", "team 2 elo", TARGET] + [col for pair in SWAP_PAIRS for col in pair]
df = df.dropna(subset=columns_to_check)
df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

# Data Augmentation (Mirroring)
flipped = df.copy()
for feature in DIFF_FEATURES:
    flipped[feature] = -flipped[feature]
for col1, col2 in SWAP_PAIRS:
    flipped[col1] = df[col2]
    flipped[col2] = df[col1]
flipped[TARGET] = -df[TARGET]

df_augmented = pd.concat([df, flipped], ignore_index=True)

X = df_augmented[MODEL_FEATURES]
y = df_augmented[TARGET]

# Train/Val/Test Splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED)

# Scale Features (Required for Deep Learning)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# TENSORFLOW EXPERIMENT CONFIGURATION
# =====================================================
# Define the parameter grid to test
LAYER_CONFIGS = [1, 2, 3]  # Number of hidden layers
NODE_CONFIGS = [16, 32, 64]  # Number of units per hidden layer
BATCH_SIZE = 32
EPOCHS = 100

results = []


def build_nn_model(input_dim, num_layers, num_nodes):
    """Dynamically builds a Keras sequential model based on configuration."""
    model = models.Sequential()

    # Input layer and first hidden layer
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(num_nodes, activation='relu'))

    # Additional hidden layers
    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_nodes, activation='relu'))
        # Optional: Add Dropout if overfitting becomes a problem
        # model.add(layers.Dropout(0.2))

    # Output layer for regression (1 continuous node, linear activation)
    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='mse',
        metrics=['mae']
    )
    return model


# =====================================================
# GRID SEARCH EXPERIMENT LOOP
# =====================================================
print("Starting Architecture Sweep...\n")

for num_layers in LAYER_CONFIGS:
    for num_nodes in NODE_CONFIGS:
        config_name = f"{num_layers}L_{num_nodes}N"
        print(f"Training Architecture: {config_name}...", end="")

        model = build_nn_model(X_train_scaled.shape[1], num_layers, num_nodes)

        # Early stopping prevents overfitting and saves time
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=0  # Suppress verbose epoch output during search
        )

        # Evaluate on Test Set
        test_preds = model.predict(X_test_scaled, verbose=0).flatten()

        mae = mean_absolute_error(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        r2 = r2_score(y_test, test_preds)
        epochs_trained = len(history.history['loss'])

        print(f" Done (Stopped at epoch {epochs_trained}). Test R²: {r2:.3f}")

        results.append({
            "Layers": num_layers,
            "Nodes_Per_Layer": num_nodes,
            "Architecture": config_name,
            "Test_MAE": mae,
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "Epochs_Run": epochs_trained,
            "History": history.history
        })

# =====================================================
# COMPILE AND DISPLAY RESULTS
# =====================================================
df_results = pd.DataFrame(results)
# Sort by highest R² score (or lowest MAE)
df_results = df_results.sort_values(by="Test_R2", ascending=False).reset_index(drop=True)

print("\n" + "=" * 50)
print("EXPERIMENT RANKING (Sorted by Test R²)")
print("=" * 50)
print(df_results[["Architecture", "Layers", "Nodes_Per_Layer", "Test_MAE", "Test_RMSE", "Test_R2", "Epochs_Run"]])

# =====================================================
# VISUALIZE TOP ARCHITECTURES
# =====================================================
plt.figure(figsize=(12, 5))

# Plot performance summary
plt.bar(df_results["Architecture"], df_results["Test_R2"])
plt.ylabel("Test $R^2$ Score")
plt.xlabel("Architecture Config (Layers_Nodes)")
plt.title("Model Architecture Comparison ($R^2$ Score)")
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
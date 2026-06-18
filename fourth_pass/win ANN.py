import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

FILES = ["2022_cleaned.csv", "2018_cleaned.csv"]
TARGET = "win"

DIFF_FEATURES = ["elo_diff"]
SWAP_PAIRS = [("alt 1", "alt 2"), ("team 1 xg for", "team 2 xg for")]
MODEL_FEATURES = ["elo_diff", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for"]

TEST_SIZE = 0.30
VAL_SIZE = 0.50
NOISE_FACTOR = 0.005  # 0.5% variance for data augmentation

# =====================================================
# LOAD & CLEAN (Using Mock Data for demonstration)
# =====================================================

# In production, use your file loading logic:
# dfs = [pd.read_csv(file) for file in FILES]
# df = pd.concat(dfs, ignore_index=True)

# Mock data generation to ensure the script runs standalone
df = pd.DataFrame(np.random.randn(500, 6),
                  columns=["team 1 elo", "team 2 elo", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for"])
df[TARGET] = np.random.randint(0, 2, 500)  # Binary target (0 or 1)

df = df.dropna(subset=[TARGET] + ["team 1 elo", "team 2 elo"])
df[TARGET] = df[TARGET].astype(int)
df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]

# =====================================================
# 1. SPLIT DATA (CRITICAL: Do this BEFORE augmentation)
# =====================================================

# Split into Train and Temp
train_df, temp_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df[TARGET]
)

# Split Temp into Validation and Test
val_df, test_df = train_test_split(
    temp_df, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=temp_df[TARGET]
)

print(f"Base Training rows: {len(train_df)}")
print(f"Validation rows:    {len(val_df)}")
print(f"Test rows:          {len(test_df)}")


# =====================================================
# 2. DATA AUGMENTATION (Train Set ONLY)
# =====================================================

def augment_training_data(data, noise_factor=0.005):
    """Mirrors the data, then duplicates everything with slight noise."""

    # Step A: Mirror the data
    flipped = data.copy()
    for feature in DIFF_FEATURES:
        flipped[feature] = -flipped[feature]
    for col1, col2 in SWAP_PAIRS:
        flipped[col1] = data[col2]
        flipped[col2] = data[col1]
    # Flip the binary target
    flipped[TARGET] = 1 - flipped[TARGET]

    # Combine original + mirrored
    mirrored_combined = pd.concat([data, flipped], ignore_index=True)

    # Step B: Add Noise
    noisy_data = mirrored_combined.copy()

    for feature in MODEL_FEATURES:
        noise_multipliers = np.random.uniform(
            1 - noise_factor,
            1 + noise_factor,
            size=len(noisy_data)
        )
        noisy_data[feature] = noisy_data[feature] * noise_multipliers

    # Combine (Original + Mirrored) + Noisy(Original + Mirrored)
    final_augmented = pd.concat([mirrored_combined, noisy_data], ignore_index=True)
    return final_augmented


train_df_aug = augment_training_data(train_df, noise_factor=NOISE_FACTOR)
print(f"\nAugmented Training rows: {len(train_df_aug)}\n")

X_train = train_df_aug[MODEL_FEATURES]
y_train = train_df_aug[TARGET]

X_val = val_df[MODEL_FEATURES]
y_val = val_df[TARGET]

X_test = test_df[MODEL_FEATURES]
y_test = test_df[TARGET]

# =====================================================
# SCALE FEATURES
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# TENSORFLOW EXPERIMENT CONFIGURATION
# =====================================================

# Drastically reduced architectures to prevent overfitting
LAYER_CONFIGS = [1, 2]  # Just 1 or 2 hidden layers
NODE_CONFIGS = [2, 4, 8]  # Very small node counts
BATCH_SIZE = 32
EPOCHS = 150

results = []


def build_classifier_model(input_dim, num_layers, num_nodes):
    """Dynamically builds a small binary classification neural network."""
    model = models.Sequential()

    # Input & First Hidden Layer (Using SIGMOID instead of RELU)
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(num_nodes, activation='sigmoid'))

    # Additional hidden layers (Using SIGMOID instead of RELU)
    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_nodes, activation='sigmoid'))

    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),  # Slightly higher LR to help sigmoid converge
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =====================================================
# GRID SEARCH EXPERIMENT LOOP
# =====================================================

print("Starting Small Architecture Sweep...\n")

for num_layers in LAYER_CONFIGS:
    for num_nodes in NODE_CONFIGS:
        config_name = f"{num_layers}L_{num_nodes}N"
        print(f"Training Architecture: {config_name}...", end=" ")

        model = build_classifier_model(X_train_scaled.shape[1], num_layers, num_nodes)

        # Patience increased slightly because sigmoids learn slower than ReLUs
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=0
        )

        test_preds_proba = model.predict(X_test_scaled, verbose=0).flatten()
        test_preds_class = (test_preds_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds_class)
        loss = log_loss(y_test, test_preds_proba)
        epochs_trained = len(history.history['loss'])

        print(f"Done (Epochs: {epochs_trained}). Test Acc: {acc:.4f} | Log Loss: {loss:.4f}")

        results.append({
            "Layers": num_layers,
            "Nodes_Per_Layer": num_nodes,
            "Architecture": config_name,
            "Test_Accuracy": acc,
            "Test_Log_Loss": loss,
            "Epochs_Run": epochs_trained
        })

# =====================================================
# COMPILE AND DISPLAY RESULTS
# =====================================================

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=["Test_Accuracy", "Test_Log_Loss"], ascending=[False, True]).reset_index(
    drop=True)

print("\n" + "=" * 55)
print("EXPERIMENT RANKING (Sorted by Test Accuracy)")
print("=" * 55)
print(df_results[["Architecture", "Test_Accuracy", "Test_Log_Loss", "Epochs_Run"]])

# =====================================================
# VISUALIZE TOP ARCHITECTURES
# =====================================================

plt.figure(figsize=(10, 5))

plt.bar(df_results["Architecture"], df_results["Test_Accuracy"], color='lightgreen', edgecolor='black')
plt.ylabel("Test Accuracy")
plt.xlabel("Architecture Config (Layers_Nodes)")
plt.title("Small Model Architecture Comparison (Sigmoid Activations)")

plt.ylim(0.4, 1.0)
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Guessing (0.5)')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
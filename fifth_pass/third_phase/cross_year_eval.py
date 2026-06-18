import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Ensure TF logs are minimal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_year(year):
    features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

    matches = pd.read_csv(f"{year}_matches.csv")
    teams = pd.read_csv(f"data/{year}_teams.csv")
    
    m1 = pd.merge(matches, teams, left_on='team 1', right_on='Team', how='left')
    m1 = m1.rename(columns={f: f"{f}_1" for f in features_to_diff})
    
    m2 = pd.merge(m1, teams, left_on='team 2', right_on='Team', how='left')
    m2 = m2.rename(columns={f: f"{f}_2" for f in features_to_diff})
    
    for f, d_col in zip(features_to_diff, diff_cols):
        m2[d_col] = pd.to_numeric(m2[f"{f}_1"], errors='coerce') - pd.to_numeric(m2[f"{f}_2"], errors='coerce')
        
    df = m2.dropna(subset=['win', 'score advantage'] + diff_cols).copy()
    df['win'] = df['win'].astype(int)
    return df, diff_cols

print("Loading and preparing data...")
df_2018, diff_cols = load_and_prepare_year(2018)
df_2022, _ = load_and_prepare_year(2022)

def get_augmented(df):
    flipped = df.copy()
    for d_col in diff_cols:
        flipped[d_col] = -flipped[d_col]
    flipped['win'] = 1 - flipped['win']
    flipped['score advantage'] = -flipped['score advantage']
    return pd.concat([df, flipped], ignore_index=True)

# Train on 2018 (augmented), Test on 2022 (raw matches)
train_df = get_augmented(df_2018)
test_df = df_2022

# ---------------------------------------------------------
# 1. Classification (Win Prediction)
# Architecture 1 from phase 2: Width=32, Swish, Dropout=0.2, L2=0.0001, Adam lr=0.001
# Best 7 features from phase 2
# ---------------------------------------------------------
clf_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
X_train_c = train_df[clf_feats].values
y_train_c = train_df['win'].values
X_test_c = test_df[clf_feats].values
y_test_c = test_df['win'].values

scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)

print(f"\n--- WIN PREDICTION (Classification) ---")
print(f"Training on 2018 ({len(X_train_c)} samples), Testing on 2022 ({len(X_test_c)} samples)")
print(f"Features used: {clf_feats}")

# Average over a few seeds for stability
accs = []
for rs in range(5):
    np.random.seed(42 + rs)
    tf.random.set_seed(42 + rs)
    
    clf_model = Sequential([
        Dense(32, input_dim=len(clf_feats), activation='swish', kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    clf_model.fit(X_train_c, y_train_c, epochs=50, batch_size=32, verbose=0)
    
    _, clf_acc = clf_model.evaluate(X_test_c, y_test_c, verbose=0)
    accs.append(clf_acc)

print(f"-> Accuracy across 5 runs: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")

# ---------------------------------------------------------
# 2. Regression (Score Prediction)
# Architecture 2 from phase 2: Width=16, Swish, L2=0.001, RMSprop lr=0.003
# Best 3 features from phase 2
# ---------------------------------------------------------
reg_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
X_train_r = train_df[reg_feats].values
y_train_r = train_df['score advantage'].values
X_test_r = test_df[reg_feats].values
y_test_r = test_df['score advantage'].values

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

print(f"\n--- SCORE PREDICTION (Regression) ---")
print(f"Training on 2018 ({len(X_train_r)} samples), Testing on 2022 ({len(X_test_r)} samples)")
print(f"Features used: {reg_feats}")

maes, rmses, r2s = [], [], []
for rs in range(5):
    np.random.seed(42 + rs)
    tf.random.set_seed(42 + rs)
    
    reg_model = Sequential([
        Dense(16, input_dim=len(reg_feats), activation='swish', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    reg_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.003), loss='mse', metrics=['mae'])
    reg_model.fit(X_train_r, y_train_r, epochs=50, batch_size=32, verbose=0)
    
    preds_r = reg_model.predict(X_test_r, verbose=0)
    maes.append(mean_absolute_error(y_test_r, preds_r))
    rmses.append(np.sqrt(mean_squared_error(y_test_r, preds_r)))
    r2s.append(r2_score(y_test_r, preds_r))

print(f"-> MAE:  {np.mean(maes):.4f} (+/- {np.std(maes):.4f})")
print(f"-> RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
print(f"-> R^2:  {np.mean(r2s):.4f} (+/- {np.std(r2s):.4f})")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_data():
    base_features = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age']
    old_value_features = ['Total Value', 'Avg per Player']
    new_value_features = ['StdDev', 'Min Player', 'Max Player', 'Star Dependency']
    
    all_features = base_features + old_value_features + new_value_features
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in all_features]

    all_dfs = []
    for year in [2018, 2022]:
        matches = pd.read_csv(f"../fifth_pass/second_pass/{year}_matches.csv")
        teams = pd.read_csv(f"data/{year}_teams.csv")
        
        m1 = pd.merge(matches, teams, left_on='team 1', right_on='Team', how='left')
        m1 = m1.rename(columns={f: f"{f}_1" for f in all_features})
        
        m2 = pd.merge(m1, teams, left_on='team 2', right_on='Team', how='left')
        m2 = m2.rename(columns={f: f"{f}_2" for f in all_features})
        
        for f, d_col in zip(all_features, diff_cols):
            m2[d_col] = pd.to_numeric(m2[f"{f}_1"], errors='coerce') - pd.to_numeric(m2[f"{f}_2"], errors='coerce')
            
        df = m2.dropna(subset=['win', 'score advantage'] + diff_cols).copy()
        df['win'] = df['win'].astype(int)
        all_dfs.append(df)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Augment
    flipped = df.copy()
    for d_col in diff_cols:
        flipped[d_col] = -flipped[d_col]
    flipped['win'] = 1 - flipped['win']
    flipped['score advantage'] = -flipped['score advantage']
    
    return pd.concat([df, flipped], ignore_index=True), diff_cols

def build_win_model(input_dim):
    model = Sequential([
        Dense(32, input_dim=input_dim, activation='swish', kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

def build_score_model(input_dim):
    model = Sequential([
        Dense(16, input_dim=input_dim, activation='swish', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.003), loss='mse')
    return model

def evaluate_feature_set(df, features, name):
    print(f"\n--- Evaluating {name} ---")
    print(f"Features ({len(features)}): {features}")
    
    X = df[features].values
    y_win = df['win'].values
    y_score = df['score advantage'].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    win_accs = []
    score_maes = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        yw_train, yw_test = y_win[train_idx], y_win[test_idx]
        ys_train, ys_test = y_score[train_idx], y_score[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Win Model
        win_model = build_win_model(len(features))
        win_model.fit(X_train_s, yw_train, epochs=50, batch_size=32, verbose=0)
        win_preds = (win_model.predict(X_test_s, verbose=0).flatten() > 0.5).astype(int)
        win_accs.append(accuracy_score(yw_test, win_preds))
        
        # Score Model
        score_model = build_score_model(len(features))
        score_model.fit(X_train_s, ys_train, epochs=50, batch_size=32, verbose=0)
        score_preds = score_model.predict(X_test_s, verbose=0).flatten()
        score_maes.append(mean_absolute_error(ys_test, score_preds))
        
    print(f"Win Model Accuracy: {np.mean(win_accs):.4f} (+/- {np.std(win_accs):.4f})")
    print(f"Score Model MAE:    {np.mean(score_maes):.4f} (+/- {np.std(score_maes):.4f})")
    return np.mean(win_accs), np.mean(score_maes)

def main():
    df, diff_cols = load_and_prepare_data()
    print(f"Dataset Shape: {df.shape}")
    
    base = ['elo_diff', 'xg_for_diff', 'alt_diff', 'humid_diff', 'avg_age_diff']
    old_val = ['total_value_diff', 'avg_per_player_diff']
    new_val = ['stddev_diff', 'min_player_diff', 'max_player_diff', 'star_dependency_diff']
    
    # 1. Baseline
    feat_baseline = base + old_val
    evaluate_feature_set(df, feat_baseline, "Baseline (Fifth Pass Features)")
    
    # 2. Additive (All)
    feat_additive = base + old_val + new_val
    evaluate_feature_set(df, feat_additive, "Model A (Baseline + New Features)")
    
    # 3. Replacement
    feat_replace = base + new_val
    evaluate_feature_set(df, feat_replace, "Model B (Base + New Features, NO Old Value Metrics)")

if __name__ == '__main__':
    main()

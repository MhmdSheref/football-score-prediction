import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from itertools import combinations
import gc
import os

# Ensure TF logs are minimal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_data():
    years = [2018, 2022]
    all_matches = []
    features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

    for year in years:
        matches = pd.read_csv(f"{year}_matches.csv")
        teams = pd.read_csv(f"data/{year}_teams.csv")
        
        m1 = pd.merge(matches, teams, left_on='team 1', right_on='Team', how='left')
        m1 = m1.rename(columns={f: f"{f}_1" for f in features_to_diff})
        
        m2 = pd.merge(m1, teams, left_on='team 2', right_on='Team', how='left')
        m2 = m2.rename(columns={f: f"{f}_2" for f in features_to_diff})
        
        for f, d_col in zip(features_to_diff, diff_cols):
            m2[d_col] = pd.to_numeric(m2[f"{f}_1"], errors='coerce') - pd.to_numeric(m2[f"{f}_2"], errors='coerce')
            
        all_matches.append(m2)

    df = pd.concat(all_matches, ignore_index=True)
    df = df.dropna(subset=['win'] + diff_cols)
    df['win'] = df['win'].astype(int)

    flipped = df.copy()
    for d_col in diff_cols:
        flipped[d_col] = -flipped[d_col]
    flipped['win'] = 1 - flipped['win']

    augmented = pd.concat([df, flipped], ignore_index=True)
    return augmented, diff_cols

augmented, diff_cols = load_and_prepare_data()

# Prepare feature sets
top_5_df = pd.read_csv('win_combinatorial_results.csv').head(5)
top_5_combos = [c.split(', ') for c in top_5_df['features'].tolist()]
seven_feat_combos = [list(c) for c in combinations(diff_cols, 7)]
eight_feat_combos = [diff_cols]

all_combos_to_test = top_5_combos + seven_feat_combos + eight_feat_combos

# Dedup
unique_combos = []
for c in all_combos_to_test:
    sc = tuple(sorted(c))
    if sc not in unique_combos:
        unique_combos.append(sc)

# Architectures
archs = [
    {'name': 'Arch1', 'width': 32, 'activation': 'swish', 'dropout': 0.2, 'l2': 0.0001, 'optimizer': 'adam', 'lr': 0.001},
    {'name': 'Arch2', 'width': 32, 'activation': 'swish', 'dropout': 0.0, 'l2': 0.0, 'optimizer': 'adam', 'lr': 0.001},
    {'name': 'Arch3', 'width': 32, 'activation': 'swish', 'dropout': 0.2, 'l2': 0.0, 'optimizer': 'adam', 'lr': 0.001}
]

SEEDS = [42 + i for i in range(5)] # using 5 seeds for NNs

results = []
total_runs = len(unique_combos) * len(archs)
print(f"Total configurations to test: {total_runs}")

run_idx = 1
for combo in unique_combos:
    combo_list = list(combo)
    X = augmented[combo_list]
    y = augmented['win']
    
    for arch in archs:
        val_accs = []
        for rs in SEEDS:
            # Set random seeds for reproducibility
            np.random.seed(rs)
            tf.random.set_seed(rs)
            
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=rs)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=rs)
            
            model = Sequential([
                Dense(arch['width'], input_dim=len(combo_list), activation=arch['activation'], 
                      kernel_regularizer=l2(arch['l2'])),
                Dropout(arch['dropout']),
                Dense(1, activation='sigmoid')
            ])
            
            opt = tf.keras.optimizers.Adam(learning_rate=arch['lr'])
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, 
                      validation_data=(X_val, y_val))
            
            _, val_acc = model.evaluate(X_val, y_val, verbose=0)
            val_accs.append(val_acc)
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
            
        avg_val_acc = np.mean(val_accs)
        results.append({
            'arch': arch['name'],
            'num_features': len(combo_list),
            'features': ", ".join(combo_list),
            'val_accuracy': avg_val_acc
        })
        print(f"Progress: {run_idx}/{total_runs} -> Arch: {arch['name']}, Feats: {len(combo_list)}, Val Acc: {avg_val_acc:.4f}")
        run_idx += 1

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by='val_accuracy', ascending=False)
res_df.to_csv("win_nn_evaluation_results.csv", index=False)
print("Saved win_nn_evaluation_results.csv")

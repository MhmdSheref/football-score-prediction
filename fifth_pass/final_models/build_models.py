import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_all_data():
    features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

    all_dfs = []
    for year in [2018, 2022]:
        matches = pd.read_csv(f"../second_pass/{year}_matches.csv")
        teams = pd.read_csv(f"{year}_teams.csv")
        
        m1 = pd.merge(matches, teams, left_on='team 1', right_on='Team', how='left')
        m1 = m1.rename(columns={f: f"{f}_1" for f in features_to_diff})
        
        m2 = pd.merge(m1, teams, left_on='team 2', right_on='Team', how='left')
        m2 = m2.rename(columns={f: f"{f}_2" for f in features_to_diff})
        
        for f, d_col in zip(features_to_diff, diff_cols):
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

train_df, diff_cols = load_all_data()

# -----------------
# 1. Win Model
# -----------------
clf_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
Xc = train_df[clf_feats].values
yc = train_df['win'].values

scaler_c = StandardScaler()
Xc_s = scaler_c.fit_transform(Xc)

with open('win_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_c, f)

tf.random.set_seed(42)
win_model = Sequential([
    Dense(32, input_shape=(7,), activation='swish', kernel_regularizer=l2(0.0001), name="Hidden_Layer_32"),
    Dropout(0.2, name="Dropout_0.2"),
    Dense(1, activation='sigmoid', name="Output_Layer")
], name="Win_Prediction_Model")
win_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
win_model.fit(Xc_s, yc, epochs=50, batch_size=32, verbose=0)

win_model.save('win_model.keras')

try:
    plot_model(win_model, to_file='win_model_arch.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print("Could not plot win model arch:", e)

# -----------------
# 2. Score Model
# -----------------
reg_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
Xr = train_df[reg_feats].values
yr = train_df['score advantage'].values

scaler_r = StandardScaler()
Xr_s = scaler_r.fit_transform(Xr)

with open('score_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_r, f)

tf.random.set_seed(42)
score_model = Sequential([
    Dense(16, input_shape=(3,), activation='swish', kernel_regularizer=l2(0.001), name="Hidden_Layer_16"),
    Dense(1, name="Output_Layer")
], name="Score_Prediction_Model")
score_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.003), loss='mse', metrics=['mae'])
score_model.fit(Xr_s, yr, epochs=50, batch_size=32, verbose=0)

score_model.save('score_model.keras')

try:
    plot_model(score_model, to_file='score_model_arch.png', show_shapes=True, show_layer_names=True)
except Exception as e:
    print("Could not plot score model arch:", e)

print("Models and scalers built successfully.")

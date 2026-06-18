import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import os
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.style.use('default')

# Paths
brain_dir = "report"

# --- 1. Evolution of Performance ---
def plot_evolution():
    # Win Accuracies: Linear (~82.2), NN CV (87.4), NN Test (86.8)
    # Score R2: Linear (0.355), NN CV (0.520), NN Test (0.398)
    labels = ['Baseline (Linear)', 'Optimized NN (CV)', 'Cross-Year Test']
    win_acc = [82.2, 87.4, 86.8]
    score_r2 = [0.355*100, 0.520*100, 0.398*100] # scale to 100 for easy viewing next to acc

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, win_acc, width, label='Win Accuracy (%)', color='#00d2ff')
    rects2 = ax.bar(x + width/2, score_r2, width, label='Score R² (x100)', color='#ff007f')

    ax.set_ylabel('Score')
    ax.set_title('Evolution of Model Performance Across Phases')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add values
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(os.path.join(brain_dir, 'evolution_of_performance.png'), dpi=300)
    plt.close()

# --- 2. Dimensionality Impact ---
def plot_dimensionality():
    win_df = pd.read_csv('second_pass/win_nn_evaluation_results.csv')
    score_df = pd.read_csv('second_pass/score_nn_evaluation_results.csv')
    
    # Average across architectures for a given dimensionality
    win_dim = win_df.groupby('num_features')['val_accuracy'].max() * 100
    score_dim = score_df.groupby('num_features')['val_rmse'].min()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#00d2ff'
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Win Accuracy (%)', color=color)
    ax1.plot(win_dim.index, win_dim.values, marker='o', color=color, linewidth=2, label='Win Acc')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color2 = '#ff007f'
    ax2.set_ylabel('Score RMSE (Lower is Better)', color=color2)
    ax2.plot(score_dim.index, score_dim.values, marker='s', color=color2, linewidth=2, label='Score RMSE')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Impact of Dimensionality on Neural Network Performance')
    fig.tight_layout()
    plt.savefig(os.path.join(brain_dir, 'dimensionality_impact.png'), dpi=300)
    plt.close()

# --- 3. Feature Importance Heatmap ---
def plot_feature_importance():
    win_df = pd.read_csv('second_pass/win_combinatorial_results.csv').head(20)
    score_df = pd.read_csv('second_pass/score_combinatorial_results.csv').head(20)
    
    features = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features]
    
    win_counts = {f: 0 for f in diff_cols}
    score_counts = {f: 0 for f in diff_cols}
    
    for c in win_df['features']:
        for f in c.split(', '):
            win_counts[f] += 1
            
    for c in score_df['features']:
        for f in c.split(', '):
            score_counts[f] += 1
            
    heatmap_data = pd.DataFrame({'Win Prediction': win_counts, 'Score Prediction': score_counts})
    # normalize by dividing by 20
    heatmap_data = heatmap_data / 20.0
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='magma', fmt='.0%')
    plt.title('Feature Inclusion Frequency in Top 20 Linear Models')
    plt.tight_layout()
    plt.savefig(os.path.join(brain_dir, 'feature_importance_heatmap.png'), dpi=300)
    plt.close()

# --- 4. Cross Year Validation ---
def plot_cross_year():
    # Simple visual representation of Train vs Test density
    df_18 = pd.read_csv('second_pass/2018_matches.csv')
    df_22 = pd.read_csv('second_pass/2022_matches.csv')
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_18['score advantage'].dropna(), fill=True, label='2018 Data (Train)', color='#00d2ff')
    sns.kdeplot(df_22['score advantage'].dropna(), fill=True, label='2022 Data (Test)', color='#ff007f')
    plt.title('Score Advantage Distribution: Train (2018) vs Test (2022)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(brain_dir, 'cross_year_validation.png'), dpi=300)
    plt.close()

# --- 5. Sensitivity Curves (Partial Dependence) ---
def plot_sensitivity():
    # We will load the data, train the final models on 2018, and sweep one parameter while holding others at mean
    features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

    matches = pd.read_csv("second_pass/2018_matches.csv")
    teams = pd.read_csv("second_pass/data/2018_teams.csv")
    m1 = pd.merge(matches, teams, left_on='team 1', right_on='Team', how='left')
    m1 = m1.rename(columns={f: f"{f}_1" for f in features_to_diff})
    m2 = pd.merge(m1, teams, left_on='team 2', right_on='Team', how='left')
    m2 = m2.rename(columns={f: f"{f}_2" for f in features_to_diff})
    for f, d_col in zip(features_to_diff, diff_cols):
        m2[d_col] = pd.to_numeric(m2[f"{f}_1"], errors='coerce') - pd.to_numeric(m2[f"{f}_2"], errors='coerce')
    df = m2.dropna(subset=['win', 'score advantage'] + diff_cols).copy()
    
    flipped = df.copy()
    for d_col in diff_cols:
        flipped[d_col] = -flipped[d_col]
    flipped['win'] = 1 - flipped['win']
    flipped['score advantage'] = -flipped['score advantage']
    train_df = pd.concat([df, flipped], ignore_index=True)

    # Reg Model: 3 features
    reg_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
    Xr = train_df[reg_feats].values
    yr = train_df['score advantage'].values
    sr = StandardScaler()
    Xr_s = sr.fit_transform(Xr)

    np.random.seed(42)
    tf.random.set_seed(42)
    reg_model = Sequential([
        Dense(16, input_dim=3, activation='swish', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    reg_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.003), loss='mse')
    reg_model.fit(Xr_s, yr, epochs=50, batch_size=32, verbose=0)

    # Win Model: 7 features
    clf_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
    Xc = train_df[clf_feats].values
    yc = train_df['win'].values
    sc = StandardScaler()
    Xc_s = sc.fit_transform(Xc)

    np.random.seed(42)
    tf.random.set_seed(42)
    clf_model = Sequential([
        Dense(32, input_dim=7, activation='swish', kernel_regularizer=l2(0.0001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
    clf_model.fit(Xc_s, yc, epochs=50, batch_size=32, verbose=0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for ax in axes:
        ax.set_visible(False)
        
    fig.suptitle("Final Model Sensitivity Curves (Holding other features at mean)", fontsize=16)

    # Plot for Score Prediction (3 features)
    for i, feat in enumerate(reg_feats):
        ax = axes[i]
        ax.set_visible(True)
        mean_vals = np.mean(Xr, axis=0)
        feat_idx = reg_feats.index(feat)
        
        # sweep from 5th to 95th percentile
        p5 = np.percentile(Xr[:, feat_idx], 5)
        p95 = np.percentile(Xr[:, feat_idx], 95)
        sweep = np.linspace(p5, p95, 100)
        
        X_sweep = np.tile(mean_vals, (100, 1))
        X_sweep[:, feat_idx] = sweep
        X_sweep_s = sr.transform(X_sweep)
        preds = reg_model.predict(X_sweep_s, verbose=0)
        
        ax.plot(sweep, preds, color='#ff007f', linewidth=2)
        ax.set_title(f"Score Pred vs {feat}")
        ax.set_ylabel("Predicted Score Advantage")
        ax.set_xlabel(feat)

    # Plot for Win Prediction (First 4 features of 7 to fit nicely)
    for i, feat in enumerate(clf_feats[:4]):
        ax = axes[4+i]
        ax.set_visible(True)
        mean_vals = np.mean(Xc, axis=0)
        feat_idx = clf_feats.index(feat)
        
        p5 = np.percentile(Xc[:, feat_idx], 5)
        p95 = np.percentile(Xc[:, feat_idx], 95)
        sweep = np.linspace(p5, p95, 100)
        
        X_sweep = np.tile(mean_vals, (100, 1))
        X_sweep[:, feat_idx] = sweep
        X_sweep_s = sc.transform(X_sweep)
        preds = clf_model.predict(X_sweep_s, verbose=0)
        
        ax.plot(sweep, preds, color='#00d2ff', linewidth=2)
        ax.set_title(f"Win Prob vs {feat}")
        ax.set_ylabel("Predicted Win Probability")
        ax.set_xlabel(feat)

    fig.tight_layout()
    plt.savefig(os.path.join(brain_dir, 'sensitivity_curves.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_evolution()
    plot_dimensionality()
    plot_feature_importance()
    plot_cross_year()
    plot_sensitivity()
    print("All graphics generated successfully.")

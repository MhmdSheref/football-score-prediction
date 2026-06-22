import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from generate_requested_graphs import load_data

def main():
    out_dir = '../report'
    os.makedirs(out_dir, exist_ok=True)
    
    train_df = load_data()
    
    win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
    Xc = train_df[win_feats].values
    
    win_model = tf.keras.models.load_model('win_model.keras')
    with open('win_scaler.pkl', 'rb') as f:
        win_scaler = pickle.load(f)

    feats_to_plot = ['avg_attend_diff', 'alt_diff']
    titles = ['Strong Feature: Attendance Diff', 'Weak Feature: Altitude Diff']
    colors = ['#E15759', '#4E79A7']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Win Probability Sensitivity: Strong vs Weak Feature\n(Holding all other features at their mean)", fontsize=18, fontweight='bold', y=1.05)
    
    for i, feat in enumerate(feats_to_plot):
        ax = axes[i]
        mean_vals = np.mean(Xc, axis=0)
        feat_idx = win_feats.index(feat)
        
        p5 = np.percentile(Xc[:, feat_idx], 5)
        p95 = np.percentile(Xc[:, feat_idx], 95)
        sweep = np.linspace(p5, p95, 100)
        
        X_sweep = np.tile(mean_vals, (100, 1))
        X_sweep[:, feat_idx] = sweep
        
        X_sweep_s = win_scaler.transform(X_sweep)
        preds = win_model.predict(X_sweep_s, verbose=0).flatten()
        
        ax.plot(sweep, preds, color=colors[i], lw=4)
        ax.set_title(titles[i], fontsize=16, pad=15)
        ax.set_xlabel(f"{feat} (5th to 95th Percentile)", fontsize=14)
        if i == 0:
            ax.set_ylabel('Predicted Win Probability', fontsize=14)
            
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.4, linestyle='--')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sensitivity_strong_vs_weak.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
from sklearn.metrics import confusion_matrix
from generate_requested_graphs import load_data

def main():
    out_dir = '../report'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the dataset (2018 and 2022 matches augmented)
    df = load_data()
    
    # Load Models and Scalers
    win_model = tf.keras.models.load_model('win_model.keras')
    score_model = tf.keras.models.load_model('score_model.keras')
    
    with open('win_scaler.pkl', 'rb') as f:
        win_scaler = pickle.load(f)
    with open('score_scaler.pkl', 'rb') as f:
        score_scaler = pickle.load(f)
        
    win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
    score_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
    
    X_win = win_scaler.transform(df[win_feats].values)
    X_score = score_scaler.transform(df[score_feats].values)
    
    y_true = df['win'].values
    
    # Get Predictions
    win_preds_prob = win_model.predict(X_win, verbose=0).flatten()
    win_preds = (win_preds_prob > 0.5).astype(int)
    
    score_preds_raw = score_model.predict(X_score, verbose=0).flatten()
    # If predicted score advantage > 0, it's a win
    score_preds = (score_preds_raw > 0).astype(int)
    
    # Calculate Confusion Matrices
    cm_win = confusion_matrix(y_true, win_preds)
    cm_score = confusion_matrix(y_true, score_preds)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Confusion Matrices (Test/Evaluation Set)', fontsize=18, fontweight='bold', y=1.05)
    
    # Function to plot a single CM
    def plot_cm(ax, cm, title):
        group_names = ['True Negative\n(Correct Loss/Draw)', 'False Positive\n(Predicted Win, Actual Loss/Draw)', 
                       'False Negative\n(Predicted Loss/Draw, Actual Win)', 'True Positive\n(Correct Win)']
        group_counts = [f"{value}" for value in cm.flatten()]
        group_percentages = [f"{value:.1%}" for value in cm.flatten()/np.sum(cm)]
        
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax, cbar=False, 
                    annot_kws={'size': 12}, square=True, linewidths=1, linecolor='black')
        
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.xaxis.set_ticklabels(['Loss / Draw', 'Win'], fontsize=12)
        ax.yaxis.set_ticklabels(['Loss / Draw', 'Win'], fontsize=12)
        
    plot_cm(axes[0], cm_win, 'Win Probability Model')
    plot_cm(axes[1], cm_score, 'Score Difference Model\n(Advantage > 0 = Win)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrices generated successfully.")

if __name__ == '__main__':
    main()

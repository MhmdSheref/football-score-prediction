import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import os

plt.style.use('default')

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 32
})


out_dir = '../report/2x'
os.makedirs(out_dir, exist_ok=True)

features = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features]

def get_counts(df_path):
    df = pd.read_csv(df_path).head(20)
    counts = {f: 0 for f in diff_cols}
    for c in df['features']:
        if pd.isna(c):
            continue
        for f in str(c).split(', '):
            f = f.strip()
            if f in counts:
                counts[f] += 1
    return [counts[f]/20.0 for f in diff_cols]

def main():
    # Paths
    p_lin_win = '../second_pass/win_combinatorial_results.csv'
    p_lin_score = '../second_pass/score_combinatorial_results.csv'
    p_nn_win = '../second_pass/win_nn_evaluation_results.csv'
    p_nn_score = '../second_pass/score_nn_evaluation_results.csv'
    
    # Get normalized counts
    lin_win_counts = get_counts(p_lin_win)
    lin_score_counts = get_counts(p_lin_score)
    nn_win_counts = get_counts(p_nn_win)
    nn_score_counts = get_counts(p_nn_score)
    
    # 1. Bar graph for 2 models (Linear Win vs Linear Score)
    x = np.arange(len(diff_cols))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, [c*100 for c in lin_win_counts], width, label='Linear Win Model', color='#4E79A7')
    rects2 = ax.bar(x + width/2, [c*100 for c in lin_score_counts], width, label='Linear Score Model', color='#F28E2B')
    
    ax.set_ylabel('Inclusion Frequency (%)')
    ax.set_title('Feature Inclusion Frequency in Top 20 Linear Models')
    ax.set_xticks(x)
    ax.set_xticklabels(diff_cols, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importance_bars.png'), dpi=300)
    plt.close()
    
    # 2. Bar graph for 4 models
    width4 = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - 1.5*width4, [c*100 for c in lin_win_counts], width4, label='Linear Win', color='#4E79A7')
    ax.bar(x - 0.5*width4, [c*100 for c in lin_score_counts], width4, label='Linear Score', color='#F28E2B')
    ax.bar(x + 0.5*width4, [c*100 for c in nn_win_counts], width4, label='NN Win', color='#59A14F')
    ax.bar(x + 1.5*width4, [c*100 for c in nn_score_counts], width4, label='NN Score', color='#E15759')
    
    ax.set_ylabel('Inclusion Frequency (%)')
    ax.set_ylim(0, 120)
    ax.set_title('Feature Inclusion Frequency in Top 20 Models')
    ax.set_xticks(x)
    ax.set_xticklabels(diff_cols, rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importance_4models.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    main()

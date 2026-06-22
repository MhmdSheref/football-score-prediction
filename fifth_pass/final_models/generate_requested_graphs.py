import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.style.use('default')

# Output directory
out_dir = '../report'
os.makedirs(out_dir, exist_ok=True)

def load_data():
    features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

    # Load 2018 and 2022
    m18 = pd.read_csv("../second_pass/2018_matches.csv")
    t18 = pd.read_csv("../second_pass/data/2018_teams.csv")
    m22 = pd.read_csv("../second_pass/2022_matches.csv")
    t22 = pd.read_csv("../second_pass/data/2022_teams.csv")
    
    def process(matches, teams):
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
        return pd.concat([df, flipped], ignore_index=True)
        
    df18 = process(m18, t18)
    df22 = process(m22, t22)
    return pd.concat([df18, df22], ignore_index=True)

def generate_shap(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats):
    # Prepare data
    X_win = df[win_feats].values
    X_score = df[score_feats].values
    
    X_win_s = win_scaler.transform(X_win)
    X_score_s = score_scaler.transform(X_score)
    
    # We will use DeepExplainer or GradientExplainer
    # If DeepExplainer fails, we fall back to KernelExplainer with a sample
    bg_win = X_win_s[np.random.choice(X_win_s.shape[0], 100, replace=False)]
    bg_score = X_score_s[np.random.choice(X_score_s.shape[0], 100, replace=False)]
    
    try:
        explainer_win = shap.DeepExplainer(win_model, bg_win)
        shap_values_win = explainer_win.shap_values(X_win_s)
    except:
        explainer_win = shap.KernelExplainer(win_model.predict, bg_win)
        shap_values_win = explainer_win.shap_values(X_win_s[:200])
        X_win_s = X_win_s[:200]
        X_win = X_win[:200]
        
    try:
        explainer_score = shap.DeepExplainer(score_model, bg_score)
        shap_values_score = explainer_score.shap_values(X_score_s)
    except:
        explainer_score = shap.KernelExplainer(score_model.predict, bg_score)
        shap_values_score = explainer_score.shap_values(X_score_s[:200])
        X_score_s = X_score_s[:200]
        X_score = X_score[:200]

    all_feats = ['elo_diff', 'xg_for_diff', 'alt_diff', 'humid_diff', 'avg_age_diff', 'total_value_diff', 'avg_per_player_diff', 'avg_attend_diff']
    
    # Pad Win Model SHAP
    if isinstance(shap_values_win, list):
        sv_w_raw = shap_values_win[0]
    elif len(shap_values_win.shape) == 3 and shap_values_win.shape[2] == 1:
        sv_w_raw = shap_values_win[:, :, 0]
    else:
        sv_w_raw = shap_values_win
        
    sv_w = np.zeros((sv_w_raw.shape[0], len(all_feats)))
    X_w_padded = np.zeros((X_win.shape[0], len(all_feats)))
    for i, f in enumerate(all_feats):
        if f in win_feats:
            idx = win_feats.index(f)
            sv_w[:, i] = sv_w_raw[:, idx]
            X_w_padded[:, i] = X_win[:, idx]

    # Pad Score Model SHAP
    if isinstance(shap_values_score, list):
        sv_s_raw = shap_values_score[0]
    elif len(shap_values_score.shape) == 3 and shap_values_score.shape[2] == 1:
        sv_s_raw = shap_values_score[:, :, 0]
    else:
        sv_s_raw = shap_values_score
        
    sv_s = np.zeros((sv_s_raw.shape[0], len(all_feats)))
    X_s_padded = np.zeros((X_score.shape[0], len(all_feats)))
    for i, f in enumerate(all_feats):
        if f in score_feats:
            idx = score_feats.index(f)
            sv_s[:, i] = sv_s_raw[:, idx]
            X_s_padded[:, i] = X_score[:, idx]
        
    # Generate Merged SHAP plot
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 20,
        'axes.titlesize': 24
    })
    
    fig = plt.figure(figsize=(28, 20))
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    base_cmap = cm.get_cmap('YlOrBr')
    # Start at 0.3 to remove the extremely light yellow that is hard to see
    brown_cmap = mcolors.LinearSegmentedColormap.from_list('darker_YlOrBr', base_cmap(np.linspace(0.3, 1, 256)))
    
    plt.subplot(2, 2, 1)
    shap.summary_plot(sv_w, X_w_padded, feature_names=all_feats, plot_type='bar', show=False, color='#8C564B', plot_size=None)
    plt.title('Win Model Prediction (Classification) - Mean SHAP')
    
    plt.subplot(2, 2, 2)
    shap.summary_plot(sv_w, X_w_padded, feature_names=all_feats, show=False, cmap=brown_cmap, plot_size=None)
    plt.title('Win Model - Feature Impact')
    
    plt.subplot(2, 2, 3)
    shap.summary_plot(sv_s, X_s_padded, feature_names=all_feats, plot_type='bar', show=False, color='#8C564B', plot_size=None)
    plt.title('Score Model Prediction (Regression) - Mean SHAP')
    
    plt.subplot(2, 2, 4)
    shap.summary_plot(sv_s, X_s_padded, feature_names=all_feats, show=False, cmap=brown_cmap, plot_size=None)
    plt.title('Score Model - Feature Impact')
    
    plt.subplots_adjust(wspace=0.8, hspace=0.6, left=0.15, right=0.95, top=0.95, bottom=0.1)
    fig.savefig(os.path.join(out_dir, '1_shap_values_merged.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

def generate_calibration(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats):
    X_win = win_scaler.transform(df[win_feats].values)
    y_win = df['win'].values
    y_pred_win = win_model.predict(X_win, verbose=0).flatten()
    
    X_score = score_scaler.transform(df[score_feats].values)
    y_score = df['score advantage'].values
    y_pred_score = score_model.predict(X_score, verbose=0).flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Win Model Calibration Curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_win, y_pred_win, n_bins=10)
    
    ax1_twin = ax1.twinx()
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax1.plot(prob_pred, prob_true, marker='o', color='blue', label='Win Model')
    ax1.set_ylabel('Actual Win Probability')
    ax1.set_xlabel('Predicted Win Probability')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('Calibration Curve & Probability Distribution - Win Model')
    
    ax1_twin.hist(y_pred_win, bins=20, alpha=0.3, color='blue')
    ax1_twin.set_ylabel('Frequency')
    ax1.legend(loc='upper left')
    
    # 2. Score Difference Model "Calibration" (Actual vs Predicted)
    # Bin the predicted scores and calculate mean actual score
    bins = np.linspace(y_pred_score.min(), y_pred_score.max(), 10)
    bin_indices = np.digitize(y_pred_score, bins)
    
    bin_means_pred = [y_pred_score[bin_indices == i].mean() for i in range(1, len(bins))]
    bin_means_actual = [y_score[bin_indices == i].mean() for i in range(1, len(bins))]
    
    ax2_twin = ax2.twinx()
    # plot ideal line
    min_val = min(y_pred_score.min(), y_score.min())
    max_val = max(y_pred_score.max(), y_score.max())
    ax2.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray')
    ax2.plot(bin_means_pred, bin_means_actual, marker='o', color='red', label='Goal Diff Model')
    
    ax2.set_ylabel('Actual Goal Difference (Mean)')
    ax2.set_xlabel('Predicted Goal Difference')
    ax2.set_title('Binned Actual vs Predicted & Distribution - Goal Diff Model')
    
    ax2_twin.hist(y_pred_score, bins=20, alpha=0.3, color='red')
    ax2_twin.set_ylabel('Frequency')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2_calibration_distribution.png'), dpi=300)
    plt.close()

def generate_win_vs_score_correctness(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats):
    X_win = win_scaler.transform(df[win_feats].values)
    y_win = df['win'].values
    y_pred_win = win_model.predict(X_win, verbose=0).flatten()
    
    X_score = score_scaler.transform(df[score_feats].values)
    y_pred_score = score_model.predict(X_score, verbose=0).flatten()
    
    # Correct guess from model 1 (Win Model)
    # pred prob > 0.5 and actual win == 1 OR pred prob < 0.5 and actual win == 0
    correct_guess = ((y_pred_win >= 0.5) & (y_win == 1)) | ((y_pred_win < 0.5) & (y_win == 0))
    
    plt.figure(figsize=(10, 8))
    
    # Correct: Blue circles
    plt.scatter(y_pred_score[correct_guess], y_pred_win[correct_guess], 
                c='blue', marker='o', alpha=0.6, label='Correct Win Prediction')
                
    # Wrong: Red crosses
    plt.scatter(y_pred_score[~correct_guess], y_pred_win[~correct_guess], 
                c='red', marker='x', alpha=0.8, label='Wrong Win Prediction')
                
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.xlabel('Predicted Score Difference')
    plt.ylabel('Win Model Probability')
    plt.title('Win Model Probability vs Predicted Score Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_win_prob_vs_score_diff.png'), dpi=300)
    plt.close()

def generate_score_error_graph(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats):
    X_win = win_scaler.transform(df[win_feats].values)
    y_pred_win = win_model.predict(X_win, verbose=0).flatten()
    
    X_score = score_scaler.transform(df[score_feats].values)
    y_score = df['score advantage'].values
    y_pred_score = score_model.predict(X_score, verbose=0).flatten()
    
    # Calculate error
    error = y_score - y_pred_score
    abs_error = np.abs(error)
    
    plt.figure(figsize=(10, 8))
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap_rbp = LinearSegmentedColormap.from_list('rby', ['darkred', 'red', 'blue', 'yellow', 'darkgoldenrod'])
    max_err = 4.0 # Fixed to ensure -4 to +4 mapping
    
    cond_mid = (error >= -1.15) & (error <= 1.15)
    cond_high = (error > 1.15)
    cond_low = (error < -1.15)
    
    # Plot in three pieces to get the distinct markers, but use the exact same colormap scale
    sc_mid = plt.scatter(y_pred_score[cond_mid], y_pred_win[cond_mid], 
                c=error[cond_mid], cmap=cmap_rbp, marker='o', alpha=0.8, 
                vmin=-max_err, vmax=max_err, label='Near Zero Error (|err| <= 1.15)')
                
    sc_high = plt.scatter(y_pred_score[cond_high], y_pred_win[cond_high], 
                c=error[cond_high], cmap=cmap_rbp, marker='s', alpha=0.8, 
                vmin=-max_err, vmax=max_err, label='Large Positive Error (err > 1.15)')
                
    sc_low = plt.scatter(y_pred_score[cond_low], y_pred_win[cond_low], 
                c=error[cond_low], cmap=cmap_rbp, marker='x', alpha=0.9, 
                vmin=-max_err, vmax=max_err, label='Large Negative Error (err < -1.15)')
                
    # Add colorbar for the gradient (from any of the scatter objects since they share the same scale)
    plt.colorbar(sc_mid, label='Prediction Error (Real - Predicted)')
                
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.xlabel('Predicted Score Difference')
    plt.ylabel('Win Model Probability')
    plt.title('Score Model Error Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_score_error_graph.png'), dpi=300)
    plt.close()

def main():
    df = load_data()
    
    win_model = tf.keras.models.load_model('win_model.keras')
    score_model = tf.keras.models.load_model('score_model.keras')
    with open('win_scaler.pkl', 'rb') as f:
        win_scaler = pickle.load(f)
    with open('score_scaler.pkl', 'rb') as f:
        score_scaler = pickle.load(f)
        
    win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
    score_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
    
    print("Generating SHAP values...")
    generate_shap(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats)
    
    print("Generating Calibration curve...")
    generate_calibration(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats)
    
    print("Generating Win vs Score diff correctness...")
    generate_win_vs_score_correctness(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats)
    
    print("Generating Score error graph...")
    generate_score_error_graph(df, win_model, score_model, win_scaler, score_scaler, win_feats, score_feats)
    
    print("Done!")

if __name__ == '__main__':
    main()

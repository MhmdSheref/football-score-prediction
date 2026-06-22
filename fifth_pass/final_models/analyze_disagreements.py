import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

def load_data():
    features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
    diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

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

def analyze_disagreements():
    # Load data
    df = load_data()
    
    # Define features
    win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
    score_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
    
    # Load models
    win_model = tf.keras.models.load_model('win_model.keras')
    score_model = tf.keras.models.load_model('score_model.keras')
    
    # Load scalers
    with open('win_scaler.pkl', 'rb') as f:
        win_scaler = pickle.load(f)
    with open('score_scaler.pkl', 'rb') as f:
        score_scaler = pickle.load(f)
        
    # Prepare inputs
    X_win = win_scaler.transform(df[win_feats].values)
    X_score = score_scaler.transform(df[score_feats].values)
    
    # Predict
    win_prob = win_model.predict(X_win, verbose=0).flatten()
    score_pred = score_model.predict(X_score, verbose=0).flatten()
    
    # Actual outcomes
    actual_score_diff = df['score advantage'].values
    actual_win = (actual_score_diff > 0).astype(int)
    
    # Model binary predictions
    pred_win_model = (win_prob > 0.5).astype(int)
    pred_score_model = (score_pred > 0).astype(int)
    
    # Find disagreements
    disagreements = (pred_win_model != pred_score_model)
    
    total_cases = len(df)
    total_disagreements = np.sum(disagreements)
    
    # Accuracy when disagreeing
    win_correct_on_disagree = np.sum((pred_win_model == actual_win) & disagreements)
    score_correct_on_disagree = np.sum((pred_score_model == actual_win) & disagreements)
    
    print(f"Total Matches: {total_cases}")
    print(f"Total Disagreements: {total_disagreements} ({total_disagreements/total_cases*100:.1f}%)")
    print("-" * 40)
    print("When models disagree:")
    print(f"Win Model Correct: {win_correct_on_disagree} times ({(win_correct_on_disagree/total_disagreements)*100:.1f}%)")
    print(f"Score Model Correct: {score_correct_on_disagree} times ({(score_correct_on_disagree/total_disagreements)*100:.1f}%)")
    print("-" * 40)
    
    # Print some actual examples
    print("\nExamples of Disagreements:")
    disagree_idx = np.where(disagreements)[0]
    
    # Display up to 5 examples
    for i in disagree_idx[:5]:
        row = df.iloc[i]
        match = f"{row.get('team1', 'Team A')} vs {row.get('team2', 'Team B')}"
        print(f"\nMatch: {match}")
        print(f"Actual Goal Diff: {actual_score_diff[i]} (Win: {bool(actual_win[i])})")
        print(f"Win Model Predicts: {'Win' if pred_win_model[i] else 'Not Win'} (Prob: {win_prob[i]:.3f})")
        print(f"Score Model Predicts: {'Win' if pred_score_model[i] else 'Not Win'} (Pred Diff: {score_pred[i]:.3f})")
        
if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    analyze_disagreements()

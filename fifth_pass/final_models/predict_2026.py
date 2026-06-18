import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # 1. Load 2026 data and map columns to match our expected format
    df_26 = pd.read_csv('2026.csv')
    
    col_mapping = {
        'xG diff (MD1)': 'xG_for',
        'Home Altitude (m)': 'Alt',
        'Home Humidity (%)': 'Humid',
        'Total Squad Value (€M)': 'Total Value',
        'Average Player Value (€M)': 'Avg per Player',
        'Est. Attendance': 'Avg Attend.'
    }
    df_26 = df_26.rename(columns=col_mapping)
    
    # Create dictionary for quick lookups
    teams_dict = {}
    for _, row in df_26.iterrows():
        # Handle naming discrepancies
        t_name = str(row['Team']).strip().lower()
        if t_name == "bosnia":
            teams_dict["bosnia and herzegovina"] = row.to_dict()
        elif t_name == "dr congo":
            teams_dict["congo dr"] = row.to_dict()
        teams_dict[t_name] = row.to_dict()
        
    # 2. Define Matches
    matches = [
        ("Mexico", "South Africa"),
        ("Korea Republic", "Czechia"),
        ("Canada", "Bosnia and Herzegovina"),
        ("USA", "Paraguay"),
        ("Haiti", "Scotland"),
        ("Australia", "Turkiye"),
        ("Brazil", "Morocco"),
        ("Qatar", "Switzerland"),
        ("Cote d'Ivoire", "Ecuador"),
        ("Germany", "Curacao"),
        ("Netherlands", "Japan"),
        ("Sweden", "Tunisia"),
        ("Saudi Arabia", "Uruguay"),
        ("Spain", "Cabo Verde"),
        ("IR Iran", "New Zealand"),
        ("Belgium", "Egypt"),
        ("France", "Senegal"),
        ("Iraq", "Norway"),
        ("Argentina", "Algeria"),
        ("Austria", "Jordan"),
        ("Ghana", "Panama"),
        ("England", "Croatia"),
        ("Portugal", "Congo DR"),
        ("Uzbekistan", "Colombia"),
        ("Czechia", "South Africa"),
        ("Switzerland", "Bosnia and Herzegovina"),
        ("Canada", "Qatar"),
        ("Mexico", "Korea Republic"),
        ("USA", "Australia"),
        ("Scotland", "Morocco"),
        ("Brazil", "Haiti"),
        ("Turkiye", "Paraguay"),
        ("Germany", "Cote d'Ivoire"),
        ("Ecuador", "Curacao"),
        ("Netherlands", "Sweden"),
        ("Tunisia", "Japan"),
        ("Uruguay", "Cabo Verde"),
        ("Spain", "Saudi Arabia"),
        ("Belgium", "IR Iran"),
        ("New Zealand", "Egypt"),
        ("Norway", "Senegal"),
        ("France", "Iraq"),
        ("Argentina", "Austria"),
        ("Jordan", "Algeria"),
        ("England", "Ghana"),
        ("Panama", "Croatia"),
        ("Portugal", "Uzbekistan"),
        ("Colombia", "Congo DR"),
        ("Scotland", "Brazil"),
        ("Morocco", "Haiti"),
        ("Switzerland", "Canada"),
        ("Bosnia and Herzegovina", "Qatar"),
        ("Czechia", "Mexico"),
        ("South Africa", "Korea Republic"),
        ("Curacao", "Cote d'Ivoire"),
        ("Ecuador", "Germany"),
        ("Japan", "Sweden"),
        ("Tunisia", "Netherlands"),
        ("Turkiye", "USA"),
        ("Paraguay", "Australia"),
        ("Norway", "France"),
        ("Senegal", "Iraq"),
        ("Egypt", "IR Iran"),
        ("New Zealand", "Belgium"),
        ("Cabo Verde", "Saudi Arabia"),
        ("Uruguay", "Spain"),
        ("Panama", "England"),
        ("Croatia", "Ghana"),
        ("Algeria", "Austria"),
        ("Jordan", "Argentina"),
        ("Colombia", "Portugal"),
        ("Congo DR", "Uzbekistan")
    ]
    
    # 3. Load Models & Scalers
    win_model = tf.keras.models.load_model('win_model.keras')
    score_model = tf.keras.models.load_model('score_model.keras')
    with open('win_scaler.pkl', 'rb') as f:
        win_scaler = pickle.load(f)
    with open('score_scaler.pkl', 'rb') as f:
        score_scaler = pickle.load(f)
        
    win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
    score_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
    
    results = []
    
    for t1, t2 in matches:
        t1_key = t1.lower()
        t2_key = t2.lower()
        
        if t1_key not in teams_dict:
            print(f"Warning: {t1} not found in 2026.csv")
            continue
        if t2_key not in teams_dict:
            print(f"Warning: {t2} not found in 2026.csv")
            continue
            
        d1 = teams_dict[t1_key]
        d2 = teams_dict[t2_key]
        
        diffs = {
            'elo_diff': d1['Elo'] - d2['Elo'],
            'xg_for_diff': d1['xG_for'] - d2['xG_for'],
            'alt_diff': d1['Alt'] - d2['Alt'],
            'humid_diff': d1['Humid'] - d2['Humid'],
            'total_value_diff': d1['Total Value'] - d2['Total Value'],
            'avg_per_player_diff': d1['Avg per Player'] - d2['Avg per Player'],
            'avg_attend_diff': d1['Avg Attend.'] - d2['Avg Attend.']
        }
        
        win_input = np.array([[diffs[f] for f in win_feats]])
        score_input = np.array([[diffs[f] for f in score_feats]])
        
        win_input_scaled = win_scaler.transform(win_input)
        score_input_scaled = score_scaler.transform(score_input)
        
        win_prob = win_model.predict(win_input_scaled, verbose=0)[0][0]
        score_adv = score_model.predict(score_input_scaled, verbose=0)[0][0]
        
        if win_prob >= 0.5:
            predicted_winner = t1
            predicted_loser = t2
            win_p = win_prob
        else:
            predicted_winner = t2
            predicted_loser = t1
            win_p = 1.0 - win_prob
            
        # Score advantage is T1 - T2
        # if score_adv > 0, T1 is expected to win by that amount
        predicted_score_diff = abs(score_adv)
        
        results.append({
            'Team 1': t1,
            'Team 2': t2,
            'Team 1 Win Prob (%)': round(win_prob * 100, 2),
            'Team 2 Win Prob (%)': round((1 - win_prob) * 100, 2),
            'Predicted Winner': predicted_winner,
            'Expected Goal Margin': round(predicted_score_diff, 2),
            'Expected Score Advantage for T1': round(score_adv, 2)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('2026_predictions.csv', index=False)
    print("Successfully predicted 2026 matches and saved to 2026_predictions.csv")

if __name__ == "__main__":
    main()

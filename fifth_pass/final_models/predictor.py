import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_historical_data():
    try:
        df_22 = pd.read_csv('2022_teams.csv')
    except:
        df_22 = pd.DataFrame()
        
    try:
        df_18 = pd.read_csv('2018_teams.csv')
    except:
        df_18 = pd.DataFrame()
        
    try:
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
    except:
        df_26 = pd.DataFrame()
        
    teams_dict = {}
    
    if not df_18.empty:
        for _, row in df_18.iterrows():
            teams_dict[row['Team'].lower()] = row.to_dict()
            
    if not df_22.empty:
        for _, row in df_22.iterrows():
            teams_dict[row['Team'].lower()] = row.to_dict()
            
    if not df_26.empty:
        for _, row in df_26.iterrows():
            t_name = str(row['Team']).strip().lower()
            if t_name == "bosnia":
                teams_dict["bosnia and herzegovina"] = row.to_dict()
            elif t_name == "dr congo":
                teams_dict["congo dr"] = row.to_dict()
            teams_dict[t_name] = row.to_dict()
            
    return teams_dict

def get_param(team_name, param_name, user_val, teams_dict):
    if user_val is not None and str(user_val).strip() != "":
        return float(user_val)
    
    team_key = team_name.lower()
    if team_key in teams_dict and param_name in teams_dict[team_key]:
        return float(teams_dict[team_key][param_name])
        
    raise ValueError(f"Parameter '{param_name}' for team '{team_name}' not provided and not found in historical data.")

def run_interactive_mode(args, teams_dict):
    print("\n" + "="*50)
    print(" ⚽ INTERACTIVE FOOTBALL MATCH PREDICTOR ⚽ ")
    print("="*50)
    
    args.team1 = input("Enter Team 1 name: ").strip()
    args.team2 = input("Enter Team 2 name: ").strip()
    
    print("\nWhich prediction would you like to run?")
    print("1: Win Probability Only")
    print("2: Expected Score Margin Only")
    print("3: Both")
    choice = input("Choice [3]: ").strip()
    if choice == "":
        choice = "3"
        
    args.run_win = choice in ['1', '3']
    args.run_score = choice in ['2', '3']
    
    win_params = ['elo', 'xg', 'alt', 'humid', 'value', 'player_val', 'attend']
    score_params = ['xg', 'player_val', 'attend']
    
    active_params = set()
    if args.run_win:
        active_params.update(win_params)
    if args.run_score:
        active_params.update(score_params)
        
    all_configs = [
        ('elo', 'Elo', 'Elo Rating'),
        ('xg', 'xG_for', 'Expected Goals (xG)'),
        ('alt', 'Alt', 'Altitude'),
        ('humid', 'Humid', 'Humidity'),
        ('value', 'Total Value', 'Total Squad Value'),
        ('player_val', 'Avg per Player', 'Average Value per Player'),
        ('attend', 'Avg Attend.', 'Average Attendance')
    ]
    
    params_config = [c for c in all_configs if c[0] in active_params]
    
    print("\nFor each parameter below, press ENTER to accept the historical default [shown in brackets], or type a new number to override it.")
    
    for team_num, team_name in [(1, args.team1), (2, args.team2)]:
        print(f"\n--- Parameters for {team_name.upper()} ---")
        team_key = team_name.lower()
        
        for arg_prefix, dict_key, desc in params_config:
            default_val = ""
            if team_key in teams_dict and dict_key in teams_dict[team_key]:
                default_val = teams_dict[team_key][dict_key]
                
            while True:
                prompt_str = f"{desc} [{default_val}]: " if default_val != "" else f"{desc} (NO DEFAULT, input required): "
                user_input = input(prompt_str).strip()
                
                if user_input == "":
                    if default_val != "":
                        setattr(args, f"{arg_prefix}{team_num}", None)
                        break
                    else:
                        print("This value is required since we have no historical data. Please enter a number.")
                else:
                    try:
                        float(user_input)
                        setattr(args, f"{arg_prefix}{team_num}", user_input)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid number.")

    return args

def main():
    parser = argparse.ArgumentParser(description="Football Match Predictor using Final NNs")
    parser.add_argument("--team1", help="Name of Team 1")
    parser.add_argument("--team2", help="Name of Team 2")
    parser.add_argument("--only-win", action="store_true", help="Only run win prediction")
    parser.add_argument("--only-score", action="store_true", help="Only run score prediction")
    
    for p in ['elo', 'xg', 'alt', 'humid', 'value', 'player_val', 'attend']:
        parser.add_argument(f"--{p}1", type=float, help=f"{p} for Team 1")
        parser.add_argument(f"--{p}2", type=float, help=f"{p} for Team 2")
        
    args = parser.parse_args()
    teams_dict = load_historical_data()
    
    if len(sys.argv) == 1:
        args = run_interactive_mode(args, teams_dict)
    elif not args.team1 or not args.team2:
        print("Error: If passing arguments, --team1 and --team2 are required.")
        return
    else:
        # Defaults for CLI mode
        if args.only_win:
            args.run_win = True
            args.run_score = False
        elif args.only_score:
            args.run_win = False
            args.run_score = True
        else:
            args.run_win = True
            args.run_score = True
            
    diffs = {}
    try:
        if args.run_win:
            elo_1 = get_param(args.team1, 'Elo', getattr(args, 'elo1', None), teams_dict)
            elo_2 = get_param(args.team2, 'Elo', getattr(args, 'elo2', None), teams_dict)
            alt_1 = get_param(args.team1, 'Alt', getattr(args, 'alt1', None), teams_dict)
            alt_2 = get_param(args.team2, 'Alt', getattr(args, 'alt2', None), teams_dict)
            humid_1 = get_param(args.team1, 'Humid', getattr(args, 'humid1', None), teams_dict)
            humid_2 = get_param(args.team2, 'Humid', getattr(args, 'humid2', None), teams_dict)
            val_1 = get_param(args.team1, 'Total Value', getattr(args, 'value1', None), teams_dict)
            val_2 = get_param(args.team2, 'Total Value', getattr(args, 'value2', None), teams_dict)
            
            diffs['elo_diff'] = elo_1 - elo_2
            diffs['alt_diff'] = alt_1 - alt_2
            diffs['humid_diff'] = humid_1 - humid_2
            diffs['total_value_diff'] = val_1 - val_2
            
        if args.run_win or args.run_score:
            xg_1 = get_param(args.team1, 'xG_for', getattr(args, 'xg1', None), teams_dict)
            xg_2 = get_param(args.team2, 'xG_for', getattr(args, 'xg2', None), teams_dict)
            pval_1 = get_param(args.team1, 'Avg per Player', getattr(args, 'player_val1', None), teams_dict)
            pval_2 = get_param(args.team2, 'Avg per Player', getattr(args, 'player_val2', None), teams_dict)
            attend_1 = get_param(args.team1, 'Avg Attend.', getattr(args, 'attend1', None), teams_dict)
            attend_2 = get_param(args.team2, 'Avg Attend.', getattr(args, 'attend2', None), teams_dict)
            
            diffs['xg_for_diff'] = xg_1 - xg_2
            diffs['avg_per_player_diff'] = pval_1 - pval_2
            diffs['avg_attend_diff'] = attend_1 - attend_2
            
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    print("\n" + "="*50)
    print(f" MATCH PREDICTION: {args.team1.upper()} vs {args.team2.upper()}")
    print("="*50)
    
    if args.run_win:
        win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
        win_input = np.array([[diffs[f] for f in win_feats]])
        try:
            win_model = tf.keras.models.load_model('win_model.keras')
            with open('win_scaler.pkl', 'rb') as f:
                win_scaler = pickle.load(f)
            win_input_scaled = win_scaler.transform(win_input)
            win_prob = win_model.predict(win_input_scaled, verbose=0)[0][0]
            print("\n--- Win Prediction (Classification) ---")
            print(f"Probability {args.team1} wins: {win_prob*100:.2f}%")
            print(f"Probability {args.team2} wins/draws: {(1-win_prob)*100:.2f}%")
        except Exception as e:
            print("Error loading win model/scaler.")
            
    if args.run_score:
        score_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
        score_input = np.array([[diffs[f] for f in score_feats]])
        try:
            score_model = tf.keras.models.load_model('score_model.keras')
            with open('score_scaler.pkl', 'rb') as f:
                score_scaler = pickle.load(f)
            score_input_scaled = score_scaler.transform(score_input)
            score_adv = score_model.predict(score_input_scaled, verbose=0)[0][0]
            print("\n--- Score Prediction (Regression) ---")
            if score_adv > 0:
                print(f"Expected Goal Margin: {args.team1} wins by {score_adv:.2f} goals.")
            else:
                print(f"Expected Goal Margin: {args.team2} wins by {abs(score_adv):.2f} goals.")
        except Exception as e:
            print("Error loading score model/scaler.")
        
    print("\n[Input Differentials Used]")
    for k, v in diffs.items():
        print(f"  {k}: {v:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

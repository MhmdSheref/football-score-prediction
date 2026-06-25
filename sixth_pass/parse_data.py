import pandas as pd
import os

raw_2018 = """France 1,078 26 41.5 28.4 0.5 180 Very High Spread
Brazil 943 26 36.3 32.1 0.4 200 Very High Spread
Spain 876 26 33.7 22.8 0.3 65 High Spread
England 834 26 32.1 26.9 0.2 135 Very High Spread
Germany 816 26 31.4 18.2 0.3 85 High Spread
Argentina 735 26 28.3 21.6 0.2 180 Very High Spread
Belgium 704 26 27.1 19.4 0.2 150 Very High Spread
Portugal 658 26 25.3 16.8 0.1 120 High Spread
Uruguay 450 23 19.6 13.2 0.1 100 High Spread
Croatia 420 23 18.3 8.2 0.1 70 Medium Spread
Mexico 280 23 12.2 5.6 0.1 18 Low-Medium Spread
Colombia 275 23 12.0 6.1 0.1 45 Medium Spread
Switzerland 270 23 11.7 5.8 0.1 30 Low Spread
Denmark 230 23 10.0 4.2 0.1 70 High Spread
Poland 221 23 9.6 4.4 0.1 60 Medium Spread
Sweden 210 23 9.1 4.0 0.1 1 Special Case
South Korea 195 23 8.5 3.2 0.1 40 Medium Spread
Japan 190 23 8.3 3.1 0.1 10 Low Spread
Russia 148 23 6.4 2.2 0.1 10 Low Spread
Australia 140 23 6.1 2.0 0.1 8 Very Low Spread
Senegal 138 23 6.0 2.8 0.1 70 High Spread
Iran 120 23 5.2 1.6 0.1 18 Low Spread
Nigeria 115 23 5.0 1.8 0.1 22 Low Spread
Morocco 110 23 4.8 1.9 0.1 12 Low Spread
Serbia 105 23 4.6 2.1 0.1 20 Low Spread
Peru 100 23 4.3 1.5 0.1 8 Very Low Spread
Costa Rica 95 23 4.1 1.3 0.1 25 Medium Spread
Iceland 80 23 3.5 1.0 0.1 20 Low Spread
Egypt 75 23 3.3 1.1 0.1 100 High Spread
Saudi Arabia 55 23 2.4 0.6 0.1 4 Very Low Spread
Tunisia 50 23 2.2 0.5 0.1 15 Low-Medium Spread
Panama 9 23 0.4 0.1 0.0 1 Minimal"""

raw_2022 = """England 1,490 26 57.3 34.2 0.8 202 Very High Spread
Brazil 1,450 26 55.8 41.1 1.2 200 Very High Spread
France 1,340 25 53.6 38.9 0.9 185 Very High Spread
Spain 1,200 26 46.2 29.4 0.6 120 High Spread
Portugal 1,150 26 44.2 31.2 0.5 110 High Spread
Germany 1,020 26 39.2 24.8 0.4 95 High Spread
Argentina 780 26 30.0 22.1 0.3 50 Medium Spread
Netherlands 760 26 29.2 20.3 0.4 80 Medium Spread
Belgium 720 26 27.7 19.6 0.5 100 High Spread
Uruguay 520 23 22.6 14.2 0.2 45 Medium Spread
USA 485 23 21.1 12.8 0.3 28 Low-Medium Spread
Croatia 460 23 20.0 10.4 0.2 40 Medium Spread
Poland 420 23 18.3 9.2 0.1 45 Medium Spread
Denmark 410 23 17.8 8.6 0.1 20 Low Spread
Mexico 400 23 17.4 8.1 0.2 35 Low-Medium Spread
Switzerland 390 23 17.0 7.9 0.1 25 Low Spread
Morocco 350 23 15.2 7.4 0.1 60 Medium Spread
Senegal 330 23 14.3 6.8 0.1 80 High Spread
Japan 310 23 13.5 6.2 0.1 30 Medium Spread
South Korea 300 23 13.0 5.9 0.1 70 High Spread
Australia 260 23 11.3 4.2 0.1 10 Low Spread
Serbia 250 23 10.9 5.1 0.1 80 High Spread
Canada 240 23 10.4 4.8 0.1 50 Medium Spread
Iran 130 23 5.7 2.1 0.1 20 Low Spread
Ecuador 128 23 5.6 2.0 0.1 35 Medium Spread
Tunisia 120 23 5.2 1.8 0.1 15 Low Spread
Wales 185 23 8.0 4.2 0.1 3 Very Low Spread
Cameroon 170 23 7.4 3.1 0.1 12 Low Spread
Ghana 150 23 6.5 2.8 0.1 25 Medium Spread
Saudi Arabia 95 23 4.1 1.2 0.1 5 Very Low Spread
Qatar 45 23 2.0 0.6 0.1 5 Very Low Spread
Costa Rica 23 23 1.0 0.2 0.1 2 Very Low Spread"""

def parse_raw_text(text, year):
    data = []
    for line in text.strip().split('\n'):
        parts = line.split()
        # Find where the numbers start
        name_parts = []
        for p in parts:
            if not p.replace(',', '').replace('.', '').replace('-', '').isdigit():
                name_parts.append(p)
            else:
                break
        
        team_name = ' '.join(name_parts)
        
        # Name mapping
        if team_name == 'USA':
            team_name = 'United States'
        elif team_name == 'South Korea' and year == 2022:
            team_name = 'Korea Republic'
            
        num_start = len(name_parts)
        
        # Format: Team | Total Value | Players | Avg | StdDev | Min | Max | Spread...
        total_val = float(parts[num_start].replace(',', ''))
        players = int(parts[num_start+1])
        avg_val = float(parts[num_start+2])
        stddev = float(parts[num_start+3])
        min_val = float(parts[num_start+4])
        max_val = float(parts[num_start+5])
        
        star_dep = max_val / total_val if total_val > 0 else 0
        
        data.append({
            'Team': team_name,
            'StdDev': stddev,
            'Min Player': min_val,
            'Max Player': max_val,
            'Star Dependency': star_dep
        })
    return pd.DataFrame(data)

def process_year(year, raw_text):
    new_df = parse_raw_text(raw_text, year)
    
    # Load old teams data
    old_path = f"../fifth_pass/second_pass/data/{year}_teams.csv"
    if not os.path.exists(old_path):
        # Maybe look somewhere else
        old_path = f"../second_pass/data/{year}_teams.csv"
        if not os.path.exists(old_path):
            raise Exception(f"Could not find {year}_teams.csv")
            
    old_df = pd.read_csv(old_path)
    
    # Merge
    merged_df = pd.merge(old_df, new_df, on='Team', how='left')
    
    # Check for missing
    missing = merged_df[merged_df['StdDev'].isna()]
    if len(missing) > 0:
        print(f"Warning: {year} missing data for teams: {missing['Team'].tolist()}")
    
    # Save to sixth_pass
    merged_df.to_csv(f"data/{year}_teams.csv", index=False)
    print(f"Saved {year}_teams.csv with shape {merged_df.shape}")

process_year(2018, raw_2018)
process_year(2022, raw_2022)

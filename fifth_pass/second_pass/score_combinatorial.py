import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load and merge data
years = [2018, 2022]
all_matches = []

features_to_diff = ['Elo', 'xG_for', 'Alt', 'Humid', 'Avg Age', 'Total Value', 'Avg per Player', 'Avg Attend.']
diff_cols = [f.replace(' ', '_').replace('.', '').lower() + '_diff' for f in features_to_diff]

for year in years:
    matches = pd.read_csv(f"{year}_matches.csv")
    teams = pd.read_csv(f"data/{year}_teams.csv")
    
    # Merge team 1
    m1 = pd.merge(matches, teams, left_on='team 1', right_on='Team', how='left')
    m1 = m1.rename(columns={f: f"{f}_1" for f in features_to_diff})
    
    # Merge team 2
    m2 = pd.merge(m1, teams, left_on='team 2', right_on='Team', how='left')
    m2 = m2.rename(columns={f: f"{f}_2" for f in features_to_diff})
    
    # Compute diffs
    for f, d_col in zip(features_to_diff, diff_cols):
        m2[d_col] = pd.to_numeric(m2[f"{f}_1"], errors='coerce') - pd.to_numeric(m2[f"{f}_2"], errors='coerce')
        
    all_matches.append(m2)

df = pd.concat(all_matches, ignore_index=True)
df = df.dropna(subset=['score advantage'] + diff_cols)

# 2. Data Augmentation
flipped = df.copy()
for d_col in diff_cols:
    flipped[d_col] = -flipped[d_col]
flipped['score advantage'] = -flipped['score advantage']

augmented = pd.concat([df, flipped], ignore_index=True)

# 3. Generate all combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

all_combos = list(powerset(diff_cols))
results = []
SEEDS = [42 + i for i in range(50)]

print(f"Testing {len(all_combos)} combinations over {len(SEEDS)} seeds...")

for i, combo in enumerate(all_combos):
    combo = list(combo)
    val_maes = []
    val_rmses = []
    val_r2s = []
    
    X = augmented[combo]
    y = augmented['score advantage']
    
    for rs in SEEDS:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=rs)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=rs)
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('linreg', LinearRegression())
        ])
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        val_maes.append(mean_absolute_error(y_val, preds))
        val_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        val_r2s.append(r2_score(y_val, preds))
        
    results.append({
        'num_features': len(combo),
        'features': ", ".join(combo),
        'val_mae': np.mean(val_maes),
        'val_rmse': np.mean(val_rmses),
        'val_r2': np.mean(val_r2s)
    })
    
    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{len(all_combos)}")

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by='val_rmse', ascending=True)
res_df.to_csv("score_combinatorial_results.csv", index=False)
print("Saved score_combinatorial_results.csv")

# 4. Generate Graph
plt.figure(figsize=(10, 6))
sns.boxplot(data=res_df, x='num_features', y='val_rmse', palette='rocket_r')
plt.title('Score Prediction Validation RMSE by Number of Features')
plt.xlabel('Number of Features in Model')
plt.ylabel('Validation RMSE (50 Seed Avg) - Lower is Better')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('score_feature_combinations.png', dpi=300)
print("Saved score_feature_combinations.png")

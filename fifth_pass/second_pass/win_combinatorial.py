import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
df = df.dropna(subset=['win'] + diff_cols)
df['win'] = df['win'].astype(int)

# 2. Data Augmentation
flipped = df.copy()
for d_col in diff_cols:
    flipped[d_col] = -flipped[d_col]
flipped['win'] = 1 - flipped['win']

augmented = pd.concat([df, flipped], ignore_index=True)
X_full = augmented[diff_cols]
y_full = augmented['win']

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
    val_accs = []
    
    X = augmented[combo]
    y = augmented['win']
    
    for rs in SEEDS:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=rs)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=rs)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        val_accs.append(accuracy_score(y_val, preds))
        
    avg_val = np.mean(val_accs)
    results.append({
        'num_features': len(combo),
        'features': ", ".join(combo),
        'val_accuracy': avg_val
    })
    
    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{len(all_combos)}")

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by='val_accuracy', ascending=False)
res_df.to_csv("win_combinatorial_results.csv", index=False)
print("Saved win_combinatorial_results.csv")

# 4. Generate Graph
plt.figure(figsize=(10, 6))
sns.boxplot(data=res_df, x='num_features', y='val_accuracy', palette='viridis')
plt.title('Win Prediction Validation Accuracy by Number of Features')
plt.xlabel('Number of Features in Model')
plt.ylabel('Validation Accuracy (50 Seed Avg)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('win_feature_combinations.png', dpi=300)
print("Saved win_feature_combinations.png")

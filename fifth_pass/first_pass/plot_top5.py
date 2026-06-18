import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('experiment_results.csv')

# Separate by task
clf_df = df[df['task_type'] == 'classification'].copy()
reg_df = df[df['task_type'] == 'regression'].copy()

# Rank them (1 is best)
clf_df['clf_rank'] = clf_df['val_metric'].rank(method='min', ascending=False)
reg_df['reg_rank'] = reg_df['val_metric'].rank(method='min', ascending=True)

# Define a key for merging (architecture + hyperparameters)
key_cols = ['arch', 'width', 'activation', 'dropout', 'l2', 'optimizer']
clf_df['config_key'] = clf_df[key_cols].astype(str).agg('-'.join, axis=1)
reg_df['config_key'] = reg_df[key_cols].astype(str).agg('-'.join, axis=1)

# Merge on the key to find overall rank
merged = pd.merge(clf_df[['config_key', 'clf_rank', 'val_metric'] + key_cols], 
                  reg_df[['config_key', 'reg_rank', 'val_metric']], 
                  on='config_key', 
                  suffixes=('_clf', '_reg'))

# Calculate overall score (lower is better)
merged['overall_rank'] = merged['clf_rank'] + merged['reg_rank']

# Get top 5 overall
top_overall = merged.sort_values('overall_rank').head(5).copy()

def readable_name(row):
    arch_type = "Shallow" if row['arch'] == 1 else "Medium" if row['arch'] == 2 else "Deep"
    width_type = "Narrow" if row['width'] <= 16 else "Wide" if row['width'] >= 64 else "Standard"
    act = row['activation'].replace('_', ' ').title()
    opt = row['optimizer'].title()
    drop_str = " with Dropout" if row['dropout'] > 0 else ""
    return f"{arch_type} {width_type} {act} Network ({opt}){drop_str}"

top_overall['Readable Name'] = top_overall.apply(readable_name, axis=1)

# To visualize we can plot their classification and regression ranks 
top_overall_melted = top_overall.melt(id_vars=['Readable Name'], value_vars=['clf_rank', 'reg_rank'], var_name='Task', value_name='Rank')
top_overall_melted['Task'] = top_overall_melted['Task'].map({'clf_rank': 'Win Prediction Rank', 'reg_rank': 'Score Adv. Rank'})

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

ax = sns.barplot(data=top_overall_melted, x='Rank', y='Readable Name', hue='Task', palette='muted')
plt.title('Top 5 Overall Architectures (Ranked across both tasks - Lower is Better)', fontsize=14)
plt.xlabel('Leaderboard Rank (Lower = Better)', fontsize=12)
plt.ylabel('')
plt.xlim(0, max(top_overall_melted['Rank']) + 5)

for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.text(width + 0.3, p.get_y() + p.get_height()/2. + 0.05, f'#{int(width)}', ha="left", va="center")

plt.tight_layout()
plt.savefig('top5_models.png', dpi=300)
print("Saved top5_models.png")

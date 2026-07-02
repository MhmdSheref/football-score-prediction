import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Load actuals and predictions
    actuals = pd.read_csv('2026_actuals.csv')
    preds = pd.read_csv('2026_predictions_clean.csv')
    
    # Merge on Team 1 and Team 2
    # Ensure standard names
    actuals['Team 1'] = actuals['Team 1'].str.strip()
    actuals['Team 2'] = actuals['Team 2'].str.strip()
    preds['Team 1'] = preds['Team 1'].str.strip()
    preds['Team 2'] = preds['Team 2'].str.strip()
    
    # Left join to preserve all actuals (should be 72)
    df = pd.merge(actuals, preds, on=['Team 1', 'Team 2'], how='left')
    
    # Calculate margins
    df['Actual Margin'] = df['Team 1 Score'] - df['Team 2 Score']
    df['Predicted Margin'] = df['Expected Score Advantage for T1']
    
    # Accuracies
    df['Predicted Winner Binary'] = df.apply(lambda row: row['Team 1'] if row['Predicted Margin'] > 0 else row['Team 2'], axis=1)
    
    # Strict accuracy: ties are considered "Draw", so if predicted is Team 1 or Team 2, it's a fail.
    # Our model only predicts a winner, so any "Draw" in actuals is automatically a misclassification.
    df['Correct_Strict'] = df['Winner'] == df['Predicted Winner Binary']
    strict_acc = df['Correct_Strict'].mean()
    
    # Accuracy excluding ties
    decisive_df = df[df['Winner'] != 'Draw']
    decisive_acc = decisive_df['Correct_Strict'].mean()
    
    print(f"Total Matches: {len(df)}")
    print(f"Total Decisive Matches: {len(decisive_df)}")
    print(f"Total Draws: {len(df) - len(decisive_df)}")
    print(f"Strict Accuracy (including draws as fails): {strict_acc:.1%}")
    print(f"Decisive Accuracy (excluding draws): {decisive_acc:.1%}")
    
    # Regression Metrics
    df['Margin Error'] = df['Predicted Margin'] - df['Actual Margin']
    df['Abs Error'] = df['Margin Error'].abs()
    
    mae = df['Abs Error'].mean()
    rmse = np.sqrt((df['Margin Error']**2).mean())
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Top 5 Outliers (highest absolute error)
    print("\n--- TOP 5 OUTLIERS ---")
    outliers = df.sort_values(by='Abs Error', ascending=False).head(5)
    for idx, row in outliers.iterrows():
        print(f"{row['Team 1']} vs {row['Team 2']}: Actual Score {row['Team 1 Score']}-{row['Team 2 Score']} (Margin {row['Actual Margin']}). Predicted Adv {row['Predicted Margin']}. Error {row['Abs Error']:.2f}")

    # Generate Graphs
    # 1. Bar chart for Accuracy
    plt.figure(figsize=(6, 5))
    labels = ['With Ties', 'Excluding Ties']
    accs = [strict_acc * 100, decisive_acc * 100]
    bars = plt.bar(labels, accs, color=['#e74c3c', '#2ecc71'])
    plt.ylabel('Accuracy (%)')
    plt.title('Win Prediction Accuracy on 2026 1st Round')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.savefig('accuracy_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot of Predicted vs Actual Margin
    plt.figure(figsize=(8, 6))
    
    # Highlight top 5 outliers in red
    top_outliers_idx = outliers.index
    regular_df = df.drop(index=top_outliers_idx)
    
    plt.scatter(regular_df['Actual Margin'], regular_df['Predicted Margin'], color='blue', alpha=0.6, label='Matches')
    plt.scatter(outliers['Actual Margin'], outliers['Predicted Margin'], color='red', s=80, marker='x', label='Top 5 Outliers')
    
    # Add text for outliers
    for idx, row in outliers.iterrows():
        plt.text(row['Actual Margin'] + 0.1, row['Predicted Margin'] - 0.2, f"{row['Team 1']} vs {row['Team 2']}", fontsize=8)
    
    # Perfect prediction line
    min_val = min(df['Actual Margin'].min(), df['Predicted Margin'].min()) - 1
    max_val = max(df['Actual Margin'].max(), df['Predicted Margin'].max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.xlabel('Actual Goal Margin (T1 - T2)')
    plt.ylabel('Predicted Goal Margin (T1 - T2)')
    plt.title('Predicted vs Actual Goal Margin (2026 1st Round)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('margin_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved graphs to accuracy_bar.png and margin_scatter.png")

if __name__ == "__main__":
    main()

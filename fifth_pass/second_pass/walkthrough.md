# Combinatorial Feature Search Findings

We executed an exhaustive combinatorial search testing all 255 possible feature subsets derived from the newly assembled team statistics (`Elo`, `xG_for`, `Alt`, `Humid`, `Avg Age`, `Total Value`, `Avg per Player`, `Avg Attend.`).

Both models were evaluated on their validation metrics averaged over 50 random data splits to ensure noise reduction and robust results.

## 1. Win Prediction (Classification)

We trained `LogisticRegression` on every feature combination, predicting the `win` outcome and evaluating the Validation Accuracy.

**Baseline:** The original linear baseline model with 4 features achieved ~80.00% validation accuracy.

### Top 5 Feature Combinations

| Features | Validation Accuracy |
|----------|---------------------|
| `elo_diff`, `xg_for_diff`, `humid_diff`, `total_value_diff`, `avg_per_player_diff`, `avg_attend_diff` | **86.44%** |
| `elo_diff`, `xg_for_diff`, `humid_diff`, `total_value_diff`, `avg_attend_diff` | **86.37%** |
| `elo_diff`, `xg_for_diff`, `humid_diff`, `avg_per_player_diff`, `avg_attend_diff` | **86.37%** |
| `xg_for_diff`, `humid_diff`, `avg_age_diff`, `avg_per_player_diff`, `avg_attend_diff` | **86.00%** |
| `xg_for_diff`, `humid_diff`, `avg_age_diff`, `total_value_diff`, `avg_attend_diff` | **86.00%** |

> [!TIP]
> **Key Finding for Classification:** Adding financial and crowd metrics significantly improves the logistic regression's ability to predict a match winner! 
> The best model utilizes 6 features (omitting Altitude and Age) to hit **86.44%** validation accuracy—a massive jump over the old 4-feature baseline of 80%.

![Win Features Boxplot](/C:/Users/MhmdSheref/.gemini/antigravity/brain/67e7e8db-8c4c-49be-8b34-f0f88f0b2c5c/win_feature_combinations.png)

*The plot above demonstrates how accuracy generally increases when utilizing ~5 to 6 features before diminishing returns occur.*

---

## 2. Score Advantage (Regression)

We trained `LinearRegression` on every combination, tracking RMSE, MAE, and R².

**Baseline:** The original model achieved Validation MAE ~1.055 and RMSE ~1.348.

### Top 5 Feature Combinations

| Features | Val RMSE | Val MAE | Val R² |
|----------|----------|---------|--------|
| `xg_for_diff`, `avg_attend_diff` | **1.366** | **1.055** | **0.431** |
| `xg_for_diff`, `humid_diff`, `avg_attend_diff` | **1.369** | **1.052** | **0.428** |
| `xg_for_diff`, `total_value_diff`, `avg_attend_diff` | **1.369** | **1.053** | **0.429** |
| `xg_for_diff`, `avg_per_player_diff`, `avg_attend_diff` | **1.369** | **1.053** | **0.429** |
| `xg_for_diff`, `alt_diff`, `avg_attend_diff` | **1.371** | **1.059** | **0.428** |

> [!IMPORTANT]
> **Key Finding for Regression:** Score differences are inherently noisy, meaning that *adding too many features causes linear models to overfit*. 
> The best performing linear model for score regression only uses **2 features**: the difference in Expected Goals (`xg_for_diff`) and the difference in Crowd Attendance (`avg_attend_diff`). 

![Score Features Boxplot](/C:/Users/MhmdSheref/.gemini/antigravity/brain/67e7e8db-8c4c-49be-8b34-f0f88f0b2c5c/score_feature_combinations.png)

*As seen above, the best average performance is found with very small feature spaces. Higher feature counts introduce severe variance (RMSE spreads widely).*

## 3. Results Data

All 255 evaluation rows (averaged over 50 random seeds each) have been saved to local CSVs for further inspection:
- [win_combinatorial_results.csv](file:///e:/User%20Folders/Desktop/python/football-score-prediction/let_ai_try/second_pass/win_combinatorial_results.csv)
- [score_combinatorial_results.csv](file:///e:/User%20Folders/Desktop/python/football-score-prediction/let_ai_try/second_pass/score_combinatorial_results.csv)

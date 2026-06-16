# Football Match Prediction: Elo, Altitude, and Expected Goals (xG)

This project explores predicting FIFA World Cup match outcomes (win/loss) and score advantages (goal margins) using match data from the 2018 and 2022 World Cups. The project progressed through three distinct phases of feature engineering, data cleaning, and model experimentation.

---

## Experiment Phases & Results

### 1. First Pass: Baseline Elo Ratings
Comparing predictions using baseline Elo ratings:
* **Exact Elo:** Uses both teams' Elo ratings as separate features.
* **Elo Difference:** Uses `team_1_elo - team_2_elo` as a single feature.

| Model | Task | Accuracy / R² | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **Elo Difference** | Win/Loss (Logistic Regression) | **83.3%** | - | - |
| **Exact Elo** | Win/Loss (Logistic Regression) | **83.3%** | - | - |
| **Elo Difference** | Score Advantage (Linear Regression) | **R²: 33.1%** | 1.033 | 1.238 |
| **Exact Elo** | Score Advantage (Linear Regression) | **R²: 32.4%** | 1.037 | 1.244 |

---

### 2. Second Pass: Feature Engineering (Altitude & Expected Goals)
We introduced new data points to improve prediction strength:
* **Open Play Goals:** Adding open play goal counts to the Elo score maintained the win/loss accuracy at **83.3%**.
* **Team Goal Breakdowns:** Adding averaged-out breakdown of how teams usually score led to a decrease in win/loss accuracy down to **73.3%** (overfitting/noise).
* **Altitude (Elevation):** Adding the altitude of each team's country of origin.
  * **Win/Loss Accuracy:** Increased to **86.7%**.
  * **Score Advantage:** R² improved slightly to **34.2%** (with a slightly lower RMSE and similar MAE).
* **Expected Goals (xG):** Incorporating expected goals for, against, and difference. 
  * The most promising single combination was **Elo Difference + xG_For**, yielding **86.7%** win/loss accuracy and a major jump in score prediction to **R²: 41.2%** (MAE: 0.923, RMSE: 1.161).
* **Combined Model (Elo Diff + Altitude + xG_For):**
  * **Win/Loss Accuracy:** Remained at **86.7%**.
  * **Score Advantage:** Improved further to our best baseline R² of **42.7%** (MAE: 0.922, RMSE: 1.146).

---

### 3. Third Pass: Data Cleaning (Removing Low-Stakes Group Matches)
We investigated whether removing matches where the motivation to win was not guaranteed (specifically, the last match in the group stage for teams that might rest players or strategically play for draws/specific bracket slots) improved predictability.

* **Dataset Used:** Matches with these dubious group-stage games removed (`_cleaned.csv` datasets).
* **Win/Loss Accuracy:** Rose significantly to **89.2%**.
* **Score Advantage:** Achieved a peak **R² of 44.5%**, though the overall errors increased to **MAE: 1.15** and **RMSE: 1.45** (likely due to a smaller sample size).

*Note: This clean/all distinction suggests it is worth testing again if we transition to more complex architectures.*

---

## Summary of Metric Progression (Validation Set)

### Win/Loss Prediction Accuracy
```
First Pass (Elo Diff)  [========================] 83.3%
Second Pass (Elo+Alt)  [==========================] 86.7%
Second Pass (Elo+xG)   [==========================] 86.7%
Second Pass (Combined) [==========================] 86.7%
Third Pass (Cleaned)   [============================] 89.2%
```

### Score Advantage Prediction (R² Score)
```
First Pass (Elo Diff)  [==================] 33.1%
Second Pass (Elo+Alt)  [===================] 34.2%
Second Pass (Elo+xG)   [=======================] 41.2%
Second Pass (Combined) [========================] 42.7%
Third Pass (Cleaned)   [==========================] 44.5%
```

---

## Project Structure & Scripts

```
football-score-prediction/
├── first_pass/                # Initial baseline Elo rating models
│   ├── win elo diff.py
│   ├── win elo exact.py
│   ├── score elo diff.py
│   └── score elo exact.py
│
├── second_pass/               # Experiments adding altitude & expected goals (xG)
│   ├── win elo alt.py         # Win/Loss using Elo + Altitude
│   ├── win elo xg alt.py      # Win/Loss using Elo + xG + Altitude
│   ├── win xg combinations.py # Script testing combinations of xG features
│   ├── score elo alt.py       # Score advantage using Elo + Altitude
│   ├── score elo xg alt.py    # Score advantage using Elo + xG + Altitude
│   └── score xg combinations.py
│
└── third_pass/                # Cleaning matches & introductory ANN models
    ├── win cleaned.py         # Multi-seed Logistic Regression evaluation on cleaned data
    ├── win all.py             # Single seed evaluation on cleaned data
    ├── score cleaned.py       # Linear Regression evaluation on cleaned data
    ├── win ANN.py             # TensorFlow binary classification ANN sweep
    └── score ANN.py           # TensorFlow regression ANN sweep
```

---

## Setup & Running the Scripts

### Dependencies
Install the required Python packages:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

### Running Experiments
Navigate to the desired pass folder and run scripts to evaluate:
```bash
# E.g., Running third pass evaluations
cd third_pass
python "win cleaned.py"
python "score cleaned.py"
python "win ANN.py"
```

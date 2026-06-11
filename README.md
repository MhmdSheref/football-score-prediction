# Football Match Prediction with Elo Ratings

This project explores how well international football Elo ratings can predict
World Cup match outcomes and score margins. It combines match data from the
2018 and 2022 FIFA World Cups and compares two feature representations:

- **Exact Elo:** uses both teams' Elo ratings as separate features.
- **Elo difference:** uses `team 1 elo - team 2 elo` as one feature.

The models cover two tasks:

1. **Win prediction** with logistic regression.
2. **Score-advantage prediction** with linear regression.

## Results

### Win prediction

| Model | Accuracy |
| --- | ---: |
| Elo difference | **83.33%** |
| Exact Elo ratings | **83.33%** |

Both representations achieved the same reported accuracy. This suggests that
the relative gap between the teams contains the important signal for this
linear classification task.

### Score-advantage prediction

| Model | MAE | RMSE | R^2 |
| --- | ---: | ---: | ---: |
| Elo difference | **1.033** | **1.238** | **0.331** |
| Exact Elo ratings | 1.037 | 1.244 | 0.324 |

The Elo-difference model performed slightly better across all three reported
regression metrics. Its predictions miss the actual goal margin by about
**1.03 goals on average**, while explaining roughly **33.1%** of the variance.

The estimated function for the Elo-difference model is as follows: score_advantage = 0.004821 * elo_diff + -0.020733
This can be used to approximate score advantage where every **207.426** point difference in elo correlates to an extra predicted goal advantage.

> These results come from a small dataset covering two World Cups, so they
> should be treated as an experiment rather than production-level estimates.

## Visualizations

### Win probability from Elo difference

![Logistic regression win-probability curve](win%20elo%20diff%20curve.png)

### Win-model validation

| Elo difference | Exact Elo ratings |
| --- | --- |
| ![Elo-difference win validation](win%20elo%20diff%20valid.png) | ![Exact-Elo win validation](win%20elo%20exact%20valid.png) |

### Score-model fits

| Elo difference | Exact Elo ratings |
| --- | --- |
| ![Elo-difference regression line](score%20elo%20diff%20line.png) | ![Exact-Elo regression plane](score%20elo%20exact%20plane.png) |

### Score-model validation

| Elo difference | Exact Elo ratings |
| --- | --- |
| ![Elo-difference score validation](score%20elo%20diff%20valid.png) | ![Exact-Elo score validation](score%20elo%20exact%20valid.png) |

## Method

The scripts:

1. Combine `2018.csv` and `2022.csv`.
2. Remove rows missing the required target.
3. Augment the data by reversing each matchup:
   - Team Elo ratings are swapped, or the Elo difference is negated.
   - Win labels are inverted.
   - Score advantages are negated.
4. Split the augmented data into 70% training, 15% validation, and 15% test
   sets using `random_state=42`.
5. Train and evaluate a logistic- or linear-regression model.

## Project Files

| File | Purpose |
| --- | --- |
| `2018.csv`, `2022.csv` | World Cup match data and Elo ratings |
| `win elo diff.py` | Win prediction from Elo difference |
| `win elo exact.py` | Win prediction from two Elo ratings |
| `score elo diff.py` | Score-margin prediction from Elo difference |
| `score elo exact.py` | Score-margin prediction from two Elo ratings |
| `*.png` | Generated model and validation visualizations |

## Running the Experiments

Install the dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run any experiment from the project directory:

```bash
python "win elo diff.py"
python "win elo exact.py"
python "score elo diff.py"
python "score elo exact.py"
```

Each script prints its evaluation metrics and displays the relevant plots.

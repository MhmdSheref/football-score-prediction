# Football Match Prediction: AI Agent Exploration

This project explores predicting FIFA World Cup match outcomes (win/loss) and expected goal margins using match data spanning the 2018 and 2022 World Cups. The project progressed from baseline statistical models into exhaustive Neural Network hyperparameter tuning, resulting in highly optimized, production-ready prediction models.

---

## 🏆 Final Model Performance & Findings

Following our initial baselines, we shifted to an exhaustive, programmatic search across 255 distinct feature combinations and hundreds of Neural Network architectures to find the absolute performance ceiling for this dataset.

Our most critical finding was the **divergence in feature sensitivity between classification and regression**:

### 1. Win Prediction (Classification)
* **Architecture:** 32-node Neural Network (Swish activation, L2 Regularization, 0.2 Dropout).
* **Optimal Features (7):** `Altitude`, `Attendance`, `Avg Player Value`, `Elo`, `Humidity`, `Total Squad Value`, `Expected Goals (xG)`.
* **Validation Performance:** Maintained an impressive **87.4% accuracy** when trained exclusively on 2018 data and validated against the entirely unseen 2022 dataset. Classification thrived on having more context (7 variables).

### 2. Score Margin Prediction (Regression)
* **Architecture:** Narrow 16-node Neural Network (Swish activation, heavier L2 Regularization).
* **Optimal Features (3):** `Attendance`, `Avg Player Value`, `Expected Goals (xG)`.
* **Validation Performance:** Achieved an **R² of 52.4%**. Unlike classification, regression proved extraordinarily sensitive to noise. Introducing additional parameters like Elo or Altitude caused immediate overfitting and catastrophic degradation on unseen validation data.

### 3. Cross-Year Generalization
By intentionally training models strictly on 2018 match data and testing them on 2022 data, we verified that these specific feature-architecture pairings generalize remarkably well across 4-year temporal gaps.

---

## 📂 Project Structure

The project has evolved into an experimental pipeline, heavily focused inside the `fifth_pass/` directory where the Deep Neural Network search was conducted:

```text
football-score-prediction/
├── fifth_pass/                     # Deep Learning & Feature Optimization
│   ├── first_pass/                 # Legacy baseline regressions
│   ├── second_pass/                # Combinatorial feature & NN architecture sweep
│   ├── third_phase/                # Strict cross-year (2018 -> 2022) validation
│   ├── report/                     # High-quality performance graphs & heatmaps
│   └── final_models/               # Production-ready models & inference tools
│       ├── build_models.py         # Trains the final NNs on the combined 18+22 dataset
│       ├── predictor.py            # Interactive CLI for running predictions
│       ├── predict_2026.py         # Batch predictor for 2026 WC matches
│       └── *.keras / *.pkl         # Serialized models and scalers
│
└── 2018.csv, 2022.csv              # Raw datasets
```

---

## 🚀 How to Use the Predictor

We provide a robust, interactive Command Line Interface to predict any matchup using our finalized neural networks.

### Dependencies
```bash
pip install pandas numpy tensorflow
```

### Interactive CLI Mode
Run the predictor with no arguments to launch an interactive session. It will prompt you for the team names and allow you to dynamically override their core stats. If a team is in our 2026, 2022, or 2018 database, the script will automatically load their historical stats by default!

```bash
python fifth_pass/final_models/predictor.py
```

### Command Line Overrides
You can bypass the interactive prompts entirely and pass teams and parameters as arguments. This is perfect for "What-if" scenarios:

```bash
# Predict Argentina vs France using their historical defaults
python fifth_pass/final_models/predictor.py --team1 "Argentina" --team2 "France"

# See what happens if France had a much higher Expected Goals (xG)
python fifth_pass/final_models/predictor.py --team1 "Argentina" --team2 "France" --xg2 2.5
```

### 2026 Batch Predictions
We have already processed 72 matches for the upcoming 2026 World Cup. You can view the raw output, including win probabilities and expected goal margins for every match, by opening:
`fifth_pass/final_models/2026_predictions.csv`

To regenerate or modify these batch predictions yourself:
```bash
python fifth_pass/final_models/predict_2026.py
```

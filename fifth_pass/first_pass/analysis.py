import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RESULTS_FILE = "experiment_results.csv"
FILES = ["2022_humid_clean.csv", "2018_humid_clean.csv"]

def load_and_prep(task_type, feature_set="basic"):
    dfs = [pd.read_csv(file) for file in FILES]
    df = pd.concat(dfs, ignore_index=True)
    
    target = "win" if task_type == "classification" else "score advantage"
    
    df["elo_diff"] = df["team 1 elo"] - df["team 2 elo"]
    df["alt_diff"] = df["alt 1"] - df["alt 2"]
    df["xg_diff"] = df["team 1 xg for"] - df["team 2 xg for"]
    df["humid_diff"] = df["humid 1"] - df["humid 2"]
    
    features = ["elo_diff", "alt_diff", "xg_diff", "humid_diff"]
    
    if feature_set == "interaction" or feature_set == "both":
        df["elo_xg"] = df["elo_diff"] * df["xg_diff"]
        df["elo_humid"] = df["elo_diff"] * df["humid_diff"]
        df["elo_alt"] = df["elo_diff"] * df["alt_diff"]
        df["xg_humid"] = df["xg_diff"] * df["humid_diff"]
        df["xg_alt"] = df["xg_diff"] * df["alt_diff"]
        df["humid_alt"] = df["humid_diff"] * df["alt_diff"]
        features.extend(["elo_xg", "elo_humid", "elo_alt", "xg_humid", "xg_alt", "humid_alt"])
    if feature_set == "absolute" or feature_set == "both":
        df["abs_elo"] = df["elo_diff"].abs()
        df["abs_xg"] = df["xg_diff"].abs()
        df["abs_alt"] = df["alt_diff"].abs()
        df["abs_humid"] = df["humid_diff"].abs()
        features.extend(["abs_elo", "abs_xg", "abs_alt", "abs_humid"])
        
    cols = ["team 1 elo", "team 2 elo", "alt 1", "alt 2", "team 1 xg for", "team 2 xg for", "humid 1", "humid 2", target]
    df = df.dropna(subset=cols)
    
    if task_type == "classification":
        df[target] = df[target].astype(int)
        
    flipped = df.copy()
    diff_feats = ["elo_diff", "alt_diff", "xg_diff", "humid_diff"]
    for f in diff_feats:
        flipped[f] = -flipped[f]
        
    if task_type == "classification":
        flipped[target] = 1 - flipped[target]
    else:
        flipped[target] = -flipped[target]
        
    df_aug = pd.concat([df, flipped], ignore_index=True)
    X = df_aug[features].values
    y = df_aug[target].values
    return X, y, features

def build_model(input_dim, task_type, layers=3, width=64, activation="relu", dropout=0.0, l2_reg=0.0, optimizer_name="adam", lr=0.001):
    model = keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    
    current_width = width
    for i in range(layers):
        if l2_reg > 0:
            model.add(Dense(current_width, activation=activation, kernel_regularizer=l2(l2_reg)))
        else:
            model.add(Dense(current_width, activation=activation))
        
        if dropout > 0:
            model.add(Dropout(dropout))
            
        current_width = max(8, current_width // 2)
        
    if task_type == "classification":
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(1, activation="linear"))
        loss = "mse"
        
    if optimizer_name.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name.lower() == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=lr)
    elif optimizer_name.lower() == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr)
        
    model.compile(optimizer=opt, loss=loss)
    return model

from sklearn.base import BaseEstimator

class KerasWrapper(BaseEstimator):
    def __init__(self, model, task_type):
        self.model = model
        self.task_type = task_type
        self._estimator_type = "classifier" if task_type == "classification" else "regressor"
        
    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = self.model.predict(X, verbose=0).flatten()
        if self.task_type == "classification":
            return (preds > 0.5).astype(int)
        return preds

def run_analysis():
    df = pd.read_csv(RESULTS_FILE)
    
    clf_df = df[df["task_type"] == "classification"]
    reg_df = df[df["task_type"] == "regression"]
    
    best_clf = clf_df.loc[clf_df["val_metric"].idxmax()]
    best_reg = reg_df.loc[reg_df["val_metric"].idxmin()]
    
    print("\n--- BEST CLASSIFICATION MODEL ---")
    print(best_clf)
    
    print("\n--- BEST REGRESSION MODEL ---")
    print(best_reg)
    
    # Analyze best clf
    X, y, feat_names = load_and_prep("classification", best_clf["features"])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    model = build_model(
        X_train.shape[1], "classification", 
        layers=int(best_clf["arch"]), width=int(best_clf["width"]), activation=best_clf["activation"],
        dropout=float(best_clf["dropout"]), l2_reg=float(best_clf["l2"]), optimizer_name=best_clf["optimizer"], lr=float(best_clf["lr"])
    )
    
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, callbacks=[es, rlr], verbose=0)
    
    wrapper = KerasWrapper(model, "classification")
    res = permutation_importance(wrapper, X_val, y_val, scoring='accuracy', n_repeats=10, random_state=42)
    
    plt.figure(figsize=(10, 6))
    sorted_idx = res.importances_mean.argsort()
    plt.barh(np.array(feat_names)[sorted_idx], res.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (Accuracy decrease)")
    plt.title("Feature Importance - Best Classification Model")
    plt.tight_layout()
    plt.savefig("clf_feature_importance.png")
    plt.close()
    
    # Analyze best reg
    X, y, feat_names = load_and_prep("regression", best_reg["features"])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    model = build_model(
        X_train.shape[1], "regression", 
        layers=int(best_reg["arch"]), width=int(best_reg["width"]), activation=best_reg["activation"],
        dropout=float(best_reg["dropout"]), l2_reg=float(best_reg["l2"]), optimizer_name=best_reg["optimizer"], lr=float(best_reg["lr"])
    )
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, callbacks=[es, rlr], verbose=0)
    
    wrapper = KerasWrapper(model, "regression")
    res = permutation_importance(wrapper, X_val, y_val, scoring='neg_mean_absolute_error', n_repeats=10, random_state=42)
    
    plt.figure(figsize=(10, 6))
    sorted_idx = res.importances_mean.argsort()
    plt.barh(np.array(feat_names)[sorted_idx], res.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (MAE decrease)")
    plt.title("Feature Importance - Best Regression Model")
    plt.tight_layout()
    plt.savefig("reg_feature_importance.png")
    plt.close()
    
    print("Analysis complete. Saved importance plots to disk.")

if __name__ == "__main__":
    run_analysis()

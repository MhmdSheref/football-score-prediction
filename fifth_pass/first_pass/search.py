import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, log_loss
import uuid
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Config
FILES = ["2022_humid_clean.csv", "2018_humid_clean.csv"]
SEEDS = [42, 43, 44, 45, 46]
MAX_EPOCHS = 500
RESULTS_FILE = "experiment_results.csv"

# Write headers if not exists
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("run_id,task_type,batch,arch,width,activation,dropout,l2,optimizer,lr,features,val_metric,val_sec_metric1,val_sec_metric2,test_metric,test_sec_metric1,test_sec_metric2\n")

def load_and_prep(task_type, feature_set="basic"):
    dfs = [pd.read_csv(file) for file in FILES]
    df = pd.concat(dfs, ignore_index=True)
    
    target = "win" if task_type == "classification" else "score advantage"
    
    # Base differences
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

def run_experiment(task_type, batch_name, params):
    X, y, feats = load_and_prep(task_type, params.get("features", "basic"))
    
    val_m1, val_m2, val_m3 = [], [], []
    test_m1, test_m2, test_m3 = [], [], []
    
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=(y if task_type == "classification" else None))
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=(y_temp if task_type == "classification" else None))
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        model = build_model(
            input_dim=X_train.shape[1],
            task_type=task_type,
            layers=params.get("layers", 3),
            width=params.get("width", 64),
            activation=params.get("activation", "relu"),
            dropout=params.get("dropout", 0.0),
            l2_reg=params.get("l2", 0.0),
            optimizer_name=params.get("optimizer", "adam"),
            lr=params.get("lr", 0.001)
        )
        
        es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=MAX_EPOCHS, batch_size=32, callbacks=[es, rlr], verbose=0)
        
        val_preds = model.predict(X_val, verbose=0).flatten()
        test_preds = model.predict(X_test, verbose=0).flatten()
        
        if task_type == "classification":
            val_preds_bin = (val_preds > 0.5).astype(int)
            test_preds_bin = (test_preds > 0.5).astype(int)
            
            val_m1.append(accuracy_score(y_val, val_preds_bin))
            try:
                val_m2.append(roc_auc_score(y_val, val_preds))
                val_m3.append(log_loss(y_val, val_preds))
            except:
                val_m2.append(0)
                val_m3.append(0)
                
            test_m1.append(accuracy_score(y_test, test_preds_bin))
            try:
                test_m2.append(roc_auc_score(y_test, test_preds))
                test_m3.append(log_loss(y_test, test_preds))
            except:
                test_m2.append(0)
                test_m3.append(0)
        else:
            val_m1.append(mean_absolute_error(y_val, val_preds))
            val_m2.append(np.sqrt(mean_squared_error(y_val, val_preds)))
            val_m3.append(r2_score(y_val, val_preds))
            
            test_m1.append(mean_absolute_error(y_test, test_preds))
            test_m2.append(np.sqrt(mean_squared_error(y_test, test_preds)))
            test_m3.append(r2_score(y_test, test_preds))
            
    run_id = str(uuid.uuid4())[:8]
    
    vm1, vm2, vm3 = np.mean(val_m1), np.mean(val_m2), np.mean(val_m3)
    tm1, tm2, tm3 = np.mean(test_m1), np.mean(test_m2), np.mean(test_m3)
    
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{run_id},{task_type},{batch_name},{params.get('layers',3)},{params.get('width',64)},{params.get('activation','relu')},{params.get('dropout',0.0)},{params.get('l2',0.0)},{params.get('optimizer','adam')},{params.get('lr',0.001)},{params.get('features','basic')},{vm1:.4f},{vm2:.4f},{vm3:.4f},{tm1:.4f},{tm2:.4f},{tm3:.4f}\n")
        
    print(f"[{task_type.upper()}] Run {run_id} | {batch_name} | Val_M1: {vm1:.4f} | Test_M1: {tm1:.4f} | Params: {params}")
    return {"val_m1": vm1, "val_m2": vm2, "val_m3": vm3, "test_m1": tm1, "params": params, "run_id": run_id}

def run_classification_search():
    best_val_acc = 0
    best_params = {}
    
    # Batch 1: Baseline
    res = run_experiment("classification", "batch_1_baseline", {"layers": 3, "width": 64})
    best_val_acc = res["val_m1"]
    best_params = res["params"]
    
    # Batch 2: Architecture
    for layers in [1, 2, 4]:
        for width in [16, 32, 128]:
            res = run_experiment("classification", "batch_2_arch", {"layers": layers, "width": width})
            if res["val_m1"] > best_val_acc:
                best_val_acc = res["val_m1"]
                best_params = res["params"]
                
    # Batch 3: Activation & Regularization (using best arch)
    for act in ["leaky_relu", "swish"]:
        for drop in [0.0, 0.2, 0.5]:
            for l2r in [0.0, 1e-4, 1e-2]:
                params = best_params.copy()
                params["activation"] = act
                params["dropout"] = drop
                params["l2"] = l2r
                res = run_experiment("classification", "batch_3_reg", params)
                if res["val_m1"] > best_val_acc:
                    best_val_acc = res["val_m1"]
                    best_params = res["params"]
                    
    # Batch 4: Optimizer & LR (using best previous params)
    for opt in ["adamw", "rmsprop"]:
        for lr in [1e-2, 3e-3, 3e-4]:
            params = best_params.copy()
            params["optimizer"] = opt
            params["lr"] = lr
            res = run_experiment("classification", "batch_4_opt", params)
            if res["val_m1"] > best_val_acc:
                best_val_acc = res["val_m1"]
                best_params = res["params"]
                
    # Batch 5: Features (using best previous params)
    for feat in ["interaction", "absolute", "both"]:
        params = best_params.copy()
        params["features"] = feat
        res = run_experiment("classification", "batch_5_feat", params)
        if res["val_m1"] > best_val_acc:
            best_val_acc = res["val_m1"]
            best_params = res["params"]

def run_regression_search():
    best_val_mae = float("inf")
    best_params = {}
    
    # Batch 1: Baseline
    res = run_experiment("regression", "batch_1_baseline", {"layers": 3, "width": 64})
    best_val_mae = res["val_m1"]
    best_params = res["params"]
    
    # Batch 2: Architecture
    for layers in [1, 2, 4]:
        for width in [16, 32, 128]:
            res = run_experiment("regression", "batch_2_arch", {"layers": layers, "width": width})
            if res["val_m1"] < best_val_mae:
                best_val_mae = res["val_m1"]
                best_params = res["params"]
                
    # Batch 3: Activation & Regularization (using best arch)
    for act in ["leaky_relu", "swish"]:
        for drop in [0.0, 0.2]:
            for l2r in [0.0, 1e-3]:
                params = best_params.copy()
                params["activation"] = act
                params["dropout"] = drop
                params["l2"] = l2r
                res = run_experiment("regression", "batch_3_reg", params)
                if res["val_m1"] < best_val_mae:
                    best_val_mae = res["val_m1"]
                    best_params = res["params"]
                    
    # Batch 4: Optimizer & LR (using best previous params)
    for opt in ["adamw", "rmsprop"]:
        for lr in [1e-2, 3e-3, 3e-4]:
            params = best_params.copy()
            params["optimizer"] = opt
            params["lr"] = lr
            res = run_experiment("regression", "batch_4_opt", params)
            if res["val_m1"] < best_val_mae:
                best_val_mae = res["val_m1"]
                best_params = res["params"]
                
    # Batch 5: Features (using best previous params)
    for feat in ["interaction", "absolute", "both"]:
        params = best_params.copy()
        params["features"] = feat
        res = run_experiment("regression", "batch_5_feat", params)
        if res["val_m1"] < best_val_mae:
            best_val_mae = res["val_m1"]
            best_params = res["params"]

if __name__ == "__main__":
    print("Starting Classification Search...")
    run_classification_search()
    print("Starting Regression Search...")
    run_regression_search()
    print("Done!")

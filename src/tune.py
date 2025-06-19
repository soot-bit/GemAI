import optuna
import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
from utils import *

config = load_config()
features = config["features"]
training_params = config["training"]
    

def prepare_data():
    train_df = pd.read_parquet(get_path("train_data", config))
    
    # Split train into train and validation
    train_df, val_df = train_test_split(
        train_df, test_size=training_params["test_size"], 
        random_state=training_params["seed"]
    )
    
    # Prepare features
    X_train = train_df[features["categorical"] + features["numerical"]].values
    y_train = train_df[features["target"]].values.reshape(-1, 1)
    X_val = val_df[features["categorical"] + features["numerical"]].values
    y_val = val_df[features["target"]].values.reshape(-1, 1)
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols_idx = [len(features["categorical"]) + i for i in range(len(features["numerical"]))]
    X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
    X_val[:, num_cols_idx] = scaler.transform(X_val[:, num_cols_idx])
    
    return X_train, X_val, y_train, y_val

def objective(trial):
   # Suggest hyperparameters
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'cat_emb_dim': trial.suggest_int('cat_emb_dim', 1, 16),
        'optimizer_params': {
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        },
        'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
    }
    
    # Add fixed parameters
    params['cat_idxs'] = [0, 1, 2]
    params['cat_dims'] = [5, 7, 8]
    
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data()
    
    # Initialize TabNet
    model = TabNetRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['mae'],
        max_epochs=2,
        patience=7,
        batch_size=training_params["batch_size"],
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Get best validation MAE
    best_epoch = np.argmin(model.history['val_0_mae'])
    return min(model.history['val_0_mae'])

def main():
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=training_params["seed"]),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(objective, n_trials=100, timeout=3600*4)  # 4 hours max
    
    # Save results
    print("Best trial:")
    trial = study.best_trial
    print(f"Value (MAE): {trial.value:.2f}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params
    save_pickle(trial.params, get_path("processed_dir", config) / 'best_params.pkl')
    
    print("✅ Hyperparameter tuning complete!")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    main()

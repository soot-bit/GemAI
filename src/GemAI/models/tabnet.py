import pickle
import numpy as np
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import toml

from ..config import settings, get_project_root
from ..data import load_split_data, prepare_tabnet_data
from .. import utils

def run_training(params: dict):
    """
    Trains a TabNet model with given parameters from the config file.
    """
    utils.logger.info("Loading and preparing data for TabNet training...")
    train_df, val_df = load_split_data()
    X_train, y_train, X_val, y_val, cat_idxs, cat_dims, cat_mappings = prepare_tabnet_data(train_df, val_df)
    utils.logger.info("Data prepared.")

    model = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        **params
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse'],
        eval_name=['validation'],
        max_epochs=settings.tabnet.max_epochs,
        patience=settings.tabnet.patience,
        batch_size=settings.tabnet.batch_size,
        virtual_batch_size=settings.tabnet.virtual_batch_size,
    )
    utils.logger.info("TabNet model training finished.")
    
    model_dir = get_project_root() / settings.paths.tabnet_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_model(str(model_dir / 'tabnet_model')) 
    utils.logger.info(f"Model saved to {model_dir / 'tabnet_model.zip'}")

    # Save mappings
    mappings_path = model_dir / 'cat_mappings.pkl'
    with open(mappings_path, 'wb') as f:
        pickle.dump(cat_mappings, f)
    utils.logger.info(f"Categorical mappings saved to {mappings_path}")

    return model

def objective(trial, train_df, val_df):
    """Optuna objective function for TabNet."""
    X_train, y_train, X_val, y_val, cat_idxs, cat_dims, _ = prepare_tabnet_data(train_df, val_df)

    params = {
        "n_d": trial.suggest_int("n_d", 8, 32, step=4),
        "n_a": trial.suggest_int("n_a", 8, 32, step=4),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": trial.suggest_float("lr", 2e-2, 1e-1, log=True)},
    }

    model = TabNetRegressor(**params, cat_idxs=cat_idxs, cat_dims=cat_dims, verbose=0)

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)], eval_metric=['rmse'],
        max_epochs=settings.tabnet.max_epochs, patience=settings.tabnet.patience,
        batch_size=settings.tabnet.batch_size, virtual_batch_size=settings.tabnet.virtual_batch_size,
    )
    return model.best_cost

def run_tuning():
    """
    Runs Optuna hyperparameter tuning for the TabNet model.
    """
    utils.logger.info("Starting Optuna study for TabNet...")
    utils.logger.info("Loading data for tuning...")
    train_df, val_df = load_split_data()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=settings.training.random_state),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, train_df, val_df), 
        n_trials=settings.optuna.n_trials, 
        timeout=settings.optuna.timeout
    )
    
    utils.logger.info(f"Best trial RMSE: {study.best_trial.value}")
    utils.logger.info(f"Best hyperparameters: {study.best_params}")

    config_path = get_project_root() / "configs" / "config.toml"
    config = toml.load(config_path)
    
    # Update config with best params
    best_params = study.best_params
    config["tabnet"]["initial_params"]["n_d"] = best_params["n_d"]
    config["tabnet"]["initial_params"]["n_a"] = best_params["n_a"]
    config["tabnet"]["initial_params"]["n_steps"] = best_params["n_steps"]
    config["tabnet"]["initial_params"]["gamma"] = best_params["gamma"]
    config["tabnet"]["initial_params"]["lambda_sparse"] = best_params["lambda_sparse"]
    config["tabnet"]["initial_params"]["optimizer_params"]["lr"] = best_params["lr"]

    with open(config_path, 'w') as f:
        toml.dump(config, f)
        
    utils.logger.info("Best hyperparameters saved to configs/config.toml")
    
    return study.best_params

import pickle
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import toml

from ..config import config, get_project_root
from ..data import load_data, prep_tn_data
from .. import utils


def train_tn(params: dict):
    """
    Trains a TabNet model with given parameters from the config file.
    """
    utils.log_header("TABNET TRAINING")
    utils.log("Loading and preparing data for TabNet training...")
    trn_df, val_df = load_data()
    X_trn, y_trn, X_val, y_val, cat_ixs, cat_ds, cat_maps = (
        prep_tn_data(trn_df, val_df)
    )
    utils.log("Data prepared.")

    opt_params = params.pop("opt_params", {})
    model = TabNetRegressor(cat_idxs=cat_ixs, cat_dims=cat_ds, optimizer_params=opt_params, **params)

    model.fit(
        X_train=X_trn,
        y_train=y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric=["rmse"],
        eval_name=["validation"],
        max_epochs=config.tabnet.epochs,
        patience=config.tabnet.patience,
        batch_size=config.tabnet.batch_size,
        virtual_batch_size=config.tabnet.vbatch_size,
    )

    mdl_dir = get_project_root() / config.dir.logs / config.dir.tabnet
    mdl_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    model.save_model(str(mdl_dir / "tabnet_model"))
    utils.log_header("TABNET TRAINING COMPLETED")
    utils.log(f"Model saved to {mdl_dir / 'tabnet_model.zip'}")

    # Save mappings
    map_path = mdl_dir / "cat_mappings.pkl"
    with open(map_path, "wb") as f:
        pickle.dump(cat_maps, f)
    utils.log(f"Categorical mappings saved to {map_path}")

    return model


def obj(trial, trn_df, val_df):
    """Optuna objective function for TabNet."""
    X_trn, y_trn, X_val, y_val, cat_ixs, cat_ds, _ = prep_tn_data(
        trn_df, val_df
    )

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

    model = TabNetRegressor(**params, cat_idxs=cat_ixs, cat_dims=cat_ds, verbose=0)

    model.fit(
        X_train=X_trn,
        y_train=y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric=["rmse"],
        max_epochs=config.tabnet.epochs,
        patience=config.tabnet.patience,
        batch_size=config.tabnet.batch_size,
        virtual_batch_size=config.tabnet.vbatch_size,
    )
    return model.best_cost


def _update_cfg(bst_params: dict):
    """Updates the config.toml file with the best hyperparameters."""
    config_path = get_project_root() / "Configs" / "config.toml"
    cfg_dict = toml.load(config_path)

    # Update config with best params
    cfg_dict["tabnet"]["params"]["n_d"] = bst_params["n_d"]
    cfg_dict["tabnet"]["params"]["n_a"] = bst_params["n_a"]
    cfg_dict["tabnet"]["params"]["n_steps"] = bst_params["n_steps"]
    cfg_dict["tabnet"]["params"]["gamma"] = bst_params["gamma"]
    cfg_dict["tabnet"]["params"]["lambda_sparse"] = bst_params[
        "lambda_sparse"
    ]
    cfg_dict["tabnet"]["params"]["opt_params"]["lr"] = bst_params["lr"]

    with open(config_path, "w") as f:
        toml.dump(cfg_dict, f)

    utils.log(
        f"Best hyperparameters saved to {config_path.relative_to(get_project_root())}"
    )


def tune_tn():
    """
    Runs Optuna hyperparameter tuning for the TabNet model.
    """
    utils.log_header("TABNET HYPERPARAMETER TUNING")
    utils.log("Starting Optuna study for TabNet...")
    utils.log("Loading data for tuning...")
    trn_df, val_df = load_data()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.training.seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: obj(trial, trn_df, val_df),
        n_trials=config.optuna.trials,
        timeout=config.optuna.timeout,
    )

    utils.log(f"Best trial RMSE: {study.best_trial.value}")
    utils.log(f"Best hyperparameters: {study.best_params}")

    _update_cfg(study.best_params)

    utils.log_header("TABNET HYPERPARAMETER TUNING COMPLETED")

    return study.best_params

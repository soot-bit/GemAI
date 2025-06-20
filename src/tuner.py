import numpy as np
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import utils
from src.loader import load_split, prepare_data
from src.config import settings, CONFIG_DATA
import pprint, toml



def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
        'cat_emb_dim': trial.suggest_int('cat_emb_dim', 1, 16),
        'optimizer_params': {'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True)},
        'mask_type': trial.suggest_categorical('mask_type',['sparsemax','entmax']),
        'optimizer_fn': torch.optim.Adam,
    }

    # load & split
    train_df, val_df = load_split()
    X_train, y_train, cat_idxs, cat_dims = prepare_data(train_df)
    X_val,   y_val,   _,        _        = prepare_data(val_df)

    params['cat_idxs'], params['cat_dims'] = cat_idxs, cat_dims

    model = TabNetRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["mae"],  
        **settings.model["fit"]
    )

    return float(np.min(model.history['val_0_mae']))

def run_tuning(n_trials=1, timeout=3600):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_trial
    print(f"[TUNING DONE] Best MAE: {best.value:.2f}")
    pprint.pprint(f"{best.params=}")
    # persist best params

    #update toml file
    best_lr  = best.params.pop("lr")
    CONFIG_DATA["model"]["init"].update(best.params)
    CONFIG_DATA["model"]["init"]['optimizer_params'].update({'lr': best_lr})
    
    
    
    with open(utils.get_path("config.toml") , "w") as f:
        toml.dump(CONFIG_DATA, f)

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from pathlib import Path

from src.loader import prepare_data, load_split
import utils

def run_training(config: dict):
    
    # load & split
    train_df, val_df = load_split()
    X_train, y_train, cat_idxs, cat_dims = prepare_data(train_df)
    X_val,   y_val,   _,        _        = prepare_data(val_df)

    
    model = TabNetRegressor(
        **config.model["init"],
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["mae"],  
        eval_name=["val"], 
        **config.model["fit"]
    )

    # save model
    dir = utils.get_path(config.model["checkpoint"])  
    dir.mkdir(parents=True, exist_ok=True)   
    model.save_model(dir)
    utils.eval(model, X_val, y_val, save_path=dir.parent)


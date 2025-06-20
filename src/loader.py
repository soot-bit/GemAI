import numpy as np
import pickle
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import settings
from utils import pickle_

def load_split():
    df = pd.read_csv(settings.paths["train_csv"])
    return train_test_split(
        df,
        test_size=.1,
    )

def prepare_data(df, save_artifacts: bool = False):
    cat_cols = settings.features["categorical"]
    num_cols = settings.features["numerical"]
    target   = settings.features["target"]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(df[cat_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    X = np.hstack([X_cat, X_num])
    y = df[target].values.reshape(-1,1)

    cat_idxs = list(range(len(cat_cols)))
    cat_dims = [int(df[col].nunique()) for col in cat_cols]

    if save_artifacts:
        model_dir = Path(settings.paths["models"])
        model_dir.mkdir(parents=True, exist_ok=True)
        pickle_(enc, model_dir / "encoder.pkl")
        pickle_(scaler, model_dir / "scaler.pkl")

    return X, y, cat_idxs, cat_dims

import pickle
import numpy as np
import pandas as pd
from .config import config, get_project_root
from . import utils
from sklearn.model_selection import train_test_split

# --- Preprocessing ---


def _load_raw():
    raw_path = get_project_root() / config.dir.data / config.data.raw
    df = pd.read_csv(raw_path)
    df = df.drop(columns="Unnamed: 0")
    utils.log(f"Loaded raw data with shape: {df.shape}")
    return df


def _engineer_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["price_bwp"] = (df["price"] * 13.5).round(2).astype("float32")
    df["volume"] = (df["x"] * df["y"] * df["z"]).astype("float32")
    for col in config.data.cat_feats:
        df[col] = df[col].astype("category")
    return df


def _filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df[df["volume"] > 0].copy()
    utils.log(f"Shape after removing zero-volume entries: {df_filtered.shape}")
    return df_filtered


def _rm_outliers(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = config.data.num_feats
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    mask = ((df[num_cols] >= lower_bound) & (df[num_cols] <= upper_bound)).all(axis=1)
    df_clean = df[mask]
    utils.log(f"Shape after removing outliers: {df_clean.shape}")
    return df_clean


def _split_save(df: pd.DataFrame):
    trn_df, test_df = train_test_split(
        df, test_size=0.2, random_state=config.training.seed
    )
    processed_path = (
        get_project_root() / config.dir.data / config.data.processed
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump({"train": trn_df, "test": test_df}, f)
    utils.log(f"Processed data saved to: {processed_path}")


def preprocess():
    """
    Loads the raw diamonds dataset, preprocesses it, and saves the cleaned,
    split dataset to the processed data directory. This is the entrypoint
    for the 'prep' CLI command.
    """
    utils.log_header("DATA PREPROCESSING")
    utils.log("Starting data preprocessing...")

    df = _load_raw()
    df = _engineer_feats(df)
    df = _filter_invalid(df)
    df = _rm_outliers(df)
    _split_save(df)

    utils.log_header("DATA PREPROCESSING COMPLETED")


# --- Data Loading and Preparation for Models ---


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the pre-split training and validation data from the processed data directory.
    """
    processed_path = (
        get_project_root() / config.dir.data / config.data.processed
    )
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run the 'prep' command first: \n"
            "uv run python -m src.GemAI.main prep"
        )
    with open(processed_path, "rb") as f:
        data = pickle.load(f)
    trn_df, val_df = data.values()
    return trn_df, val_df


def prep_tn_data(trn_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """
    Prepares loaded data specifically for the TabNet model.
    - Separates features and target.
    - Converts categorical columns to integer codes and creates mappings for inference.
    - Returns data as float32 numpy arrays.
    """
    target = config.data.target
    cat_feats = config.data.cat_feats

    y_trn = trn_df[target].values.reshape(-1, 1)
    X_trn = trn_df.drop(target, axis=1)
    y_val = val_df[target].values.reshape(-1, 1)
    X_val = val_df.drop(target, axis=1)

    # Ensure column order is consistent for TabNet

    cat_ixs = [X_trn.columns.get_loc(col) for col in cat_feats]
    cat_ds = []
    cat_maps = {}

    # Process categorical features: convert to codes and store mappings
    for col in cat_feats:
        # Ensure categories are consistent and get codes
        trn_cat = X_trn[col].cat.categories
        cat_dtype = pd.CategoricalDtype(categories=trn_cat, ordered=True)
        cat_maps[col] = {
            category: i for i, category in enumerate(trn_cat)
        }  # FIX

        # Convert to codes here
        X_trn[col] = X_trn[col].astype(cat_dtype).cat.codes
        X_val[col] = X_val[col].astype(cat_dtype).cat.codes

        cat_ds.append(len(trn_cat))  # Store original number of categories

    # All features in X_trn and X_val should now be numerical (either float32 or int codes)
    # Convert entire DataFrame to float32 numpy array.
    X_trn_proc = X_trn.values.astype(np.float32)
    X_val_proc = X_val.values.astype(np.float32)
    y_trn_proc = y_trn.astype(np.float32)
    y_val_proc = y_val.astype(np.float32)

    return (
        X_trn_proc,
        y_trn_proc,
        X_val_proc,
        y_val_proc,
        cat_ixs,
        cat_ds,
        cat_maps,
    )

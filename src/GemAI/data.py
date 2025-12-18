import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from .config import settings, get_project_root
from . import utils
from sklearn.model_selection import train_test_split

# --- Preprocessing ---

def run_preprocessing():
    """
    Loads the raw diamonds dataset, preprocesses it, and saves the cleaned,
    split dataset to the processed data directory. This is the entrypoint
    for the 'process-data' CLI command.
    """
    utils.logger.info("Starting data preprocessing...")
    
    # 1. Load raw data
    raw_data_path = get_project_root() / settings.paths.raw_data
    df = pd.read_csv(raw_data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    utils.logger.info(f"Loaded raw data with shape: {df.shape}")

    # 2. Basic feature engineering and type conversion
    df['price_bwp'] = (df['price'] * 13.5).round(2).astype('float32')
    df['volume'] = (df['x'] * df['y'] * df['z']).astype('float32')
    
    for col in settings.data.categorical_features:
        df[col] = df[col].astype('category')
    
    final_features = settings.data.categorical_features + settings.data.numerical_features + [settings.data.target]
    df = df[final_features]

    # 3. Filter invalid data
    df_filtered = df[df['volume'] > 0].copy()
    utils.logger.info(f"Shape after removing zero-volume entries: {df_filtered.shape}")

    # 4. Remove outliers
    numeric_cols = settings.data.numerical_features + [settings.data.target]
    Q1 = df_filtered[numeric_cols].quantile(0.25)
    Q3 = df_filtered[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    mask = ((df_filtered[numeric_cols] >= lower_bound) & (df_filtered[numeric_cols] <= upper_bound)).all(axis=1)
    df_clean = df_filtered[mask]
    utils.logger.info(f"Shape after removing outliers: {df_clean.shape}")

    # 5. Split and save data
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=settings.training.random_state)
    processed_data_path = get_project_root() / settings.paths.processed_data
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_data_path, "wb") as f:
        pickle.dump({"train": train_df, "test": test_df}, f)
    
    utils.logger.info(f"Processed data saved to: {processed_data_path}")


# --- Data Loading and Preparation for Models ---

def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the pre-split training and validation data from the processed data directory.
    """
    processed_data_path = get_project_root() / settings.paths.processed_data
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_data_path}. "
            "Please run the 'process-data' command first: \n"
            "uv run python -m src.GemAI.main process-data"
        )
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)
    train_df, val_df = data.values()
    return train_df, val_df

def prepare_tabnet_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """
    Prepares loaded data specifically for the TabNet model.
    - Separates features and target.
    - Converts categorical columns to integer codes and creates mappings for inference.
    - Returns data as float32 numpy arrays.
    """
    target = settings.data.target
    categorical_features = settings.data.categorical_features

    y_train = train_df[target].values.reshape(-1, 1)
    X_train = train_df.drop(target, axis=1)
    y_val = val_df[target].values.reshape(-1, 1)
    X_val = val_df.drop(target, axis=1)

    # Ensure column order is consistent for TabNet
    all_features_except_target = X_train.columns.tolist()

    cat_idxs = [X_train.columns.get_loc(col) for col in categorical_features]
    cat_dims = []
    cat_mappings = {}

    # Process categorical features: convert to codes and store mappings
    for col in categorical_features:
        # Ensure categories are consistent and get codes
        train_categories = X_train[col].cat.categories
        cat_dtype = pd.CategoricalDtype(categories=train_categories, ordered=True)
        cat_mappings[col] = cat_dtype
        
        # Convert to codes here
        X_train[col] = X_train[col].astype(cat_dtype).cat.codes
        X_val[col] = X_val[col].astype(cat_dtype).cat.codes
        
        cat_dims.append(len(train_categories)) # Store original number of categories

    # All features in X_train and X_val should now be numerical (either float32 or int codes)
    # Convert entire DataFrame to float32 numpy array.
    X_train_processed = X_train.values.astype(np.float32)
    X_val_processed = X_val.values.astype(np.float32)
    y_train_processed = y_train.astype(np.float32)
    y_val_processed = y_val.astype(np.float32)

    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, cat_idxs, cat_dims, cat_mappings

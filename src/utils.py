import os
import torch
import toml
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------
# cofigs 
# ----------------------
CONFIG_DIR = Path(__file__).parent.parent
def load_config():
    return toml.load(CONFIG_DIR / "config.toml")

def get_path(key, config):
    """Get full path for a file resource"""
    base_dir = CONFIG_DIR 
    paths = config["paths"]
    
    if key == "raw_data":
        return base_dir / paths["data_dir"] / paths["raw_data"]
    if key in ["train_data", "test_data", "feature_metadata", "scaler", "model"]:
        processed_dir = base_dir / paths["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir / paths[key]
    return base_dir / paths.get(key, key)

# ----------------------
# Data Preprocessing
# ----------------------
def preprocess(df, config):
    """Preprocess diamond dataframe"""
    # Load parameters
    exchange_rate = config["training"]["exchange_rate"]
    features = config["features"]
    
    # Clean and engineer features
    df = df.drop(columns='Unnamed: 0', errors='ignore')
    df['price_bwp'] = (df['price'] * exchange_rate).round(2)
    df['volume'] = df['x'] * df['y'] * df['z']
    
    # ordinal encoding
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    
    df['cut'] = pd.Categorical(df['cut'], categories=cut_order, ordered=True).codes
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarity_order, ordered=True).codes
    df['color'] = pd.Categorical(df['color'], categories=color_order, ordered=True).codes
    
    # Create metadata
    metadata = {
        'cut_mapping': dict(enumerate(cut_order)),
        'clarity_mapping': dict(enumerate(clarity_order)),
        'color_mapping': dict(enumerate(color_order)),
        'num_stats': {
            'mean': df[features["numerical"]].mean().to_dict(),
            'std': df[features["numerical"]].std().to_dict()
        }
    }
    
    return df, metadata

def get_data(config):
    """Prepare and split training data"""
    # Load and preprocess
    df = pd.read_csv(get_path("raw_data", config))
    df, metadata = preprocess(df, config)
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=config["training"]["test_size"], 
    )
    
    # Save metadata
    save_pickle(metadata, get_path("feature_metadata", config))
    return train_df, test_df, metadata

def scale_features(X_train, X_test, config):
    """Scale numerical features"""
    features = config["features"]
    scaler = StandardScaler()
    num_cols_idx = [len(features["categorical"]) + i for i in range(len(features["numerical"]))]
    
    X_train[:, num_cols_idx] = scaler.fit_transform(X_train[:, num_cols_idx])
    if X_test is not None:
        X_test[:, num_cols_idx] = scaler.transform(X_test[:, num_cols_idx])
    
    pickle_(scaler, get_path("scaler", config))
    return X_train, X_test, scaler

def prepare_input(input_data, config, metadata, scaler):
    """Prepare single input for prediction"""
    df = pd.DataFrame([input_data])
    df['volume'] = df['x'] * df['y'] * df['z']
    
    #  categoricals
    df['cut'] = df['cut'].map({v: k for k, v in metadata['cut_mapping'].items()})
    df['color'] = df['color'].map({v: k for k, v in metadata['color_mapping'].items()})
    df['clarity'] = df['clarity'].map({v: k for k, v in metadata['clarity_mapping'].items()})
    
    # Prepare feature array
    features = config["features"]
    all_features = features["categorical"] + features["numerical"]
    X = df[all_features].values
    
    # Scale numerical features
    num_cols_idx = [len(features["categorical"]) + i for i in range(len(features["numerical"]))]
    X[:, num_cols_idx] = scaler.transform(X[:, num_cols_idx])
    
    return X, all_features

#----------------------
#  General Utilities
# ----------------------
def pickle_(obj, path):
    """Save object to pickle file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_(path):
    """Load object from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def view(history, save_path=None):
    """Plot training history"""
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    ax1.plot(history["loss"], label="Loss") 
    ax2.plot(history["val_mae"], label="Val MAE", color="r")

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("val MAE")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def eval(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

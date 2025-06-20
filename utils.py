import pickle
from pathlib import Path
from src.config import settings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def pickle_(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def unpickle_(path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def get_path(path:str):
    root = Path(__file__).parent
    return root / path



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
        plt.savefig(save_path/"model_eval.svg")
    plt.close()

def metrics(y_true, y_pred):
    """evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    return {
        'mae': mae,
        'rmse': rmse.item(),
        'mape': mape.item()
    }

def eval(model,  X, y, save_path=None):
    """Evaluate model"""
    y_pred = model.predict(X)
    if save_path:
        view(model.history, save_path)
    print(metrics(y, y_pred))
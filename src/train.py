import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from utils import (
    load_config, get_path,  view,
    scale_features, pickle_, eval, load_
)

def main():
    config = load_config()
    
    # Load data
    train_df = pd.read_parquet(get_path("train_data", config))
    test_df = pd.read_parquet(get_path("test_data", config))
    
    # Prepare features
    features = config["features"]
    X_train = train_df[features["categorical"] + features["numerical"]].values
    y_train = train_df[features["target"]].values.reshape(-1, 1)
    X_test = test_df[features["categorical"] + features["numerical"]].values
    y_test = test_df[features["target"]].values.reshape(-1, 1)
    
    # Scale features
    X_train, X_test, _ = scale_features(X_train, X_test, config)
    
    # Initialize TabNet
    model = TabNetRegressor(**config["hparams"])
    
    # Train model
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['val'],
        eval_metric=['mae'],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        drop_last=False,
        loss_fn=torch.nn.functional.mse_loss
    )
    
    # Save model
    model.save_model(get_path("model", config))
    
    # Evaluate
    predictions = model.predict(X_test)
    metrics = eval(y_test, predictions)
    
    print(f"\nFinal Evaluation:")
    print(f"MAE: {metrics['mae']:.2f} BWP")
    print(f"RMSE: {metrics['rmse']:.2f} BWP")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    # Save training history
    pickle_(model.history, get_path("training_history", config))
    view(model.history, get_path("training_plot", config))
    
    # Feature importance
    feature_importance = model.feature_importances_
    all_features = features["categorical"] + features["numerical"]
    fi_df = pd.DataFrame({
        'feature': all_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(fi_df)
    fi_df.to_csv(get_path("feature_importance", config), index=False)
    

if __name__ == "__main__":
    main()

import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from .utils import (
    load_config, get_path, load_pickle, 
    prepare_input, get_tabnet_config
)

def load_model():
    config = load_config()
    model = TabNetRegressor()
    model.load_model(get_path("model", config))
    return model

def predict(input_data):
    # Load configuration and artifacts
    config = load_config()
    model = load_model()
    scaler = load_pickle(get_path("scaler", config))
    metadata = load_pickle(get_path("feature_metadata", config))
    
    # Prepare input
    X, feature_names = prepare_input(input_data, config, metadata, scaler)
    
    # Predict
    prediction = model.predict(X)[0][0]
    
    # Explainability
    explain_matrix, masks = model.explain(X)
    
    return {
        'price_bwp': prediction,
        'explainability': {
            'features': feature_names,
            'importance': explain_matrix[0].tolist(),
            'masks': masks
        }
    }

if __name__ == "__main__":
    # Example input
    sample_input = {
        'carat': 0.75,
        'cut': 'Ideal',
        'color': 'D',
        'clarity': 'SI1',
        'depth': 62.2,
        'table': 55.0,
        'x': 5.83,
        'y': 5.87,
        'z': 3.64
    }
    
    result = predict(sample_input)
    print(f"Predicted Price: {result['price_bwp']:.2f} BWP")
    print("Feature Importance:")
    for feature, importance in zip(result['explainability']['features'], 
                                  result['explainability']['importance']):
        print(f"  {feature}: {importance:.4f}")

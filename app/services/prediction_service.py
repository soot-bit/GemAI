import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from app.schemas import DiamondInput
from app.core.model_manager import ModelManager
from app.error_handler import InputValidationError, PredictionError, ModelNotFoundError

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.BWP_TO_USD = 0.075  # Approximate conversion rate

    def _transform_categorical_inputs(self, input_df: pd.DataFrame, cat_mappings: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """
        Transforms categorical string inputs in the DataFrame to numerical codes
        using the provided categorical mappings.
        """
        for col, mapping in cat_mappings.items():
            if col in input_df.columns:
                # Use .map() to apply the dictionary mapping
                # If a category is not found in the mapping, it will become NaN
                input_df[col] = input_df[col].map(mapping)
                
                # Fill NaN values (unseen categories) with a default code, e.g., -1
                # Or raise an error if unseen categories should not be allowed
                # For now, we'll keep -1 as it was the previous behavior
                input_df[col] = input_df[col].fillna(-1)
                
        return input_df


    def predict_price(self, data: DiamondInput) -> Dict[str, float]:
        if not self.model_manager.is_model_loaded():
            raise ModelNotFoundError(
                "Model is not loaded.",
                "Please check server logs for details and train a model if necessary.",
            )

        try:
            tabnet = self.model_manager.get_model()
            cat_mappings = self.model_manager.get_cat_mappings()

            # --- Data validation and preparation ----------------------------
            input_df = pd.DataFrame([data.model_dump()])

            # --- Feature Engineering (using the new method) -----------------
            input_df = self._transform_categorical_inputs(input_df, cat_mappings)

            # Convert to numpy array for prediction
            X = input_df.values.astype(np.float32)

            # --- Prediction -------------------------------------------------
            prediction = tabnet.predict(X)
            price_bwp = float(prediction[0][0])
            price_usd = price_bwp * self.BWP_TO_USD

            return {"price_bwp": round(price_bwp, 2), "price_usd": round(price_usd, 2)}

        except ValidationError as e:
            raise InputValidationError(errors=e.errors())
        except ModelNotFoundError as e:
            raise e  # Re-raise if model is unexpectedly unloaded during prediction
        except Exception as e:
            logger.error(f"An unexpected error occurred during prediction: {e}")
            raise PredictionError(
                "An unexpected error occurred during prediction.",
                "Please check the server logs for more details.",
            )

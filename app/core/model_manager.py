import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from GemAI.config import config, get_project_root
from GemAI.models.tabnet import TabNetRegressor
from app.error_handler import ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self._tabnet_model: Optional[TabNetRegressor] = None
        self._cat_mappings: Optional[Dict[str, Any]] = None
        self._model_loaded: bool = False
        self._load_model_and_mappings()

    def _load_model_and_mappings(self):
        try:
            MODEL_DIR = get_project_root() / config.paths.log_dir / config.tabnet.dir
            model_path = MODEL_DIR / "tabnet_model.zip"
            mappings_path = MODEL_DIR / "cat_mappings.pkl"

            if not model_path.exists() or not mappings_path.exists():
                raise ModelNotFoundError(
                    "Model or mappings not found.",
                    "Train a model first using 'uv run gemai train tabnet'",
                )

            logger.info("Loading model and mappings...")
            tabnet = TabNetRegressor()
            tabnet.load_model(str(model_path))

            with open(mappings_path, "rb") as f:
                cat_mappings = pickle.load(f)

            self._tabnet_model = tabnet
            self._cat_mappings = cat_mappings
            self._model_loaded = True
            logger.info("Model and mappings loaded successfully.")

        except ModelNotFoundError as e:
            logger.error(f"{e.message} Suggestion: {e.suggestion}")
            self._tabnet_model = None
            self._cat_mappings = None
            self._model_loaded = False
        except Exception as e:
            logger.error(f"An unexpected error occurred during model loading: {e}")
            self._tabnet_model = None
            self._cat_mappings = None
            self._model_loaded = False

    def get_model(self) -> TabNetRegressor:
        if not self._model_loaded or self._tabnet_model is None:
            raise ModelNotFoundError(
                "Model is not loaded.",
                "Please check server logs for details and train a model if necessary.",
            )
        return self._tabnet_model

    def get_cat_mappings(self) -> Dict[str, Any]:
        if not self._model_loaded or self._cat_mappings is None:
            raise ModelNotFoundError(
                "Categorical mappings are not loaded.",
                "Please check server logs for details and train a model if necessary.",
            )
        return self._cat_mappings

    def is_model_loaded(self) -> bool:
        return self._model_loaded


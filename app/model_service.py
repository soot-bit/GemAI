from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

from GemAI.config import config, get_project_root
from GemAI.data import load_data

from app.schemas import PredictRequest


BWP_TO_USD = 0.075


class ModelService:
    def __init__(self) -> None:
        self.model: TabNetRegressor | None = None
        self.cat_mappings: dict[str, dict[str, int]] = {}
        self.feature_order: list[str] = []

    def load(self) -> None:
        model_dir = get_project_root() / config.dir.logs / config.dir.tabnet
        model_path = model_dir / "tabnet_model.zip"
        mapping_path = model_dir / "cat_mappings.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Categorical mappings not found: {mapping_path}")

        trn_df, _ = load_data()
        self.feature_order = [c for c in trn_df.columns if c != config.data.target]

        with mapping_path.open("rb") as f:
            self.cat_mappings = pickle.load(f)

        self.model = TabNetRegressor()
        self.model.load_model(str(model_path))

    def unload(self) -> None:
        self.model = None

    def health(self) -> dict[str, Any]:
        return {
            "model_loaded": self.model is not None,
            "feature_count": len(self.feature_order),
            "feature_order": self.feature_order,
            "categorical_features": list(self.cat_mappings.keys()),
        }

    def metadata(self) -> dict[str, Any]:
        cat_opts = {
            feature: sorted(list(mapping.keys()))
            for feature, mapping in self.cat_mappings.items()
        }
        return {
            "feature_order": self.feature_order,
            "categorical_options": cat_opts,
            "optional_features": [],
        }

    def _encode_categoricals(self, in_df: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in self.cat_mappings.items():
            if col in in_df.columns:
                in_df[col] = in_df[col].map(mapping).fillna(-1)
        return in_df

    def _build_feature_row(self, payload: PredictRequest) -> pd.DataFrame:
        row = payload.model_dump()
        row["volume"] = float(row["x"] * row["y"] * row["z"])

        missing = [feature for feature in self.feature_order if feature not in row]
        if missing:
            raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

        in_df = pd.DataFrame([row])[self.feature_order]
        in_df = self._encode_categoricals(in_df)

        for col in self.feature_order:
            if col not in self.cat_mappings:
                in_df[col] = pd.to_numeric(in_df[col], errors="coerce")

        if in_df.isna().any().any():
            raise ValueError("Input contains invalid values after preprocessing.")

        return in_df.astype(np.float32)

    def predict(self, payload: PredictRequest) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        in_df = self._build_feature_row(payload)
        prediction = self.model.predict(in_df.values)
        bwp = float(prediction[0][0])
        usd = bwp * BWP_TO_USD

        return {"bwp": round(bwp, 2), "usd": round(usd, 2)}

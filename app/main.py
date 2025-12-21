import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from GemAI.config import config, get_project_root
from GemAI.models.tabnet import TabNetRegressor
from app.schemas import Diamond, Prediction

# --- Configuration and Setup ----------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- FastAPI setup --------------------------------------------------
app = FastAPI(title="GemAI Diamond Price Predictor")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# --- Model Loading --------------------------------------------------
model: Optional[TabNetRegressor] = None
mappings: Optional[Dict[str, Any]] = None

@app.on_event("startup")
def load_model():
    global model, mappings
    try:
        mdl_dir = get_project_root() / config.dir.logs / config.dir.tabnet
        mdl_path = mdl_dir / "tabnet_model.zip"
        map_path = mdl_dir / "cat_mappings.pkl"

        if not mdl_path.exists() or not map_path.exists():
            raise FileNotFoundError("Model or mappings not found. Train a model first.")

        logger.info("Loading model and mappings...")
        model = TabNetRegressor()
        model.load_model(str(mdl_path))

        with open(map_path, "rb") as f:
            mappings = pickle.load(f)
        logger.info("Model and mappings loaded successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}")
        model = None
        mappings = None

# --- Prediction Logic -----------------------------------------------
BWP_TO_USD = 0.075

def transform_cats(in_df: pd.DataFrame, cat_maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    for col, mapping in cat_maps.items():
        if col in in_df.columns:
            in_df[col] = in_df[col].map(mapping)
            in_df[col] = in_df[col].fillna(-1)
    return in_df

# --- API Endpoints --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/info", response_class=HTMLResponse)
def info(request: Request):
    return templates.TemplateResponse("info.html", {"request": request})

@app.post("/predict", response_model=Prediction)
def predict(data: Diamond):
    if not model or not mappings:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    try:
        in_df = pd.DataFrame([data.model_dump()])
        in_df = transform_cats(in_df, mappings)
        X = in_df.values.astype(np.float32)
        
        prediction_raw = model.predict(X)
        bwp = float(prediction_raw[0][0])
        usd = bwp * BWP_TO_USD

        return Prediction(bwp=round(bwp, 2), usd=round(usd, 2))

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")
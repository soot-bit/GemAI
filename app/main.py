import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pytorch_tabnet.tab_model import TabNetRegressor

# Add src to path to allow importing GemAI modules
# This is a common pattern for standalone apps within a larger project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.GemAI.config import settings, get_project_root

# --- Load artifacts -------------------------------------------------
# Use paths from the centralized config
MODEL_DIR = get_project_root() / settings.paths.tabnet_dir
model_path = MODEL_DIR / "tabnet_model.zip"
mappings_path = MODEL_DIR / "cat_mappings.pkl"

if not model_path.exists() or not mappings_path.exists():
    print("Error: Model or mappings not found. Train a model first using 'uv run python -m src.GemAI.main train tabnet'")
    sys.exit(1)

tabnet = TabNetRegressor()
tabnet.load_model(str(model_path))

with open(mappings_path, 'rb') as f:
    cat_mappings = pickle.load(f)

# --- FastAPI setup --------------------------------------------------
app = FastAPI(title="GemAI Diamond Price Predictor")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# --- Pydantic Schemas -----------------------------------------------
class DiamondInput(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

class PredictResponse(BaseModel):
    price_bwp: float

# --- Routes ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictResponse)
def predict(data: DiamondInput):
    # Create a single-row DataFrame from the input
    # The column order is implicitly handled by Pydantic and DataFrame creation
    input_df = pd.DataFrame([data.model_dump()])

    # Apply the same categorical dtypes from training
    for col, dtype in cat_mappings.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(dtype)

    # Convert dataframe to a float32 numpy array for the model
    X = input_df.values.astype(np.float32)

    # Get prediction
    prediction = tabnet.predict(X)
    price_bwp = float(prediction[0][0])

    return {"price_bwp": round(price_bwp, 2)}

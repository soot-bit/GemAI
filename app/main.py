from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

from src.config import settings
import utils

# ─── Load config ─────────────────────────────────────────
FEATURES   = settings.features
MODEL_CONF = settings.model
BASE_DIR   = Path(__file__).resolve().parent

# ─── Load artifacts ──────────────────────────────────────
model_ckpt = utils.get_path(MODEL_CONF["checkpoint"]).with_suffix(".zip")
encoder    = utils.unpickle_(settings.paths["models"] + "/encoder.pkl")
scaler     = utils.unpickle_(settings.paths["models"] + "/scaler.pkl")

tabnet = TabNetRegressor()
tabnet.load_model(model_ckpt)

# ─── FastAPI setup ───────────────────────────────────────
app = FastAPI(title="Diamond Price Predictor")

# Static & template paths
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ─── Pydantic Schemas ────────────────────────────────────

class DiamondInput(BaseModel):
    carat: float
    cut: int
    color: int
    clarity: int
    depth: float
    table: float
    x: float
    y: float
    z: float





class PredictResponse(BaseModel):
    price: float

# ─── Routes ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Mappings from integer code to original strings the encoder expects
cut_map_inv = {0: "Fair", 1: "Good", 2: "Very Good", 3: "Premium", 4: "Ideal"}
color_map_inv = {0: "J", 1: "I", 2: "H", 3: "G", 4: "F", 5: "E", 6: "D"}
clarity_map_inv = {0: "I1", 1: "SI2", 2: "SI1", 3: "VS2", 4: "VS1", 5: "VVS2", 6: "VVS1", 7: "IF"}
USD_TO_BWP = 13.5  # update with current rate
@app.post("/predict")
def predict(data: DiamondInput):
    # Convert integers back to strings before encoding
    cut_str = cut_map_inv.get(data.cut)
    color_str = color_map_inv.get(data.color)
    clarity_str = clarity_map_inv.get(data.clarity)

    categorical = [[cut_str, color_str, clarity_str]]
    numerical = [[data.carat, data.depth, data.table, data.x, data.y, data.z]]

    X_cat_enc = encoder.transform(categorical)  # now encoding strings
    X_num_scaled = scaler.transform(numerical)

    X = np.hstack([X_cat_enc, X_num_scaled])
    prediction = tabnet.predict(X)[0]   # use your loaded TabNet model

    prediction_usd = float(tabnet.predict(X)[0])
    prediction_bwp = prediction_usd * USD_TO_BWP

    return {
        "price_usd": round(prediction_usd, 2),
        "price_bwp": round(prediction_bwp, 2)
    }



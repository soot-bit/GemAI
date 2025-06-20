import toml
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pytorch_tabnet.tab_model import TabNetRegressor

# ─── Load config ─────────────────────────────────────────
_cfg = toml.load(Path(__file__).parent.parent / "config.toml")
FEATURES   = _cfg["features"]
MODEL_CONF = _cfg["model"]

# ─── Load artifacts ─────────────────────────────────────
_model_ckpt = Path(MODEL_CONF["tabnet_checkpoint"])
_encoder    = pickle.load(open(Path(MODEL_CONF["save_dir"]) / "encoder.pkl", "rb"))
_scaler     = pickle.load(open(Path(MODEL_CONF["save_dir"]) / "scaler.pkl",  "rb"))

# load TabNet
_tabnet = TabNetRegressor()
_tabnet.load_model(str(_model_ckpt))

# ─── FastAPI setup ──────────────────────────────────────
app = FastAPI(title="Diamond Price Predictor")
templates = Jinja2Templates(directory="app/templates")

# ─── Pydantic schemas ───────────────────────────────────
class PredictRequest(BaseModel):
    categorical: list[int]
    numerical:   list[float]

class PredictResponse(BaseModel):
    prediction: float

# ─── Helper ─────────────────────────────────────────────
def make_prediction(cat: list[int], num: list[float]) -> float:
    X_cat = np.array([cat])
    X_num = np.array([num])
    X_cat_enc = _encoder.transform(X_cat)
    X_num_sc  = _scaler.transform(X_num)
    X = np.hstack([X_cat_enc, X_num_sc])
    pred = _tabnet.predict(X)[0]
    return float(pred)

# ─── Routes ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # pass feature names into template
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "features": FEATURES}
    )

@app.post("/predict", response_model=PredictResponse)
async def predict_api(body: PredictRequest):
    price = make_prediction(body.categorical, body.numerical)
    return PredictResponse(prediction=price)

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model_service import ModelService
from app.schemas import HealthResponse, PredictRequest, PredictResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gemai.web")


service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        service.load()
        logger.info("Model service initialized")
    except Exception:
        logger.exception("Failed to initialize model service")
    yield
    service.unload()


app = FastAPI(title="GemAI Web", version="2.0.0", lifespan=lifespan)
base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=base_dir / "templates")
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/info", response_class=HTMLResponse)
def info(request: Request):
    return templates.TemplateResponse("info.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.get("/api/model-meta")
def model_meta():
    if not service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service.metadata()


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = service.predict(payload)
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict", response_model=PredictResponse)
def predict_legacy(payload: PredictRequest) -> PredictResponse:
    return predict(payload)

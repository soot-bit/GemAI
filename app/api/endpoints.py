from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.schemas import DiamondInput, PredictResponse
from app.services.prediction_service import PredictionService
from app.core.model_manager import ModelManager


# Initialize Jinja2Templates for HTML responses
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Create an API router
router = APIRouter()

# Initialize ModelManager and PredictionService as singletons
# This ensures the model is loaded only once when the application starts
model_manager_singleton = ModelManager()
prediction_service_singleton = PredictionService(model_manager_singleton)

# Dependency for ModelManager
def get_model_manager() -> ModelManager:
    return model_manager_singleton

# Dependency for PredictionService
def get_prediction_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionService:
    # We return the singleton instance, but keep the dependency to ensure model_manager is available
    return prediction_service_singleton


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/predict", response_model=PredictResponse)
def predict(
    data: DiamondInput,
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    return prediction_service.predict_price(data)

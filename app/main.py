import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.error_handler import (
    DiamondPricePredictorError,
    DiamondPricePredictorReportGenerator,
    ErrorReporter,
)
from app.api.endpoints import router


# --- Configuration and Setup ----------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Error Handling Setup -------------------------------------------
report_generator = DiamondPricePredictorReportGenerator()
error_reporter = ErrorReporter(report_generator)

# --- FastAPI setup --------------------------------------------------
app = FastAPI(title="GemAI Diamond Price Predictor")


@app.exception_handler(DiamondPricePredictorError)
async def diamond_predictor_exception_handler(
    request: Request, exc: DiamondPricePredictorError
):
    report = error_reporter.report(exc)
    return JSONResponse(status_code=400, content=report.to_dict())

app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)

app.include_router(router)

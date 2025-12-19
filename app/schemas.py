from pydantic import BaseModel


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
    price_usd: float

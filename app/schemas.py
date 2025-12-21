from pydantic import BaseModel


class Diamond(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float


class Prediction(BaseModel):
    bwp: float
    usd: float

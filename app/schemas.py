from pydantic import BaseModel, ConfigDict, model_validator


class PredictRequest(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    price: float
    x: float
    y: float
    z: float

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_ranges(self) -> "PredictRequest":
        numeric_positive = ["carat", "depth", "table", "price", "x", "y", "z"]
        for field in numeric_positive:
            if getattr(self, field) <= 0:
                raise ValueError(f"'{field}' must be greater than zero")
        return self


class PredictResponse(BaseModel):
    bwp: float
    usd: float


class HealthResponse(BaseModel):
    model_loaded: bool
    feature_count: int
    feature_order: list[str]
    categorical_features: list[str]

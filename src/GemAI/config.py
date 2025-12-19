import toml
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List

# --- Pydantic Models for Config Structure ---


class PathsConfig(BaseModel):
    dir: str
    raw_data: str
    processed_data: str
    log_dir: str


class Data(BaseModel):
    target: str
    categorical_features: List[str]
    numerical_features: List[str]


class Training(BaseModel):
    random_state: int


class TabnetParams(BaseModel):
    n_d: int
    n_a: int
    n_steps: int
    gamma: float
    lambda_sparse: float
    optimizer_params: Dict[str, Any]
    scheduler_params: Dict[str, Any]


class TabnetConfig(BaseModel):
    max_epochs: int
    patience: int
    batch_size: int
    virtual_batch_size: int
    initial_params: TabnetParams
    dir: str


class OptunaConfig(BaseModel):
    n_trials: int
    timeout: int


class AutogluonConfig(BaseModel):
    time_limit: int
    dir: str


class Config(BaseModel):
    model_config = ConfigDict(extra="allow")
    paths: PathsConfig
    data: Data
    training: Training
    tabnet: TabnetConfig
    autogluon: AutogluonConfig
    optuna: OptunaConfig


# --- Config Loading ---
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_config() -> Config:
    """Loads the config.toml file and returns a Settings object."""
    config_path = get_project_root() / "Configs" / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config_data = toml.load(config_path)
    return Config(**config_data)


# Load the config once and make it available
config = load_config()

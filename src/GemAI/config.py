import toml
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List

# --- Pydantic Models for Config Structure ---


class DirCfg(BaseModel):
    data: str
    logs: str
    autogluon: str
    tabnet: str


class Data(BaseModel):
    target: str
    cat_feats: List[str]
    num_feats: List[str]
    raw: str
    processed: str


class TrainCfg(BaseModel):
    seed: int


class TabNetParams(BaseModel):
    n_d: int
    n_a: int
    n_steps: int
    gamma: float
    lambda_sparse: float
    opt_params: Dict[str, Any]
    sched_params: Dict[str, Any]


class TabNetCfg(BaseModel):
    epochs: int
    patience: int
    batch_size: int
    vbatch_size: int
    params: TabNetParams


class OptunaCfg(BaseModel):
    trials: int
    timeout: int


class AutoGluonCfg(BaseModel):
    timeout: int | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    dir: DirCfg
    data: Data
    training: TrainCfg
    tabnet: TabNetCfg
    autogluon: AutoGluonCfg
    optuna: OptunaCfg


# --- Config Loading ---
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_config() -> AppConfig:
    """Loads the config.toml file and returns a Settings object."""
    config_path = get_project_root() / "Configs" / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config_data = toml.load(config_path)
    return AppConfig(**config_data)


# Load the config once and make it available
config = load_config()

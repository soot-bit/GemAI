import toml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# --- Pydantic Models for Config Structure ---

class Paths(BaseModel):
    data_dir: str = 'data'
    raw_data: str = 'data/raw/diamonds.csv'
    processed_data: str = 'data/processed/clean_ds.plk'
    model_dir: str = 'models'
    autogluon_dir: str = 'models/autogluon'
    tabnet_dir: str = 'models/tabnet'

class Data(BaseModel):
    target: str
    problem_type: str
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

class TabnetSettings(BaseModel):
    max_epochs: int
    patience: int
    batch_size: int
    virtual_batch_size: int
    initial_params: TabnetParams

class OptunaSettings(BaseModel):
    n_trials: int
    timeout: int

class AutogluonSettings(BaseModel):
    time_limit: int
    preset: str
    eval_metric: str

class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    data: Data
    training: Training
    tabnet: TabnetSettings
    autogluon: AutogluonSettings
    optuna: OptunaSettings

# --- Config Loading ---

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent.parent

def load_config() -> Settings:
    """Loads the config.toml file and returns a Settings object."""
    config_path = get_project_root() / "configs" / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config_data = toml.load(config_path)
    return Settings(**config_data)

# Load the config once and make it available for import across the project
settings = load_config()  

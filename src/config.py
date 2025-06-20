
import toml
from pathlib import Path
from pydantic_settings import BaseSettings

class TrainingSettings(BaseSettings):
    paths: dict
    features: dict
    model: dict

# ——— TOML by hand ———
toml_path = Path(__file__).parent.parent / "config.toml"
CONFIG_DATA = toml.load(toml_path)

# Instantiate settings with the raw TOML dict
settings = TrainingSettings(**CONFIG_DATA)

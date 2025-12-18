from pathlib import Path
import pickle
import logging
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent


def get_path(file="Data"):
    return BASE_DIR / file


def unpickle(file):
    path = get_path(file)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _setup_logging():
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"autogluon_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="\n%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Now with actual file path
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger, log_file


logger, log_file = _setup_logging()

from pathlib import Path
import pickle
import logging
from datetime import datetime
from rich.console import Console, Theme
from rich.logging import RichHandler
from .config import config, get_project_root  # Import config

BASE_DIR = get_project_root()


def get_path(file="Data"):
    return BASE_DIR / file


def unpickle(file):
    path = get_path(file)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# Initialize Rich Console with a custom theme
custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "error": "bold red"})
console = Console(theme=custom_theme)


def _setup_logging():
    log_dir = BASE_DIR / config.paths.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # This log file is for general utility logging, not specific model logs
    log_file = log_dir / f"general_{timestamp}.log"

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # RichHandler will handle the formatting
        handlers=[
            RichHandler(
                console=console,
                level=logging.INFO,
                show_time=True,
                show_level=True,
                show_path=True,
                enable_link_path=True,
            ),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("rich")  # Use a named logger for Rich
    return logger, log_file


logger, log_file = _setup_logging()


def log_rule(title: str):
    """Prints a console rule with the given title."""
    console.rule(f"[bold blue]{title}[/bold blue]")


def log_info(message: str):
    """Prints an info message to the console using the Rich logger."""
    logger.info(message)

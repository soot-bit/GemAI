import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()

logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)]
)

log = logging.getLogger("app")

console.rule("[bold blue]INFO")
log.info("Model training started")

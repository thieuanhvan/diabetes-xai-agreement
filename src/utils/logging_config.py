import logging
from datetime import datetime

from src.utils.project_paths import get_logs_dir


def configure_logging():

    log_dir = get_logs_dir()
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    log_file = log_dir / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

    logging.info(f"Logging initialized. Log file: {log_file}")
import logging
import os

def setup_logging(log_file='experiment.log'):
    """Sets up logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_dir(path):
    """Ensures a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

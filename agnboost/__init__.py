from agnboost import config  # noqa: F401

import logging

# Configure the root logger for the AGNBoost package
def setup_logging(level=logging.WARNING, log_file=None):
    """
    Configure the AGNBoost logger.
    
    Parameters:
    -----------
    level : int, default=logging.INFO
        Logging level to use.
    log_file : str or None, default=None
        Path to log file. If None, only logs to console.
    """
    # Get the logger
    logger = logging.getLogger('AGNBoost')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Set up default logging
setup_logging()
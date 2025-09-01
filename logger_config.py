import logging
import os
from datetime import datetime

def setup_logger():
    """
    Sets up a custom logger that writes to a timestamped file and the console.
    """
    # Create a logs directory if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Generate a log file name with the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"{timestamp}.log")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level of logs to capture

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter to define the log message structure
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a stream handler to print logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("Logger initialized")
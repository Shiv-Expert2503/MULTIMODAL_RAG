import logging
import os
from datetime import datetime

# 1. Define the log filename with a timestamp
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# 2. Define the path for the "Logs" directory
# logs_dir = os.path.join("/tmp", "Logs") # Try adding "LOG_FILE" here as well
logs_dir = os.path.join(os.getcwd(), "Logs") # Try adding "LOG_FILE" here as well

# 3. Create the "Logs" directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# 4. Define the full path to the log file
log_file_path = os.path.join(logs_dir, LOG_FILE)

# 5. Configure the logging
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
import logging
import os

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok = True)

LOG_FILE = os.path.join(log_dir, "running_logs.log")

logging.basicConfig(
    filename = LOG_FILE,
    level = logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
)
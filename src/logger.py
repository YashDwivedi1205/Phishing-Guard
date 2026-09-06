from datetime import datetime
import logging
import os
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.info("Logger has started")
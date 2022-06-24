import logging
import time


logging.basicConfig(
    filename="../reports/logger_"+time.strftime("%Y%m%d-%H%M%S")+".log",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# logging hierarchy (each above contains every below):
# logging.DEBUG
# logging.INFO
# logging.WARNING
# logging.ERROR
# logging.CRITICAL

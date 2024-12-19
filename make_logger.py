import logging
import sys

# get the logging level from config
log_level = 'info'

# LOGGING
logger = logging.getLogger("hotspots")
if log_level == 'warning':
    logger.setLevel(logging.WARNING)
elif log_level == 'info':
    logger.setLevel(logging.INFO)
elif log_level == 'debug':
    logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

# create console handler
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logFormatter)

if log_level == 'warning':
    handler.setLevel(logging.WARNING)
elif log_level == 'info':
    handler.setLevel(logging.INFO)
elif log_level == 'debug':
    handler.setLevel(logging.DEBUG)

logger.addHandler(handler)
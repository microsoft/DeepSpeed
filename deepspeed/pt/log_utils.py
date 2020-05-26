import logging
import sys

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] "
    "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

logger = logging.getLogger("DeepSpeed")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel("INFO")
ch.setFormatter(formatter)
logger.addHandler(ch)

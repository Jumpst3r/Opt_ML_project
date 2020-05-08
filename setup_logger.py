import os
import logging

LOGGING_LEVEL = logging.INFO

# Setup logging
root = logging.getLogger()
root.setLevel(LOGGING_LEVEL)
format = "[%(asctime)s] [\u001b[32;1m %(levelname)s \u001b[0m] : %(message)s "
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(format, date_format)
console = logging.StreamHandler()
console.setLevel(level=logging.DEBUG)
console.setFormatter(formatter)
root.addHandler(console)
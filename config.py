# here go some directories used in this package
from pathlib import Path

_PKG_DIR = Path(__file__).parent
_EXAMPLEDATA_DIR = Path(_PKG_DIR, 'exampledata')

# global logging settings
LOGGING_LEVEL = 'info'  # choose from 'info', 'debug' or 'warning'
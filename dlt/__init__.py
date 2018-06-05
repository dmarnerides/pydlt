from . import config
from . import util
from . import viz
from . import train
from . import hdr
from .version import __version__

def _make_log():
    import sys
    import logging
    logger = logging.getLogger('dlt')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[dlt-{levelname}] {message}', style='{'))
    logger.addHandler(sh)

_make_log()

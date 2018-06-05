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
    sh.terminator = ''
    sh.setFormatter(logging.Formatter('[dlt-{levelname}] {message}\n', style='{'))
    logger.addHandler(sh)

_make_log()

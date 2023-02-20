import os
import logging
from colorlog import ColoredFormatter
from .common import timefn2

# base directory for this project. may be parent directory of this-file
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESOURCE_PATH = os.path.join(BASE_PATH, "resources")

__version__ = "1.0.1"
INIT_LOGGER_NAME = "fittingAI @antigravity"


def initialize_logger(LOGGER_NAME):

    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(log_color)s%(levelname)s - '
        '%(message)s'
    )
    # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s"
    formatter = ColoredFormatter(
        log_format,
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'yellow',
            'FATAL': 'red',
        }
    )

    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler("{}.log".format(LOGGER_NAME))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


runtime_logger = []


def get_runtime_logger():

    # packagename = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    packagename = INIT_LOGGER_NAME
    if len(runtime_logger) == 0:
        runtime_logger.append(packagename)
        initialize_logger(packagename)
    return logging.getLogger(packagename)


def change_runtime_logger_stream_level(level):
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    logger = get_runtime_logger()
    for hndler in logger.handlers:
        if isinstance(hndler, logging.StreamHandler):
            hndler.setLevel(level)


def test():
    packagename = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    print(packagename)
    file_dir = os.path.dirname(os.path.realpath(__file__))


    logger = get_runtime_logger()
    logger.info("test_predict")
    logger.fatal("test_predict")
    logger.fatal("test_predict")
    logger.critical("test_predict")
    logger.info("test_predict")
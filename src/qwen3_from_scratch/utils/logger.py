import logging


def get_logger(name: str):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger

import logging


def get_logger(name: str):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(name)
    return logger

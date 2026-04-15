from loguru import logger
import sys

from app.utils.config import settings


def setup_logger():
    logger.remove()  # Remove default logger

    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
    )

    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level=settings.LOG_LEVEL,
        format="{time} | {level} | {name}:{function}:{line} | {message}"
    )

    return logger


# Initialize logger
log = setup_logger()
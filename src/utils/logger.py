"""src/utils/logger.py - Loguru-based structured logging."""
import sys
import os
from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/rag.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


setup_logger()
__all__ = ["logger"]

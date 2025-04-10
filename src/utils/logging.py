import logging
from typing import Dict, Any, Optional


def setup_logger(
    name: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup a logger

    Args:
        name: Logger name (optional)
        log_level: Logging level, default is INFO

    Returns:
        Configured logger instance that only logs to console
    """
    # Get logger by name or root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


def log_dict(logger: logging.Logger, title: str, data: Dict[str, Any], level: int = logging.INFO) -> None:
    """
    Log a dictionary with a title at specified level

    Args:
        logger: Logger instance
        title: Title for the log entry
        data: Dictionary to log
        level: Logging level (default: INFO)
    """
    logger.log(level, f"{title}:")
    for key, value in data.items():
        logger.log(level, f"  {key}: {value}")


def debug(logger: logging.Logger, message: str) -> None:
    """Log a debug message"""
    logger.debug(message)


def info(logger: logging.Logger, message: str) -> None:
    """Log an info message"""
    logger.info(message)


def warning(logger: logging.Logger, message: str) -> None:
    """Log a warning message"""
    logger.warning(message)


def error(logger: logging.Logger, message: str) -> None:
    """Log an error message"""
    logger.error(message)


def critical(logger: logging.Logger, message: str) -> None:
    """Log a critical message"""
    logger.critical(message)

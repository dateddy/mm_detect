# src/utils/logger.py
"""Logging utilities with JSON-Lines file output support."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add any extra kwargs passed via logger.info("msg", extra={...})
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def get_logger(
    name: str, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Create a logger with StreamHandler and optional FileHandler.

    Args:
        name: Logger name.
        log_file: Optional path to log file. If provided, DEBUG-level events
                 are written as JSON-Lines format.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # StreamHandler at INFO level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # FileHandler at DEBUG level with JSON-Lines format (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    return logger
    
    return logger

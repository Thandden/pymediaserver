"""
Custom logger module that provides context-aware logging functionality.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Self
import logging
import sys


class LogLevel(Enum):
    """Log levels supported by the logger."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class Logger:
    """Custom logger with context support."""

    name: str
    level: LogLevel
    context: dict[str, Any] = field(default_factory=dict)
    _logger: Optional[logging.Logger] = None

    def __post_init__(self) -> None:
        """Initialize the underlying logger."""
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level.value)

        # Add console handler if none exists
        if not self._logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - [%(context)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",  # Fixed datetime format
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def _log(self, level: LogLevel, message: str) -> None:
        """Internal method to handle logging with context."""
        if self._logger is None:
            return

        context_str = (
            " ".join(f"{k}={v}" for k, v in self.context.items())
            if self.context
            else "-"
        )
        extra = {"context": context_str}
        self._logger.log(level.value, message, extra=extra)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self._log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self._log(LogLevel.ERROR, message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self._log(LogLevel.CRITICAL, message)

    def bind(self, **kwargs: Any) -> Self:
        """Create a new logger with additional persistent context."""
        new_context = self.context.copy()
        new_context.update(kwargs)
        return type(self)(self.name, self.level, new_context, self._logger)

    def with_context(self, **kwargs: Any) -> "ContextLogger":
        """Create a context manager for temporary context.

        Args:
            **kwargs: Context variables to add temporarily

        Returns:
            A context manager that provides a logger with the temporary context
        """
        return ContextLogger(self, **kwargs)


class ContextLogger:
    """Context manager for temporary logging context."""

    def __init__(self, logger: Logger, **kwargs: Any) -> None:
        """Initialize the context manager.

        Args:
            logger: The base logger to add context to
            **kwargs: Context variables to add temporarily
        """
        self.logger = logger
        self.context = kwargs
        self.temp_logger: Optional[Logger] = None

    def __enter__(self) -> Logger:
        """Enter the context, creating a new logger with the temporary context.

        Returns:
            A new logger instance with the temporary context
        """
        new_context = self.logger.context.copy()
        new_context.update(self.context)
        # Use a safer way to access the internal logger
        internal_logger = getattr(self.logger, "_logger", None)
        self.temp_logger = type(self.logger)(
            self.logger.name, self.logger.level, new_context, internal_logger
        )
        return self.temp_logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context, cleaning up the temporary logger."""
        self.temp_logger = None


def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    log_to_file: bool = False,
    log_file_path: Optional[str] = None,
) -> Logger:
    """Create a new logger instance.

    Args:
        name: Name of the logger
        level: Minimum log level to record
        log_to_file: Whether to also log to a file
        log_file_path: Path to the log file if log_to_file is True

    Returns:
        Configured Logger instance
    """
    logger = Logger(name, level)

    if log_to_file and log_file_path:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(path))  # Convert Path to str
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - [%(context)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # Fixed datetime format
        )
        file_handler.setFormatter(formatter)
        # Check if _logger exists before using it
        internal_logger = getattr(logger, "_logger", None)
        if internal_logger is not None:
            internal_logger.addHandler(file_handler)

    return logger

"""
Logging module for the project.
Provides unified logging to console and/or file.
"""

from typing import Literal
import os
import sys


# Global logger instance
_logger = None


class Logger:
    """
    Unified logging class that can write to console, file, or both.
    """

    def __init__(
        self,
        log_file: str = "../results/logs/training.log",
        mode: Literal["both", "console", "file"] = "both",
    ):
        """
        Initialize the logger.
        Args:
            log_file (str): Path to the log file.
            mode (str): Logging mode - "both", "console", or "file".
        """
        self.log_file = log_file
        self.mode = mode
        self.file_handle = None

        if mode in ["both", "file"]:
            self._ensure_dir_exists(os.path.dirname(log_file))
            self.file_handle = open(log_file, "a", encoding="utf-8")

    def _ensure_dir_exists(self, directory: str) -> None:
        """Create directory if it doesn't exist."""
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def log(self, message: str, end: str = "\n") -> None:
        """
        Log a message to console and/or file based on mode.
        Args:
            message (str): Message to log.
            end (str): End character (default: newline).
        """
        if self.mode in ["both", "console"]:
            print(message, end=end)
            sys.stdout.flush()

        if self.mode in ["both", "file"] and self.file_handle:
            self.file_handle.write(message + end)
            self.file_handle.flush()

    def close(self) -> None:
        """Close the log file if open."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self):
        """Context manager exit."""
        self.close()


def setup_logger(
    log_file: str = "../results/logs/training.log",
    mode: Literal["both", "console", "file"] = "both",
) -> Logger:
    """
    Setup and return a global logger instance.
    Args:
        log_file (str): Path to the log file.
        mode (str): Logging mode - "both", "console", or "file".
    Returns:
        Logger: Configured logger instance.
    """
    global _logger
    if _logger is not None:
        _logger.close()
    _logger = Logger(log_file, mode)
    return _logger


def get_logger() -> Logger:
    """
    Get the global logger instance. Creates one if it doesn't exist.
    Returns:
        Logger: Global logger instance.
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log(message: str, end: str = "\n") -> None:
    """
    Convenience function to log using the global logger.
    Works like print() but logs to both console and file.
    Args:
        message (str): Message to log.
        end (str): End character (default: newline).
    """
    logger = get_logger()
    logger.log(message, end=end)

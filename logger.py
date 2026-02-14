"""Logging utility that writes to both console and logs/ directory.

Redirects stdout and stderr so that all terminal output (including library
messages, tqdm bars, etc.) is also captured in the log file.
"""

import logging
import os
import sys
from datetime import datetime


class _TeeStream:
    """Stream wrapper that writes to both the original stream and a file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, msg):
        self._original.write(msg)
        self._log_file.write(msg)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


_log_file_handle = None


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to console and a timestamped file in logs/.

    On the first call, stdout and stderr are redirected so that *all* console
    output (library messages, tqdm, etc.) is also written to the log file.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ or the script name).
    """
    global _log_file_handle

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(stream=sys.__stdout__)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Redirect stdout/stderr to also write to the log file (once)
    if _log_file_handle is None:
        _log_file_handle = open(log_path, "a")
        sys.stdout = _TeeStream(sys.__stdout__, _log_file_handle)
        sys.stderr = _TeeStream(sys.__stderr__, _log_file_handle)

    return logger

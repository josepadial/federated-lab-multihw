"""
logging_utils.py

Centralized logging configuration for scripts and notebooks. Provides:
- get_logger(name): module-level logger with stream+file handlers.
- exception_safe: decorator to wrap functions with try/except and log.
- log_exceptions context manager for ad-hoc blocks.

Logs go to logs/run.log under project root. Use env LOG_LEVEL to override.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, TypeVar, Any, Optional

T = TypeVar("T")

_DEFAULT_LOG_FILE = (
        Path(__file__).resolve().parent.parent / "logs" / "run.log"
)
_DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_LOGGERS: dict[str, logging.Logger] = {}


def _level_from_env() -> int:
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)


def get_logger(name: str = "app") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(_level_from_env())
    logger.propagate = False
    # Clear existing handlers only once
    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setLevel(_level_from_env())
        sh.setFormatter(fmt)
        fh = logging.FileHandler(_DEFAULT_LOG_FILE.as_posix(), encoding="utf-8")
        fh.setLevel(_level_from_env())
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    _LOGGERS[name] = logger
    return logger


def set_level(level: int | str) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    for lg in _LOGGERS.values():
        lg.setLevel(level)
        for h in lg.handlers:
            h.setLevel(level)


def exception_safe(default: Optional[T] = None) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Decorator to catch exceptions, log them, and return a default value.
    Usage:
        @exception_safe(default=None)
        def risky(...):
            ...
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T | None]:
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            logger = get_logger(fn.__module__)
            try:
                return fn(*args, **kwargs)
            except Exception as ex:
                logger.exception("Unhandled exception in %s: %s", fn.__name__, ex)
                return default

        return wrapper

    return decorator


@contextmanager
def log_exceptions(msg: str, level: int | str = logging.ERROR):
    logger = get_logger("context")
    lvl = getattr(logging, level.upper(), level) if isinstance(level, str) else level
    try:
        yield
    except Exception as ex:
        logger.log(lvl, "%s: %s", msg, ex, exc_info=True)
        raise

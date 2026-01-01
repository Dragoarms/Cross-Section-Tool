#!/usr/bin/env python3
"""
Debug utilities and logging configuration for the geological cross-section tool suite.
Provides centralized logging control, debug decorators, and diagnostic tools.
"""

import logging
import functools
import time
import traceback
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
import inspect
import sys


class DebugLogger:
    """Centralized debug logging configuration and utilities."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the debug logger singleton."""
        if not self._initialized:
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)

            self.log_levels = {
                "TRACE": 5,  # Custom level for very detailed tracing
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }

            # Add custom TRACE level
            logging.addLevelName(5, "TRACE")

            self.formatters = {
                "detailed": logging.Formatter(
                    "%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ),
                "simple": logging.Formatter("%(levelname)s | %(name)s | %(message)s"),
                "minimal": logging.Formatter("%(message)s"),
            }

            self.active_loggers = {}
            self.log_file = None
            self.performance_stats = {}

            self.__class__._initialized = True

    def setup_module_logger(
        self,
        module_name: str,
        level: str = "INFO",
        console: bool = True,
        file: bool = True,
        format_style: str = "detailed",
    ) -> logging.Logger:
        """
        Set up a logger for a specific module with customizable output.

        Args:
            module_name: Name of the module
            level: Logging level
            console: Enable console output
            file: Enable file output
            format_style: Format style ('detailed', 'simple', 'minimal')

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(module_name)
        logger.setLevel(self.log_levels.get(level, logging.INFO))

        # Clear existing handlers
        logger.handlers = []

        formatter = self.formatters.get(format_style, self.formatters["detailed"])

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.log_levels.get(level, logging.INFO))
            logger.addHandler(console_handler)

        if file:
            file_path = self.log_dir / f"{module_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets everything
            logger.addHandler(file_handler)

            # Also add to main debug log
            if not self.log_file:
                self.log_file = self.log_dir / f"debug_{datetime.now():%Y%m%d_%H%M%S}.log"

            main_handler = logging.FileHandler(self.log_file)
            main_handler.setFormatter(self.formatters["detailed"])
            main_handler.setLevel(5)  # TRACE level
            logger.addHandler(main_handler)

        self.active_loggers[module_name] = logger
        return logger

    def set_global_level(self, level: str):
        """Set logging level for all active loggers."""
        level_value = self.log_levels.get(level, logging.INFO)
        for logger in self.active_loggers.values():
            logger.setLevel(level_value)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    handler.setLevel(level_value)

    def disable_module(self, module_name: str):
        """Disable logging for a specific module."""
        if module_name in self.active_loggers:
            self.active_loggers[module_name].disabled = True

    def enable_module(self, module_name: str):
        """Enable logging for a specific module."""
        if module_name in self.active_loggers:
            self.active_loggers[module_name].disabled = False

    def log_performance(self, module: str, function: str, duration: float, details: Dict = None):
        """Log performance metrics."""
        key = f"{module}.{function}"
        if key not in self.performance_stats:
            self.performance_stats[key] = {
                "count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
                "avg_time": 0,
            }

        stats = self.performance_stats[key]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        stats["avg_time"] = stats["total_time"] / stats["count"]

        if details:
            stats["last_details"] = details

    def get_performance_report(self) -> str:
        """Generate a performance report."""
        report = ["Performance Report", "=" * 50]

        for key, stats in sorted(self.performance_stats.items()):
            report.append(f"\n{key}:")
            report.append(f"  Calls: {stats['count']}")
            report.append(f"  Avg: {stats['avg_time']:.4f}s")
            report.append(f"  Min: {stats['min_time']:.4f}s")
            report.append(f"  Max: {stats['max_time']:.4f}s")
            report.append(f"  Total: {stats['total_time']:.4f}s")

        return "\n".join(report)


def debug_trace(
    logger: Optional[logging.Logger] = None,
    log_args: bool = True,
    log_result: bool = True,
    measure_time: bool = True,
):
    """
    Decorator to trace function calls with detailed logging.

    Args:
        logger: Logger instance to use
        log_args: Log function arguments
        log_result: Log function result
        measure_time: Measure and log execution time
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger if not provided
            nonlocal logger
            if logger is None:
                module = inspect.getmodule(func)
                module_name = module.__name__ if module else "unknown"
                logger = logging.getLogger(module_name)

            func_name = func.__name__

            # Log entry
            entry_msg = f"→ Entering {func_name}"
            if log_args:
                # Format args safely
                args_str = ", ".join(repr(arg)[:100] for arg in args)
                kwargs_str = ", ".join(f"{k}={repr(v)[:100]}" for k, v in kwargs.items())
                params = ", ".join(filter(None, [args_str, kwargs_str]))
                if params:
                    entry_msg += f"({params})"

            logger.log(5, entry_msg)  # TRACE level

            start_time = time.time() if measure_time else None
            exception = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result

            except Exception as e:
                exception = e
                logger.error(f"✗ Exception in {func_name}: {str(e)}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                raise

            finally:
                duration = time.time() - start_time if measure_time else None

                # Log exit
                exit_msg = f"← Exiting {func_name}"
                if exception:
                    exit_msg += f" (failed: {type(exception).__name__})"
                elif log_result and result is not None:
                    result_str = repr(result)[:100]
                    exit_msg += f" → {result_str}"

                if duration:
                    exit_msg += f" [{duration:.4f}s]"

                    # Log performance
                    if duration > 0.1:  # Only log if > 100ms
                        debug_logger = DebugLogger()
                        module = inspect.getmodule(func)
                        module_name = module.__name__ if module else "unknown"
                        debug_logger.log_performance(module_name, func_name, duration)

                logger.log(5, exit_msg)

        return wrapper

    return decorator


def debug_value(
    logger: logging.Logger,
    var_name: str,
    value: Any,
    truncate: int = 200,
    level: str = "DEBUG",
):
    """
    Log a variable value with automatic formatting and truncation.

    Args:
        logger: Logger instance
        var_name: Variable name for identification
        value: Value to log
        truncate: Maximum string length before truncation
        level: Logging level
    """
    level_value = getattr(logging, level, logging.DEBUG)

    # Format based on type
    if isinstance(value, (dict, list, tuple)):
        try:
            formatted = json.dumps(value, indent=2, default=str)
            if len(formatted) > truncate:
                formatted = formatted[:truncate] + "... (truncated)"
        except:
            formatted = repr(value)[:truncate]
    elif isinstance(value, str):
        formatted = value[:truncate] if len(value) > truncate else value
    else:
        formatted = repr(value)[:truncate]

    logger.log(level_value, f"{var_name} = {formatted}")


def log_state(logger: logging.Logger, obj: Any, attributes: List[str] = None):
    """
    Log the state of an object's attributes.

    Args:
        logger: Logger instance
        obj: Object to inspect
        attributes: List of attribute names to log (None = all)
    """
    class_name = obj.__class__.__name__
    logger.debug(f"State of {class_name} @ {id(obj):x}")

    if attributes is None:
        attributes = [a for a in dir(obj) if not a.startswith("_")]

    for attr in attributes:
        try:
            value = getattr(obj, attr)
            if not callable(value):
                debug_value(logger, f"  .{attr}", value, truncate=100)
        except Exception as e:
            logger.debug(f"  .{attr} = <error: {e}>")


class DebugContext:
    """Context manager for temporary debug level changes."""

    def __init__(self, level: str = "DEBUG", modules: List[str] = None):
        self.level = level
        self.modules = modules or []
        self.original_levels = {}
        self.debug_logger = DebugLogger()

    def __enter__(self):
        """Enter debug context - increase logging verbosity."""
        if self.modules:
            for module in self.modules:
                if module in self.debug_logger.active_loggers:
                    logger = self.debug_logger.active_loggers[module]
                    self.original_levels[module] = logger.level
                    logger.setLevel(self.debug_logger.log_levels[self.level])
        else:
            # Apply to all modules
            for module, logger in self.debug_logger.active_loggers.items():
                self.original_levels[module] = logger.level
                logger.setLevel(self.debug_logger.log_levels[self.level])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit debug context - restore original logging levels."""
        for module, level in self.original_levels.items():
            if module in self.debug_logger.active_loggers:
                self.debug_logger.active_loggers[module].setLevel(level)


def assert_with_info(condition: bool, message: str, **context):
    """
    Enhanced assertion with context information.

    Args:
        condition: Condition to assert
        message: Error message if assertion fails
        **context: Additional context variables to log
    """
    if not condition:
        error_msg = [f"Assertion failed: {message}"]
        if context:
            error_msg.append("Context:")
            for key, value in context.items():
                error_msg.append(f"  {key} = {repr(value)[:200]}")

        full_msg = "\n".join(error_msg)
        logging.error(full_msg)
        raise AssertionError(full_msg)


# Initialize singleton
debug_logger = DebugLogger()

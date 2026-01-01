"""
Utility modules for geological tools.
"""

try:
    from .debug_utils import (
        DebugLogger,
        debug_trace,
        debug_value,
        log_state,
        DebugContext,
        assert_with_info,
    )

    __all__ = [
        "DebugLogger",
        "debug_trace",
        "debug_value",
        "log_state",
        "DebugContext",
        "assert_with_info",
    ]
except ImportError:
    print("Warning: debug_utils not found in utils directory")

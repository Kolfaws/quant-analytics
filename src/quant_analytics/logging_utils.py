"""logging_utils — loguru 统一配置（幂等）。"""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """控制台 + 滚动文件日志；重复调用不会重复挂 handler。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    logger.add(
        "logs/analytics.log",
        level=level,
        rotation="10 MB",
        retention=5,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True

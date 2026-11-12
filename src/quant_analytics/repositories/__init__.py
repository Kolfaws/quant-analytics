"""只读数据访问层：所有 SQL 下沉到 repositories。"""

from __future__ import annotations

from .backtest_repo import (
    ExcessNav,
    get_benchmark_defs,
    get_benchmark_nav,
    get_excess_nav,
    get_nav_by_runs,
    get_rank_ic_series,
    get_run,
    get_run_metrics,
    list_success_runs,
)

__all__ = [
    "ExcessNav",
    "get_benchmark_defs",
    "get_benchmark_nav",
    "get_excess_nav",
    "get_nav_by_runs",
    "get_rank_ic_series",
    "get_run",
    "get_run_metrics",
    "list_success_runs",
]

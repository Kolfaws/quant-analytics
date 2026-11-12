"""backtest_schema — backtest.duckdb 的字段映射与兼容层。

数据库是只读的；本模块只描述“实际有什么列、缺列时怎么退化”，不执行
业务查询。页面/服务不应直接依赖列名，而是通过这里的映射拿到统一字段。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

import duckdb

RUN_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "strategy_id",
    "strategy_name",
    "benchmark_id",
    "params_json",
    "start_date",
    "end_date",
    "initial_capital",
    "cost_model",
    "status",
    "engine_version",
    "config_hash",
    "created_at",
    "finished_at",
)
RUN_OPTIONAL_FIELDS: Final[tuple[str, ...]] = (
    "model_type",
    "factor_id",
    "factor_name",
)
NAV_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "trade_date",
    "nav",
    "ret_1d",
    "cum_ret",
    "drawdown",
    "benchmark_id",
)
METRIC_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "metric_name",
    "metric_value",
    "computed_at",
)
BENCHMARK_FIELDS: Final[tuple[str, ...]] = (
    "benchmark_id",
    "name",
    "kind",
    "category",
    "is_active",
)
BASELINE_FIELDS: Final[tuple[str, ...]] = (
    "benchmark_id",
    "trade_date",
    "close",
    "ret_1d",
    "nav",
    "base_date",
    "base_value",
)
EXCESS_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "trade_date",
    "benchmark_id",
    "nav",
    "ret_1d",
    "benchmark_nav",
    "benchmark_ret_1d",
    "benchmark_nav_rel",
    "excess_nav",
    "excess_ret_1d",
)

RANK_IC_TABLE_CANDIDATES: Final[tuple[str, ...]] = (
    "bt_rank_ic",
    "bt_run_rank_ic",
    "bt_daily_rank_ic",
    "bt_factor_rank_ic",
)

METRIC_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "total_return": ("total_return",),
    "annualized_return": ("annualized_return", "annual_return"),
    "annualized_volatility": ("annualized_volatility", "annual_vol"),
    "sharpe_ratio": ("sharpe_ratio", "sharpe"),
    "max_drawdown": ("max_drawdown", "max_dd"),
    "calmar_ratio": ("calmar_ratio", "calmar"),
    "win_rate": ("win_rate",),
    "excess_total_return": ("excess_total_return",),
    "excess_annualized_return": ("excess_annualized_return", "excess_annual_return"),
    "tracking_error": ("tracking_error",),
    "information_ratio": ("information_ratio", "ir"),
    "beta": ("beta",),
}


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> tuple[str, ...]:
    """返回表/视图的实际列名；不存在返回空 tuple。"""
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table],
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """表/视图是否存在；不抛异常。"""
    return bool(table_columns(con, table))


def parse_params(raw: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """把 params_json 解析为 dict；坏 JSON 返回空 dict。"""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _first_value(params: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = params.get(key)
        if value not in (None, ""):
            return value
    return None


def run_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """把一行 bt_runs 归一化，并从 params_json 补 model/factor 信息。

    真实 bt_runs 当前没有 model_type/factor_id/factor_name 列；缺失时从
    ``params_json`` 读取，仍然缺失就返回 None。
    """
    params = parse_params(row.get("params_json"))
    factor_id = row.get("factor_id") or _first_value(params, "factor_id", "factor_ids", "factor")
    if isinstance(factor_id, list | tuple):
        factor_id = factor_id[0] if factor_id else None
    factor_name = row.get("factor_name") or _first_value(
        params, "factor_name", "factor_names", "factor"
    )
    if isinstance(factor_name, list | tuple):
        factor_name = factor_name[0] if factor_name else None
    model_type = row.get("model_type") or _first_value(params, "model_type", "model", "model_name")
    return {
        **{field: row.get(field) for field in RUN_FIELDS},
        "model_type": model_type,
        "factor_id": factor_id,
        "factor_name": factor_name,
        "params": params,
    }


def metric_value(
    wide: Mapping[str, float | None],
    metric_name: str,
) -> float | None:
    """从宽表指标行中读一个指标，尝试已知别名。"""
    for alias in METRIC_ALIASES.get(metric_name, (metric_name,)):
        value = wide.get(alias)
        if value is not None:
            return float(value)
    return None


def rank_ic_metric_names(columns: tuple[str, ...]) -> tuple[str, ...]:
    """bt_run_metrics 中实际存在的 RankIC 指标列（通常为空）。"""
    return tuple(column for column in columns if column.startswith("rank_ic"))


__all__ = [
    "BASELINE_FIELDS",
    "BENCHMARK_FIELDS",
    "EXCESS_FIELDS",
    "METRIC_ALIASES",
    "METRIC_FIELDS",
    "NAV_FIELDS",
    "RANK_IC_TABLE_CANDIDATES",
    "RUN_FIELDS",
    "RUN_OPTIONAL_FIELDS",
    "metric_value",
    "parse_params",
    "rank_ic_metric_names",
    "run_metadata",
    "table_columns",
    "table_exists",
]

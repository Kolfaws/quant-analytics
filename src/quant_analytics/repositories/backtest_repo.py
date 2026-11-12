"""backtest_repo — backtest.duckdb 的只读查询层。

约定：

* 每条公开查询只读一个库，连接始终 ``read_only=True``；
* 页面不拼 SQL；SQL 全部在本文件；
* 查询显式列裁剪，禁止 ``SELECT *``；
* 多 run 查询一次 ``WHERE run_id IN (...)``，不做 N+1；
* 缓存 key 由 ``env + db_mtime + refresh_token + 参数`` 组成。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import polars as pl
import streamlit as st

from quant_analytics.db import (
    BACKTEST_DB_PATH,
    BACKTEST_TEST_DB_PATH,
    DataAccessError,
    _connect,
)

from .backtest_schema import (
    BASELINE_FIELDS,
    BENCHMARK_FIELDS,
    EXCESS_FIELDS,
    NAV_FIELDS,
    RANK_IC_TABLE_CANDIDATES,
    RUN_FIELDS,
    run_metadata,
    table_columns,
    table_exists,
)

RUN_SCHEMA: Final[dict[str, pl.DataType]] = {
    "run_id": pl.Utf8(),
    "strategy_id": pl.Utf8(),
    "strategy_name": pl.Utf8(),
    "benchmark_id": pl.Utf8(),
    "params_json": pl.Utf8(),
    "start_date": pl.Date(),
    "end_date": pl.Date(),
    "initial_capital": pl.Float64(),
    "cost_model": pl.Utf8(),
    "status": pl.Utf8(),
    "engine_version": pl.Utf8(),
    "config_hash": pl.Utf8(),
    "created_at": pl.Datetime(time_zone="UTC"),
    "finished_at": pl.Datetime(time_zone="UTC"),
    "model_type": pl.Utf8(),
    "factor_id": pl.Utf8(),
    "factor_name": pl.Utf8(),
}
NAV_SCHEMA: Final[dict[str, pl.DataType]] = {
    "run_id": pl.Utf8(),
    "trade_date": pl.Date(),
    "nav": pl.Float64(),
    "ret_1d": pl.Float64(),
    "cum_ret": pl.Float64(),
    "drawdown": pl.Float64(),
    "benchmark_id": pl.Utf8(),
}
METRIC_SCHEMA: Final[dict[str, pl.DataType]] = {
    "run_id": pl.Utf8(),
    "metric_name": pl.Utf8(),
    "metric_value": pl.Float64(),
    "computed_at": pl.Datetime(time_zone="UTC"),
}
BENCHMARK_SCHEMA: Final[dict[str, pl.DataType]] = {
    "benchmark_id": pl.Utf8(),
    "name": pl.Utf8(),
    "kind": pl.Utf8(),
    "category": pl.Utf8(),
    "is_active": pl.Boolean(),
}
BASELINE_SCHEMA: Final[dict[str, pl.DataType]] = {
    "benchmark_id": pl.Utf8(),
    "trade_date": pl.Date(),
    "close": pl.Float64(),
    "ret_1d": pl.Float64(),
    "nav": pl.Float64(),
    "base_date": pl.Date(),
    "base_value": pl.Float64(),
}
EXCESS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "run_id": pl.Utf8(),
    "trade_date": pl.Date(),
    "benchmark_id": pl.Utf8(),
    "nav": pl.Float64(),
    "ret_1d": pl.Float64(),
    "benchmark_nav": pl.Float64(),
    "benchmark_ret_1d": pl.Float64(),
    "benchmark_nav_rel": pl.Float64(),
    "excess_nav": pl.Float64(),
    "excess_ret_1d": pl.Float64(),
}
RANK_IC_SCHEMA: Final[dict[str, pl.DataType]] = {
    "run_id": pl.Utf8(),
    "trade_date": pl.Date(),
    "rank_ic": pl.Float64(),
}


@dataclass(frozen=True, slots=True)
class ExcessNav:
    """超额净值查询结果及来源标记。"""

    frame: pl.DataFrame
    source: str  # view / fallback
    note: str = ""


def _validate_env(env: str) -> str:
    if env not in ("prod", "test"):
        raise DataAccessError(f"未知环境：{env!r}")
    return env


def _path(env: str) -> Path:
    path = BACKTEST_TEST_DB_PATH if _validate_env(env) == "test" else BACKTEST_DB_PATH
    if path is None or not path.is_file():
        raise DataAccessError(f"{env} backtest 库不存在或未配置")
    return path


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _frame(
    rows: list[tuple[object, ...]],
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, orient="row") if rows else _empty(schema)


def _run_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return _empty(RUN_SCHEMA)
    normalized = [{field: row.get(field) for field in RUN_SCHEMA} for row in rows]
    return pl.DataFrame(normalized, schema=RUN_SCHEMA)


def list_success_runs(env: str = "prod", *, refresh_token: int = 0) -> pl.DataFrame:
    """成功 run 的元数据；缺 model/factor 时从 params_json 解析。"""
    path = _path(env)
    return _list_success_runs_cached(str(path), _mtime(path), int(refresh_token))


@st.cache_data(ttl=3600, show_spinner=False)
def _list_success_runs_cached(path: str, db_mtime: float, refresh_token: int) -> pl.DataFrame:
    _ = (db_mtime, refresh_token)
    with _connect(Path(path)) as con:
        columns = set(table_columns(con, "bt_runs"))
        if not columns or not {"run_id", "status"}.issubset(columns):
            return _empty(RUN_SCHEMA)
        select = ", ".join(field for field in RUN_FIELDS if field in columns)
        if not select:
            return _empty(RUN_SCHEMA)
        rows = con.execute(
            f"SELECT {select} FROM bt_runs "
            "WHERE status = 'success' "
            "ORDER BY coalesce(finished_at, created_at) DESC"
        ).fetchall()
        names = [field for field in RUN_FIELDS if field in columns]
    return _run_frame([run_metadata(dict(zip(names, row, strict=True))) for row in rows])


def get_run(env: str, run_id: str, *, refresh_token: int = 0) -> dict[str, object] | None:
    """单个 run 元数据；不存在返回 None。"""
    path = _path(env)
    return _get_run_cached(str(path), _mtime(path), int(refresh_token), run_id)


@st.cache_data(ttl=3600, show_spinner=False)
def _get_run_cached(
    path: str, db_mtime: float, refresh_token: int, run_id: str
) -> dict[str, object] | None:
    _ = (db_mtime, refresh_token)
    with _connect(Path(path)) as con:
        columns = set(table_columns(con, "bt_runs"))
        if "run_id" not in columns:
            return None
        select = ", ".join(field for field in RUN_FIELDS if field in columns)
        row = con.execute(
            f"SELECT {select} FROM bt_runs WHERE run_id = ? LIMIT 1", [run_id]
        ).fetchone()
        if row is None:
            return None
        names = [field for field in RUN_FIELDS if field in columns]
    return run_metadata(dict(zip(names, row, strict=True)))


def get_run_metrics(
    env: str,
    run_ids: tuple[str, ...],
    *,
    refresh_token: int = 0,
) -> pl.DataFrame:
    """bt_run_metrics 长表转宽表；空/缺表返回空。"""
    if not run_ids:
        return _empty(METRIC_SCHEMA)
    path = _path(env)
    return _get_run_metrics_cached(
        str(path), _mtime(path), int(refresh_token), tuple(sorted(run_ids))
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_run_metrics_cached(
    path: str, db_mtime: float, refresh_token: int, run_ids: tuple[str, ...]
) -> pl.DataFrame:
    _ = (db_mtime, refresh_token)
    if not run_ids:
        return _empty(METRIC_SCHEMA)
    placeholders = ", ".join("?" for _ in run_ids)
    with _connect(Path(path)) as con:
        columns = set(table_columns(con, "bt_run_metrics"))
        if not columns or not {"run_id", "metric_name", "metric_value"}.issubset(columns):
            return _empty(METRIC_SCHEMA)
        select = ", ".join(
            field for field in ("run_id", "metric_name", "metric_value") if field in columns
        )
        rows = con.execute(
            f"SELECT {select} FROM bt_run_metrics "
            f"WHERE run_id IN ({placeholders}) ORDER BY run_id, metric_name",
            list(run_ids),
        ).fetchall()
    long = _frame(
        rows, {"run_id": pl.Utf8(), "metric_name": pl.Utf8(), "metric_value": pl.Float64()}
    )
    if long.is_empty():
        return _empty(METRIC_SCHEMA)
    return long.pivot(
        index="run_id", on="metric_name", values="metric_value", aggregate_function="last"
    )


def get_nav_by_runs(
    env: str,
    run_ids: tuple[str, ...],
    start: date | None = None,
    end: date | None = None,
    *,
    refresh_token: int = 0,
) -> pl.DataFrame:
    """一次查询多个 run 的日净值。"""
    if not run_ids:
        return _empty(NAV_SCHEMA)
    path = _path(env)
    return _get_nav_cached(
        str(path),
        _mtime(path),
        int(refresh_token),
        tuple(sorted(run_ids)),
        start,
        end,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_nav_cached(
    path: str,
    db_mtime: float,
    refresh_token: int,
    run_ids: tuple[str, ...],
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    _ = (db_mtime, refresh_token)
    placeholders = ", ".join("?" for _ in run_ids)
    where = [f"run_id IN ({placeholders})"]
    params: list[object] = list(run_ids)
    if start is not None:
        where.append("trade_date >= ?")
        params.append(start)
    if end is not None:
        where.append("trade_date <= ?")
        params.append(end)
    with _connect(Path(path)) as con:
        columns = set(table_columns(con, "bt_nav"))
        if "run_id" not in columns:
            return _empty(NAV_SCHEMA)
        select = ", ".join(field for field in NAV_FIELDS if field in columns)
        rows = con.execute(
            f"SELECT {select} FROM bt_nav WHERE {' AND '.join(where)} ORDER BY trade_date, run_id",
            params,
        ).fetchall()
        names = [field for field in NAV_FIELDS if field in columns]
    schema = {name: NAV_SCHEMA[name] for name in names}
    return _frame(rows, schema)


def get_benchmark_defs(env: str = "prod", *, refresh_token: int = 0) -> pl.DataFrame:
    """基准定义；只取展示所需字段。"""
    path = _path(env)
    return _get_benchmarks_cached(str(path), _mtime(path), int(refresh_token))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_benchmarks_cached(path: str, db_mtime: float, refresh_token: int) -> pl.DataFrame:
    _ = (db_mtime, refresh_token)
    with _connect(Path(path)) as con:
        columns = set(table_columns(con, "bt_benchmarks"))
        if not columns:
            return _empty(BENCHMARK_SCHEMA)
        select = ", ".join(field for field in BENCHMARK_FIELDS if field in columns)
        rows = con.execute(f"SELECT {select} FROM bt_benchmarks ORDER BY name").fetchall()
        names = [field for field in BENCHMARK_FIELDS if field in columns]
    return _frame(rows, {name: BENCHMARK_SCHEMA[name] for name in names})


def get_benchmark_nav(
    env: str,
    benchmark_ids: tuple[str, ...],
    start: date | None = None,
    end: date | None = None,
    *,
    refresh_token: int = 0,
) -> pl.DataFrame:
    """一次查询多个基准的日净值/点位。"""
    if not benchmark_ids:
        return _empty(BASELINE_SCHEMA)
    path = _path(env)
    return _get_baseline_nav_cached(
        str(path),
        _mtime(path),
        int(refresh_token),
        tuple(sorted(benchmark_ids)),
        start,
        end,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_baseline_nav_cached(
    path: str,
    db_mtime: float,
    refresh_token: int,
    benchmark_ids: tuple[str, ...],
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    _ = (db_mtime, refresh_token)
    placeholders = ", ".join("?" for _ in benchmark_ids)
    where = [f"benchmark_id IN ({placeholders})"]
    params: list[object] = list(benchmark_ids)
    if start is not None:
        where.append("trade_date >= ?")
        params.append(start)
    if end is not None:
        where.append("trade_date <= ?")
        params.append(end)
    with _connect(Path(path)) as con:
        columns = set(table_columns(con, "bt_baselines"))
        if "benchmark_id" not in columns:
            return _empty(BASELINE_SCHEMA)
        select = ", ".join(field for field in BASELINE_FIELDS if field in columns)
        rows = con.execute(
            f"SELECT {select} FROM bt_baselines WHERE {' AND '.join(where)} "
            "ORDER BY benchmark_id, trade_date",
            params,
        ).fetchall()
        names = [field for field in BASELINE_FIELDS if field in columns]
    return _frame(rows, {name: BASELINE_SCHEMA[name] for name in names})


def get_excess_nav(
    env: str,
    run_id: str,
    benchmark_id: str | None,
    start: date | None = None,
    end: date | None = None,
    *,
    refresh_token: int = 0,
) -> ExcessNav:
    """优先读 v_bt_nav_excess；视图不可用返回空并标记 fallback。"""
    path = _path(env)
    return _get_excess_cached(
        str(path),
        _mtime(path),
        int(refresh_token),
        run_id,
        benchmark_id,
        start,
        end,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _get_excess_cached(
    path: str,
    db_mtime: float,
    refresh_token: int,
    run_id: str,
    benchmark_id: str | None,
    start: date | None,
    end: date | None,
) -> ExcessNav:
    _ = (db_mtime, refresh_token)
    with _connect(Path(path)) as con:
        if not table_exists(con, "v_bt_nav_excess"):
            return ExcessNav(_empty(EXCESS_SCHEMA), "fallback", "v_bt_nav_excess 不存在")
        columns = set(table_columns(con, "v_bt_nav_excess"))
        required = {"run_id", "trade_date", "excess_nav"}
        if not required.issubset(columns):
            return ExcessNav(
                _empty(EXCESS_SCHEMA),
                "fallback",
                "v_bt_nav_excess 缺关键列",
            )
        select = ", ".join(field for field in EXCESS_FIELDS if field in columns)
        where = ["run_id = ?"]
        params: list[object] = [run_id]
        if benchmark_id:
            where.append("benchmark_id = ?")
            params.append(benchmark_id)
        if start is not None:
            where.append("trade_date >= ?")
            params.append(start)
        if end is not None:
            where.append("trade_date <= ?")
            params.append(end)
        try:
            rows = con.execute(
                f"SELECT {select} FROM v_bt_nav_excess WHERE {' AND '.join(where)} "
                "ORDER BY trade_date",
                params,
            ).fetchall()
        except Exception as exc:  # noqa: BLE001 - 视图列/权限变化时回退
            return ExcessNav(_empty(EXCESS_SCHEMA), "fallback", f"视图查询失败：{exc}")
        names = [field for field in EXCESS_FIELDS if field in columns]
    frame = _frame(rows, {name: EXCESS_SCHEMA[name] for name in names})
    if frame.is_empty():
        return ExcessNav(frame, "fallback", "v_bt_nav_excess 无重叠数据")
    return ExcessNav(frame, "view", "")


def get_rank_ic_series(
    env: str,
    run_ids: tuple[str, ...],
    *,
    refresh_token: int = 0,
) -> pl.DataFrame:
    """只在存在日度 RankIC 表时读取；否则返回空。"""
    if not run_ids:
        return _empty(RANK_IC_SCHEMA)
    path = _path(env)
    return _get_rank_ic_cached(str(path), _mtime(path), int(refresh_token), tuple(sorted(run_ids)))


@st.cache_data(ttl=3600, show_spinner=False)
def _get_rank_ic_cached(
    path: str, db_mtime: float, refresh_token: int, run_ids: tuple[str, ...]
) -> pl.DataFrame:
    _ = (db_mtime, refresh_token)
    placeholders = ", ".join("?" for _ in run_ids)
    with _connect(Path(path)) as con:
        table = next(
            (name for name in RANK_IC_TABLE_CANDIDATES if table_exists(con, name)),
            None,
        )
        if table is None:
            return _empty(RANK_IC_SCHEMA)
        columns = set(table_columns(con, table))
        required = {"run_id", "trade_date", "rank_ic"}
        if not required.issubset(columns):
            return _empty(RANK_IC_SCHEMA)
        rows = con.execute(
            f"SELECT run_id, trade_date, rank_ic FROM {table} "
            f"WHERE run_id IN ({placeholders}) ORDER BY run_id, trade_date",
            list(run_ids),
        ).fetchall()
    return _frame(rows, RANK_IC_SCHEMA)


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

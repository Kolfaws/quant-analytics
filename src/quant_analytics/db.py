"""db — 只读 DuckDB 数据访问 + Streamlit 缓存。

约定：

* 所有连接都使用 ``duckdb.connect(path, read_only=True)``，不锁库；
* 所有公开查询函数都加 ``@st.cache_data(ttl=3600)``；
* 数据库异常统一包装成 :class:`DataAccessError`，由页面用 ``st.error`` 展示；
* 这里只做查询，不做指标计算（计算在 ``quant_analytics.metrics``）。
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Final

import duckdb
import polars as pl
import streamlit as st
from loguru import logger

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


MARKET_DB_PATH: Final[Path | None] = _env_path("MARKET_DB_PATH")
BACKTEST_DB_PATH: Final[Path | None] = _env_path("BACKTEST_DB_PATH")
MARKET_TEST_DB_PATH: Final[Path | None] = _env_path("MARKET_TEST_DB_PATH")
BACKTEST_TEST_DB_PATH: Final[Path | None] = _env_path("BACKTEST_TEST_DB_PATH")


def _market_path(env: str) -> Path | None:
    _validate_env(env)
    return MARKET_TEST_DB_PATH if env == "test" else MARKET_DB_PATH


def _backtest_path(env: str) -> Path | None:
    _validate_env(env)
    return BACKTEST_TEST_DB_PATH if env == "test" else BACKTEST_DB_PATH


def _validate_env(env: str) -> str:
    if env not in ("prod", "test"):
        raise DataAccessError(f"未知环境：{env!r}")
    return env


class DataAccessError(RuntimeError):
    """数据库缺失、表缺失或查询为空。"""


def _connect(path: Path | None) -> duckdb.DuckDBPyConnection:
    if path is None:
        raise DataAccessError(
            "未配置数据库路径：请设置环境变量 MARKET_DB_PATH / BACKTEST_DB_PATH "
            "（可写入项目根目录 .env）"
        )
    if not path.is_file():
        raise DataAccessError(f"数据库不存在：{path}")
    try:
        con = duckdb.connect(str(path), read_only=True)
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '2GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = false")
        return con
    except Exception as exc:  # noqa: BLE001 - 统一包装给 UI
        logger.exception("打开 DuckDB 失败: {}", path)
        raise DataAccessError(f"打开数据库失败：{path.name}：{exc}") from exc


def _query_frame(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[object],
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    rows = con.execute(sql, params).fetchall()
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")


def _run_exists(con: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    rows = con.execute("SELECT 1 FROM bt_runs WHERE run_id = ? LIMIT 1", [run_id]).fetchall()
    return bool(rows)


def _resolve_run_id(con: duckdb.DuckDBPyConnection, requested: str) -> str:
    """允许传 run_id 或 strategy_id；strategy_id 取最新成功 run。"""
    if _run_exists(con, requested):
        return requested
    rows = con.execute(
        """
        SELECT run_id FROM bt_runs
        WHERE strategy_id = ? AND status = 'success'
        ORDER BY coalesce(finished_at, created_at) DESC
        LIMIT 1
        """,
        [requested],
    ).fetchall()
    if not rows:
        raise DataAccessError(f"找不到策略/运行：{requested}")
    return str(rows[0][0])


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_options(
    env: str = "prod",
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """返回 (策略列表, 基准列表, 因子列表)。

    单个库不可用时返回空列表，页面根据空列表给出提示，不直接崩。
    """
    strategies: list[dict[str, str]] = []
    baselines: list[dict[str, str]] = []
    factors: list[dict[str, str]] = []

    try:
        with _connect(_backtest_path(env)) as con:
            rows = con.execute(
                """
                SELECT run_id, strategy_id, coalesce(strategy_name, strategy_id),
                       start_date, end_date
                FROM bt_runs
                WHERE status = 'success'
                ORDER BY coalesce(finished_at, created_at) DESC
                """
            ).fetchall()
            for run_id, strategy_id, name, start_date, end_date in rows:
                period = f"{start_date} ~ {end_date}" if start_date else ""
                strategies.append(
                    {
                        "run_id": str(run_id),
                        "strategy_id": str(strategy_id),
                        "label": f"{name} · {run_id} · {period}",
                    }
                )
            rows = con.execute(
                "SELECT benchmark_id, name FROM bt_benchmarks ORDER BY name"
            ).fetchall()
            for benchmark_id, name in rows:
                baselines.append(
                    {"benchmark_id": str(benchmark_id), "label": f"{name} ({benchmark_id})"}
                )
    except Exception as exc:  # noqa: BLE001 - 页面降级处理
        logger.exception("读取 backtest.duckdb 选项失败: {}", exc)

    try:
        with _connect(_market_path(env)) as con:
            rows = con.execute(
                """
                SELECT factor_id, factor_name, coalesce(factor_group, ''),
                       coalesce(description, '')
                FROM meta_factor_registry
                WHERE is_active = TRUE
                ORDER BY factor_name
                """
            ).fetchall()
            for factor_id, factor_name, group, description in rows:
                label = factor_name if not group else f"{factor_name} · {group}"
                factors.append(
                    {
                        "factor_id": str(factor_id),
                        "factor_name": str(factor_name),
                        "label": label,
                        "description": str(description),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - 因子页单独提示
        logger.warning("读取因子目录失败: {}", exc)

    return strategies, baselines, factors


@st.cache_data(ttl=3600, show_spinner=False)
def get_nav_data(strategy_id: str, baseline_id: str, *, env: str = "prod") -> pl.DataFrame:
    """策略与基准的日频净值/收益，按 trade_date 对齐。"""
    try:
        with _connect(_backtest_path(env)) as con:
            run_id = _resolve_run_id(con, strategy_id)
            frame = _query_frame(
                con,
                """
                SELECT n.trade_date,
                       n.nav          AS strategy_nav,
                       n.ret_1d       AS strategy_ret,
                       b.nav          AS baseline_nav,
                       b.ret_1d       AS baseline_ret
                FROM bt_nav n
                JOIN bt_baselines b
                  ON b.trade_date = n.trade_date
                WHERE n.run_id = ? AND b.benchmark_id = ?
                ORDER BY n.trade_date
                """,
                [run_id, baseline_id],
                {
                    "trade_date": pl.Date(),
                    "strategy_nav": pl.Float64(),
                    "strategy_ret": pl.Float64(),
                    "baseline_nav": pl.Float64(),
                    "baseline_ret": pl.Float64(),
                },
            )
    except DataAccessError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取净值失败: run={} baseline={}", strategy_id, baseline_id)
        raise DataAccessError(f"读取净值失败：{exc}") from exc

    if frame.is_empty():
        raise DataAccessError(
            f"策略 {strategy_id} 与基准 {baseline_id} 在 bt_nav / bt_baselines 中没有重叠数据"
        )
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def get_factor_data(
    factor_name: str, start_date: str, end_date: str, *, env: str = "prod"
) -> pl.DataFrame:
    """因子截面 + 下期收益（future return）。

    下期收益在 Polars 中按 ``ts_code`` 分组 shift(-1) 计算，保证向量化。
    """
    try:
        with _connect(_market_path(env)) as con:
            rows = con.execute(
                "SELECT factor_id FROM meta_factor_registry WHERE factor_name = ?",
                [factor_name],
            ).fetchall()
            if not rows:
                raise DataAccessError(f"因子不存在或未启用：{factor_name}")
            factor_id = int(rows[0][0])

            raw = _query_frame(
                con,
                """
                SELECT f.ts_code,
                       f.trade_date,
                       f.value AS factor_value,
                       CAST(d.close AS DOUBLE) AS close
                FROM fact_factor_daily f
                JOIN fact_daily d
                  ON d.ts_code = f.ts_code AND d.trade_date = f.trade_date
                WHERE f.factor_id = ?
                  AND f.trade_date BETWEEN ? AND ?
                  AND f.is_stale = FALSE
                ORDER BY f.ts_code, f.trade_date
                """,
                [factor_id, start_date, end_date],
                {
                    "ts_code": pl.Utf8(),
                    "trade_date": pl.Date(),
                    "factor_value": pl.Float64(),
                    "close": pl.Float64(),
                },
            )
    except DataAccessError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取因子失败: factor={}", factor_name)
        raise DataAccessError(f"读取因子失败：{exc}") from exc

    if raw.is_empty():
        raise DataAccessError(f"因子 {factor_name} 在 {start_date} ~ {end_date} 没有数据")

    frame = (
        raw.sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("close").shift(-1).over("ts_code") / pl.col("close") - 1.0).alias(
                "forward_return"
            )
        )
        .filter(
            pl.col("factor_value").is_not_null()
            & pl.col("factor_value").is_finite()
            & pl.col("forward_return").is_not_null()
            & pl.col("forward_return").is_finite()
        )
        .drop_nulls(["ts_code", "trade_date"])
    )
    if frame.is_empty():
        raise DataAccessError(f"因子 {factor_name} 计算下期收益后为空（区间太短或价格缺失）")
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def get_factor_date_range(factor_name: str, *, env: str = "prod") -> tuple[date, date] | None:
    """因子在库中的可用日期范围，供侧边栏设置默认值。"""
    try:
        with _connect(_market_path(env)) as con:
            rows = con.execute(
                """
                SELECT min(f.trade_date), max(f.trade_date)
                FROM fact_factor_daily f
                JOIN meta_factor_registry r ON r.factor_id = f.factor_id
                WHERE r.factor_name = ? AND f.is_stale = FALSE
                """,
                [factor_name],
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取因子日期范围失败: {}", exc)
        return None
    if not rows or rows[0][0] is None:
        return None
    return rows[0][0], rows[0][1]


@st.cache_data(ttl=3600, show_spinner=False)
def get_data_status(env: str = "prod") -> dict[str, object]:
    """首页用的轻量状态：各表行数与最新日期。"""
    status: dict[str, object] = {
        "market_db": str(_market_path(env) or "未配置"),
        "backtest_db": str(_backtest_path(env) or "未配置"),
        "backtest_rows": {},
        "factor_rows": 0,
        "latest_nav": None,
        "latest_baseline": None,
    }
    try:
        with _connect(_backtest_path(env)) as con:
            for table in ("bt_runs", "bt_nav", "bt_baselines", "bt_benchmarks"):
                rows = con.execute(f'SELECT count(*) FROM "{table}"').fetchall()
                status["backtest_rows"][table] = int(rows[0][0])  # type: ignore[index]
            latest = con.execute("SELECT max(trade_date) FROM bt_nav").fetchall()
            status["latest_nav"] = latest[0][0] if latest else None
            latest = con.execute("SELECT max(trade_date) FROM bt_baselines").fetchall()
            status["latest_baseline"] = latest[0][0] if latest else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取回测库状态失败: {}", exc)
    try:
        with _connect(_market_path(env)) as con:
            rows = con.execute("SELECT count(*) FROM fact_factor_daily").fetchall()
            status["factor_rows"] = int(rows[0][0]) if rows else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取因子库状态失败: {}", exc)
    return status


__all__ = [
    "BACKTEST_DB_PATH",
    "BACKTEST_TEST_DB_PATH",
    "MARKET_DB_PATH",
    "MARKET_TEST_DB_PATH",
    "DataAccessError",
    "get_available_options",
    "get_data_status",
    "get_factor_data",
    "get_factor_date_range",
    "get_nav_data",
]

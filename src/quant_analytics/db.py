"""db — 只读 DuckDB 数据访问 + Streamlit 缓存。

约定：

* 所有连接都使用 ``duckdb.connect(path, read_only=True)``，不锁库；
* 所有公开查询函数都加 ``@st.cache_data(ttl=3600)``；
* 数据库异常统一包装成 :class:`DataAccessError`，由页面用 ``st.error`` 展示；
* 这里只做查询，不做指标计算（计算在 ``quant_analytics.metrics``）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


def db_cache_token(env: str, which: str = "market") -> int:
    """库文件 mtime，用作 ``st.cache_data`` 的 cache key 组成部分。

    文件不存在/不可 stat 时返回 0；不打开数据库。
    """
    if which == "market":
        path = _market_path(env)
    elif which == "backtest":
        path = _backtest_path(env)
    else:
        raise ValueError(f"未知库类型：{which!r}；可选 ('market', 'backtest')")
    if path is None:
        return 0
    try:
        return path.stat().st_mtime_ns if path.is_file() else 0
    except OSError:
        return 0


class DataAccessError(RuntimeError):
    """数据库缺失、表缺失或查询为空。"""


@dataclass(frozen=True, slots=True)
class FactorSampleReport:
    """因子样本过滤统计：不可交易/缺收益的行显式计数，不静默丢弃。"""

    total_rows: int
    kept_rows: int
    dropped_missing_flag: int = 0
    dropped_out_of_universe: int = 0
    dropped_suspended: int = 0
    dropped_limit_up: int = 0
    dropped_not_buyable: int = 0
    dropped_invalid_factor: int = 0
    dropped_no_next_trade_date: int = 0
    dropped_missing_forward_return: int = 0
    kept_not_sellable: int = 0

    @property
    def dropped_rows(self) -> int:
        return self.total_rows - self.kept_rows

    def describe(self) -> list[str]:
        """页面提示用中文摘要；只列出非零剔除项。"""
        lines = [f"样本 {self.kept_rows:,} / {self.total_rows:,} 行（剔除 {self.dropped_rows:,}）"]
        reasons = (
            ("缺少 fact_tradable_flag 行", self.dropped_missing_flag),
            ("in_universe=FALSE", self.dropped_out_of_universe),
            ("信号日停牌", self.dropped_suspended),
            ("信号日涨停（不可买）", self.dropped_limit_up),
            ("is_buyable=FALSE（其他）", self.dropped_not_buyable),
            ("因子值无效", self.dropped_invalid_factor),
            ("下一交易日未知", self.dropped_no_next_trade_date),
            ("下期收益缺失（停牌/缺行）", self.dropped_missing_forward_return),
        )
        dropped = [f"{label} {count:,}" for label, count in reasons if count]
        if dropped:
            lines.append("剔除：" + "；".join(dropped))
        if self.kept_not_sellable:
            lines.append(
                f"注意：保留样本中 {self.kept_not_sellable:,} 行在信号日 is_sellable=FALSE；"
                "卖出侧未并入本过滤，按 is_sellable 另算"
            )
        return lines


@dataclass(frozen=True, slots=True)
class FactorData:
    """因子页数据：方向调整后的因子值 + 下期收益 + 样本过滤报告。"""

    factor_name: str
    direction: int
    frame: pl.DataFrame
    report: FactorSampleReport

    @property
    def direction_label(self) -> str:
        if self.direction == 1:
            return "正向（direction=1）"
        return f"负向（direction={self.direction}），因子值已翻转"


def _connect(path: Path | None) -> duckdb.DuckDBPyConnection:
    if path is None:
        raise DataAccessError(
            "未配置数据库路径：请设置环境变量 MARKET_DB_PATH / BACKTEST_DB_PATH "
            "（可写入项目根目录 .env）"
        )
    if not path.is_file():
        raise DataAccessError(f"数据库文件不存在：{path.name}")
    try:
        con = duckdb.connect(str(path), read_only=True)
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '2GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = false")
        return con
    except Exception as exc:  # noqa: BLE001 - 统一包装给 UI
        logger.exception("打开 DuckDB 失败: {}", path.name)
        raise DataAccessError(f"打开数据库失败：{path.name}：{exc}") from exc


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_options(
    env: str = "prod", *, cache_token: int = 0
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """返回 (策略列表, 基准列表, 因子列表)。

    ``cache_token`` 由页面传 ``db_cache_token``；库文件更新后缓存自动失效。
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
                       coalesce(description, ''), CAST(direction AS INTEGER)
                FROM meta_factor_registry
                WHERE is_active = TRUE
                ORDER BY factor_name
                """
            ).fetchall()
            for factor_id, factor_name, group, description, direction in rows:
                label = factor_name if not group else f"{factor_name} · {group}"
                factors.append(
                    {
                        "factor_id": str(factor_id),
                        "factor_name": str(factor_name),
                        "label": label,
                        "description": str(description),
                        "direction": "-1" if direction == -1 else "1",
                    }
                )
    except Exception as exc:  # noqa: BLE001 - 因子页单独提示
        logger.warning("读取因子目录失败: {}", exc)

    return strategies, baselines, factors


@st.cache_data(ttl=3600, show_spinner=False)
def get_factor_data(
    factor_name: str,
    start_date: str,
    end_date: str,
    *,
    env: str = "prod",
    cache_token: int = 0,
) -> FactorData:
    """因子截面 + 下期收益（future return）+ 可交易过滤报告。

    口径：

    * 下期收益 = ``v_daily_return.ret_1d``（``pct_chg/100``）在**下一交易日**的值，
      不使用未复权 ``close.shift(-1)/close-1``；下一交易日停牌/缺行则剔除并计数，
      多日收益不会被当成一日收益。
    * 样本 join ``fact_tradable_flag``：信号日必须 ``in_universe`` 且 ``is_buyable``；
      剔除行按原因计数返回，不静默丢弃。``is_sellable``（卖出侧）不并入本过滤，
      保留在返回列中供卖出侧另算。
    * 读 ``meta_factor_registry.direction``：``-1``（负向）时对因子值取反，
      使 IC / 分层 G1 都按正向口径展示；方向随 :class:`FactorData` 返回。
    """
    # 延迟导入：repositories/__init__ 会导入 backtest_repo，而后者依赖本模块
    # 的路径常量；在 db 模块导入阶段引入会形成循环导入。
    from .repositories.factor_repo import FactorQueryError, load_factor_sample

    try:
        with _connect(_market_path(env)) as con:
            raw, direction = load_factor_sample(con, factor_name, start_date, end_date)
        return _build_factor_data(raw, factor_name, direction, start_date, end_date)
    except FactorQueryError as exc:
        raise DataAccessError(str(exc)) from exc
    except DataAccessError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("读取因子失败: factor={}", factor_name)
        raise DataAccessError(f"读取因子失败：{exc}") from exc


def _build_factor_data(
    raw: pl.DataFrame,
    factor_name: str,
    direction: int,
    start_date: str,
    end_date: str,
) -> FactorData:
    if raw.is_empty():
        raise DataAccessError(f"因子 {factor_name} 在 {start_date} ~ {end_date} 没有数据")

    total_rows = raw.height
    flagged = raw.filter(pl.col("in_universe").is_not_null())
    missing_flag = total_rows - flagged.height
    out_of_universe = flagged.filter(pl.col("in_universe") == False)  # noqa: E712
    in_universe = flagged.filter(pl.col("in_universe") == True)  # noqa: E712
    suspended = in_universe.filter(pl.col("is_suspended") == True)  # noqa: E712
    traded = in_universe.filter(pl.col("is_suspended") == False)  # noqa: E712
    limit_up = traded.filter(pl.col("is_limit_up") == True)  # noqa: E712
    boughtable = traded.filter(pl.col("is_limit_up") == False)  # noqa: E712
    not_buyable = boughtable.filter(pl.col("is_buyable") == False)  # noqa: E712
    eligible = boughtable.filter(pl.col("is_buyable") == True)  # noqa: E712

    valid_factor = eligible.filter(
        pl.col("factor_value_raw").is_not_null() & pl.col("factor_value_raw").is_finite()
    )
    invalid_factor = eligible.height - valid_factor.height
    dated = valid_factor.filter(pl.col("next_trade_date").is_not_null())
    no_next_date = valid_factor.height - dated.height
    returns = dated.filter(
        pl.col("forward_return").is_not_null() & pl.col("forward_return").is_finite()
    )
    missing_forward = dated.height - returns.height
    kept = returns
    kept_not_sellable = kept.filter(pl.col("is_sellable") == False).height  # noqa: E712

    report = FactorSampleReport(
        total_rows=total_rows,
        kept_rows=kept.height,
        dropped_missing_flag=missing_flag,
        dropped_out_of_universe=out_of_universe.height,
        dropped_suspended=suspended.height,
        dropped_limit_up=limit_up.height,
        dropped_not_buyable=not_buyable.height,
        dropped_invalid_factor=invalid_factor,
        dropped_no_next_trade_date=no_next_date,
        dropped_missing_forward_return=missing_forward,
        kept_not_sellable=kept_not_sellable,
    )
    if kept.is_empty():
        breakdown = "；".join(report.describe())
        raise DataAccessError(
            f"因子 {factor_name} 在 {start_date} ~ {end_date} 过滤后没有可用样本：{breakdown}"
        )

    frame = (
        kept.with_columns(
            (pl.col("factor_value_raw") * float(direction)).alias("factor_value"),
            pl.lit(direction, dtype=pl.Int8).alias("direction"),
        )
        .select(
            "ts_code",
            "trade_date",
            "factor_value",
            "factor_value_raw",
            "forward_return",
            "direction",
            "in_universe",
            "is_buyable",
            "is_sellable",
        )
        .sort(["ts_code", "trade_date"])
    )
    return FactorData(
        factor_name=factor_name,
        direction=direction,
        frame=frame,
        report=report,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_factor_date_range(
    factor_name: str, *, env: str = "prod", cache_token: int = 0
) -> tuple[date, date] | None:
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


__all__ = [
    "BACKTEST_DB_PATH",
    "BACKTEST_TEST_DB_PATH",
    "MARKET_DB_PATH",
    "MARKET_TEST_DB_PATH",
    "DataAccessError",
    "FactorData",
    "FactorSampleReport",
    "get_available_options",
    "get_factor_data",
    "get_factor_date_range",
]

"""factor_repo — 因子页的只读 SQL / 关系校验。

只依赖 DuckDB 连接与 Polars；不读取环境变量、不连库、不做 Streamlit 缓存。
``db.get_factor_data`` 负责连接与缓存，本模块负责 SQL 与样本行。
"""

from __future__ import annotations

from typing import Final

import duckdb
import polars as pl

#: 因子页下期收益与可交易过滤依赖的 market 关系（表/视图）。
REQUIRED_RELATIONS: Final[tuple[str, ...]] = (
    "fact_factor_daily",
    "meta_factor_registry",
    "fact_tradable_flag",
    "dim_trade_cal",
    "v_daily_return",
)

_FACTOR_ROW_SCHEMA: Final[dict[str, pl.DataType]] = {
    "ts_code": pl.Utf8(),
    "trade_date": pl.Date(),
    "factor_value_raw": pl.Float64(),
    "next_trade_date": pl.Date(),
    "in_universe": pl.Boolean(),
    "is_buyable": pl.Boolean(),
    "is_sellable": pl.Boolean(),
    "is_suspended": pl.Boolean(),
    "is_limit_up": pl.Boolean(),
    "forward_return": pl.Float64(),
}


class FactorQueryError(RuntimeError):
    """因子页所需关系缺失、因子不存在或方向非法。"""


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


def _existing_relations(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _relation_columns(con: duckdb.DuckDBPyConnection, name: str) -> tuple[str, ...]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [name],
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _missing_relations(con: duckdb.DuckDBPyConnection) -> list[str]:
    existing = _existing_relations(con)
    return [name for name in REQUIRED_RELATIONS if name not in existing]


def _factor_sample_frame(
    con: duckdb.DuckDBPyConnection,
    factor_id: int,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """因子行 + 信号日可交易标记 + 下一交易日 pct_chg 收益。

    下期收益取下一**交易日**（``dim_trade_cal``）的 ``v_daily_return.ret_1d``；
    下一交易日停牌/缺行时收益为 NULL，由调用方剔除并计数，不把多日收益当一日。
    """
    return _query_frame(
        con,
        """
        WITH cal AS (
            SELECT cal_date AS trade_date,
                   lead(cal_date) OVER (ORDER BY cal_date) AS next_trade_date
            FROM (SELECT DISTINCT cal_date FROM dim_trade_cal WHERE is_open = TRUE)
        )
        SELECT f.ts_code,
               f.trade_date,
               CAST(f.value AS DOUBLE) AS factor_value_raw,
               cal.next_trade_date,
               t.in_universe,
               t.is_buyable,
               t.is_sellable,
               t.is_suspended,
               t.is_limit_up,
               CAST(r.ret_1d AS DOUBLE) AS forward_return
        FROM fact_factor_daily f
        LEFT JOIN fact_tradable_flag t
          ON t.ts_code = f.ts_code AND t.trade_date = f.trade_date
        LEFT JOIN cal
          ON cal.trade_date = f.trade_date
        LEFT JOIN v_daily_return r
          ON r.ts_code = f.ts_code AND r.trade_date = cal.next_trade_date
        WHERE f.factor_id = ?
          AND f.trade_date BETWEEN ? AND ?
          AND f.is_stale = FALSE
        ORDER BY f.ts_code, f.trade_date
        """,
        [factor_id, start_date, end_date],
        _FACTOR_ROW_SCHEMA,
    )


def load_factor_sample(
    con: duckdb.DuckDBPyConnection,
    factor_name: str,
    start_date: str,
    end_date: str,
) -> tuple[pl.DataFrame, int]:
    """返回 ``(因子样本行, direction)``；关系/因子/方向不合法时抛 ``FactorQueryError``。"""
    missing = _missing_relations(con)
    if missing:
        raise FactorQueryError(
            "market 库缺少因子页依赖：" + "、".join(missing) + "；"
            "fact_tradable_flag 请先运行 build_tradable_flags.py，"
            "v_daily_return 需 schema >= 1.3.0"
        )
    registry_columns = _relation_columns(con, "meta_factor_registry")
    if "direction" not in registry_columns:
        raise FactorQueryError("meta_factor_registry 缺少 direction 列，无法标注因子方向")
    rows = con.execute(
        "SELECT factor_id, direction FROM meta_factor_registry WHERE factor_name = ?",
        [factor_name],
    ).fetchall()
    if not rows:
        raise FactorQueryError(f"因子不存在或未启用：{factor_name}")
    factor_id = int(rows[0][0])
    direction = 1 if rows[0][1] is None else int(rows[0][1])
    if direction not in (1, -1):
        raise FactorQueryError(f"因子 {factor_name} 的 direction={direction} 非法（只允许 1 / -1）")
    return _factor_sample_frame(con, factor_id, start_date, end_date), direction


__all__ = [
    "FactorQueryError",
    "REQUIRED_RELATIONS",
    "load_factor_sample",
]

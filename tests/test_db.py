"""db.py 因子页 DB 级测试：下期收益口径、可交易过滤、direction。

用 tmp_path 建最小 market 库（fact_daily + v_daily_return + fact_tradable_flag
+ dim_trade_cal + factor 表），不连正式库。
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from quant_analytics import db
from quant_analytics.metrics import calc_daily_ic

D1 = date(2024, 1, 2)
D2 = date(2024, 1, 3)
D3 = date(2024, 1, 4)
D4 = date(2024, 1, 5)

FlagRow = tuple[str, date, bool, bool, bool, bool, bool, bool, bool]


def _flag(
    ts_code: str,
    day: date,
    *,
    in_universe: bool = True,
    buyable: bool = True,
    sellable: bool = True,
    suspended: bool = False,
    limit_up: bool = False,
) -> FlagRow:
    """fact_tradable_flag 行：(ts_code, date, is_suspended, is_st, is_limit_up,
    is_limit_down, is_buyable, is_sellable, in_universe)。"""
    return (ts_code, day, suspended, False, limit_up, False, buyable, sellable, in_universe)


def _create_market_db(
    path: Path,
    *,
    calendar: tuple[date, ...],
    daily: list[tuple[str, date, float, float | None]],
    factors: list[tuple[str, date, float | None]],
    flags: list[FlagRow] | None = None,
    direction: int = 1,
    with_return_view: bool = True,
) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE dim_trade_cal ("
            "exchange VARCHAR NOT NULL, cal_date DATE NOT NULL, "
            "is_open BOOLEAN NOT NULL, pretrade_date DATE)"
        )
        con.executemany(
            "INSERT INTO dim_trade_cal VALUES ('SSE', ?, TRUE, NULL)",
            [(day,) for day in calendar],
        )
        con.execute(
            "CREATE TABLE fact_daily ("
            "ts_code VARCHAR NOT NULL, trade_date DATE NOT NULL, "
            "close DOUBLE, pct_chg DOUBLE)"
        )
        con.executemany(
            "INSERT INTO fact_daily VALUES (?, ?, ?, ?)",
            list(daily),
        )
        if with_return_view:
            con.execute(
                "CREATE VIEW v_daily_return AS "
                "SELECT ts_code, trade_date, CAST(pct_chg AS DOUBLE) / 100.0 AS ret_1d "
                "FROM fact_daily WHERE pct_chg IS NOT NULL"
            )
        con.execute(
            "CREATE TABLE meta_factor_registry ("
            "factor_id INTEGER NOT NULL, factor_name VARCHAR NOT NULL, "
            "factor_group VARCHAR, value_type VARCHAR, "
            "direction SMALLINT NOT NULL DEFAULT 1, description VARCHAR, "
            "is_active BOOLEAN NOT NULL DEFAULT TRUE)"
        )
        con.execute(
            "INSERT INTO meta_factor_registry (factor_id, factor_name, direction) "
            "VALUES (1, 'F', ?)",
            [direction],
        )
        con.execute(
            "CREATE TABLE fact_factor_daily ("
            "ts_code VARCHAR NOT NULL, trade_date DATE NOT NULL, "
            "factor_id INTEGER NOT NULL, value DOUBLE, "
            "is_stale BOOLEAN NOT NULL DEFAULT FALSE, source_batch_id BIGINT)"
        )
        con.executemany(
            "INSERT INTO fact_factor_daily (ts_code, trade_date, factor_id, value) "
            "VALUES (?, ?, 1, ?)",
            list(factors),
        )
        if flags is not None:
            con.execute(
                "CREATE TABLE fact_tradable_flag ("
                "ts_code VARCHAR NOT NULL, trade_date DATE NOT NULL, "
                "is_suspended BOOLEAN NOT NULL, is_st BOOLEAN NOT NULL, "
                "is_limit_up BOOLEAN NOT NULL, is_limit_down BOOLEAN NOT NULL, "
                "is_buyable BOOLEAN NOT NULL, is_sellable BOOLEAN NOT NULL, "
                "in_universe BOOLEAN NOT NULL)"
            )
            con.executemany(
                "INSERT INTO fact_tradable_flag VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                list(flags),
            )
    finally:
        con.close()


@pytest.fixture()
def market_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离 market 测试库；正式库路径指向不存在的文件，防止误连。"""
    path = tmp_path / "market_test.duckdb"
    monkeypatch.setattr(db, "MARKET_TEST_DB_PATH", path)
    monkeypatch.setattr(db, "MARKET_DB_PATH", tmp_path / "prod_not_allowed.duckdb")
    db.get_factor_data.clear()
    yield path
    db.get_factor_data.clear()


def _query(path: Path) -> db.FactorData:
    return db.get_factor_data("F", D1.isoformat(), D3.isoformat(), env="test")


def test_forward_return_uses_pct_chg_on_split_day(market_path: Path) -> None:
    """拆股日：未复权 close 腰斩，forward_return 必须对齐 pct_chg/100。"""
    _create_market_db(
        market_path,
        calendar=(D1, D2, D3, D4),
        daily=[
            # A 在 D2 1 拆 2：未复权 close 10 -> 5，pct_chg 是复权后收益 +2%
            ("A", D1, 10.0, 0.0),
            ("A", D2, 5.0, 2.0),
            ("A", D3, 5.1, 2.0),
            ("B", D1, 20.0, 0.0),
            ("B", D2, 20.4, 2.0),
            ("B", D3, 20.8, 1.96),
        ],
        factors=[
            ("A", D1, 1.0),
            ("A", D2, 2.0),
            ("B", D1, 3.0),
            ("B", D2, 1.0),
        ],
        flags=[
            _flag("A", D1),
            _flag("A", D2),
            _flag("A", D3),
            _flag("B", D1),
            _flag("B", D2),
            _flag("B", D3),
        ],
    )
    result = _query(market_path)
    frame = result.frame

    assert result.direction == 1
    assert result.report.total_rows == 4
    assert result.report.kept_rows == 4
    split_row = frame.filter((pl.col("ts_code") == "A") & (pl.col("trade_date") == D1))
    forward = split_row["forward_return"].item()
    assert forward == pytest.approx(0.02)
    # 旧口径 close.shift(-1)/close-1 在拆股日会得到 -50%
    assert forward != pytest.approx(-0.5)
    a_next = frame.filter((pl.col("ts_code") == "A") & (pl.col("trade_date") == D2))
    assert a_next["forward_return"].item() == pytest.approx(0.02)
    b_first = frame.filter((pl.col("ts_code") == "B") & (pl.col("trade_date") == D1))
    assert b_first["forward_return"].item() == pytest.approx(0.02)


def test_suspended_next_day_is_dropped_not_multiday(market_path: Path) -> None:
    """下一交易日停牌/缺行：剔除并计数，不能把多日收益当一日。"""
    _create_market_db(
        market_path,
        calendar=(D1, D2, D3, D4),
        daily=[
            ("A", D1, 10.0, 0.0),
            # A 在 D2 停牌（无 fact_daily 行），D3 的 pct_chg 覆盖多日 +10%
            ("A", D3, 11.0, 10.0),
            ("A", D4, 11.5, 4.55),
            ("B", D1, 20.0, 0.0),
            ("B", D2, 20.2, 1.0),
            ("B", D3, 20.4, 0.99),
            ("B", D4, 20.6, 0.98),
        ],
        factors=[
            ("A", D1, 1.0),
            ("A", D3, 2.0),
            ("B", D1, 3.0),
            ("B", D3, 1.5),
        ],
        flags=[
            _flag("A", D1),
            _flag("A", D3),
            _flag("A", D4),
            _flag("B", D1),
            _flag("B", D2),
            _flag("B", D3),
            _flag("B", D4),
        ],
    )
    result = _query(market_path)
    frame = result.frame

    # A@D1 的下一交易日 D2 没有收益行 -> 剔除并计数，不拿 D3 的 +10% 顶替
    assert result.report.dropped_missing_forward_return == 1
    assert result.report.kept_rows == 3
    assert frame.filter(pl.col("ts_code") == "A")["trade_date"].to_list() == [D3]
    assert frame["forward_return"].max() == pytest.approx(0.0455)
    assert 0.10 not in frame["forward_return"].to_list()


def test_tradability_filter_reports_dropped_rows(market_path: Path) -> None:
    """in_universe / is_buyable 过滤；卖出侧单独计数，不静默丢弃。"""
    _create_market_db(
        market_path,
        calendar=(D1, D2),
        daily=[
            ("A", D1, 10.0, 0.0),
            ("A", D2, 10.1, 1.0),
            ("B", D1, 20.0, 0.0),
            ("B", D2, 21.0, 5.0),
            ("C", D1, 30.0, 0.0),
            ("C", D2, 30.3, 1.0),
            ("D", D1, 40.0, 0.0),
            ("D", D2, 40.4, 1.0),
        ],
        factors=[("A", D1, 1.0), ("B", D1, 2.0), ("C", D1, 3.0), ("D", D1, 4.0)],
        flags=[
            # A 可买但信号日不可卖：卖出侧另算，不算进买入过滤
            _flag("A", D1, sellable=False),
            _flag("A", D2),
            # B 涨停 -> is_buyable False
            _flag("B", D1, buyable=False, limit_up=True),
            _flag("B", D2),
            # C 不在股票池
            _flag("C", D1, in_universe=False, buyable=False, sellable=False),
            _flag("C", D2),
            # D 完全没有可交易标记行
        ],
    )
    result = db.get_factor_data("F", D1.isoformat(), D1.isoformat(), env="test")
    frame = result.frame

    assert frame["ts_code"].to_list() == ["A"]
    report = result.report
    assert report.total_rows == 4
    assert report.kept_rows == 1
    assert report.dropped_out_of_universe == 1
    assert report.dropped_limit_up == 1
    assert report.dropped_missing_flag == 1
    assert report.kept_not_sellable == 1
    # 所有剔除原因都进入报告，不静默丢弃
    assert report.dropped_rows == 3
    assert any("剔除" in line for line in report.describe())
    assert any("is_sellable" in line for line in report.describe())


def test_negative_direction_flips_factor_value(market_path: Path) -> None:
    """direction=-1：因子值取反，IC 符号翻转，并标注方向。"""
    _create_market_db(
        market_path,
        calendar=(D1, D2),
        daily=[
            ("A", D1, 10.0, 0.0),
            ("A", D2, 10.1, 1.0),
            ("B", D1, 20.0, 0.0),
            ("B", D2, 20.4, 2.0),
        ],
        factors=[("A", D1, 1.0), ("B", D1, 3.0)],
        flags=[_flag("A", D1), _flag("A", D2), _flag("B", D1), _flag("B", D2)],
        direction=-1,
    )
    result = _query(market_path)
    frame = result.frame

    assert result.direction == -1
    assert "-1" in result.direction_label
    assert frame["direction"].to_list() == [-1, -1]
    # 原始值保留，用于追溯；因子值已按负向取反
    assert frame["factor_value_raw"].to_list() == [1.0, 3.0]
    assert frame["factor_value"].to_list() == [-1.0, -3.0]

    raw_ic = frame.select(pl.corr("factor_value_raw", "forward_return").alias("ic"))["ic"].item()
    assert raw_ic > 0
    adjusted = calc_daily_ic(frame)
    assert adjusted["ic"].item() < 0


def test_missing_tradable_flag_table_raises(market_path: Path) -> None:
    _create_market_db(
        market_path,
        calendar=(D1, D2),
        daily=[("A", D1, 10.0, 0.0), ("A", D2, 10.1, 1.0)],
        factors=[("A", D1, 1.0)],
        flags=None,
    )
    with pytest.raises(db.DataAccessError, match="fact_tradable_flag"):
        _query(market_path)


def test_missing_return_view_raises(market_path: Path) -> None:
    _create_market_db(
        market_path,
        calendar=(D1, D2),
        daily=[("A", D1, 10.0, 0.0), ("A", D2, 10.1, 1.0)],
        factors=[("A", D1, 1.0)],
        flags=[_flag("A", D1), _flag("A", D2)],
        with_return_view=False,
    )
    with pytest.raises(db.DataAccessError, match="v_daily_return"):
        _query(market_path)


def test_all_rows_filtered_raises_with_breakdown(market_path: Path) -> None:
    _create_market_db(
        market_path,
        calendar=(D1, D2),
        daily=[("B", D1, 20.0, 0.0), ("B", D2, 21.0, 5.0)],
        factors=[("B", D1, 2.0)],
        flags=[_flag("B", D1, buyable=False, limit_up=True), _flag("B", D2)],
    )
    with pytest.raises(db.DataAccessError, match="过滤后没有可用样本"):
        _query(market_path)


def test_report_categories_are_mutually_exclusive_and_complete(market_path: Path) -> None:
    """剔除原因计数必须完整可加总，不能有静默丢弃。"""
    _create_market_db(
        market_path,
        calendar=(D1, D2),
        daily=[
            ("A", D1, 10.0, 0.0),
            ("A", D2, 10.1, 1.0),
            ("B", D1, 20.0, 0.0),
            ("B", D2, 20.8, 4.0),
            ("C", D1, 30.0, 0.0),
            ("C", D2, 30.3, 1.0),
            ("D", D1, 40.0, 0.0),
            ("D", D2, 40.4, 1.0),
            ("F", D1, 50.0, 0.0),  # 次交易日无行情 -> forward return 缺失
            ("G", D1, 60.0, 0.0),  # 信号日停牌
            ("H", D2, 70.0, 1.0),  # 区间最后一天 -> 下一交易日未知
        ],
        factors=[
            ("A", D1, 1.0),
            ("B", D1, 2.0),
            ("C", D1, 3.0),
            ("D", D1, 4.0),
            ("E", D1, None),  # 因子值无效
            ("F", D1, 5.0),
            ("G", D1, 6.0),
            ("H", D2, 7.0),
        ],
        flags=[
            _flag("A", D1),
            _flag("A", D2),
            _flag("B", D1, buyable=False, limit_up=True),
            _flag("B", D2),
            _flag("C", D1, in_universe=False, buyable=False, sellable=False),
            _flag("C", D2),
            _flag("E", D1),
            _flag("F", D1),
            _flag("G", D1, suspended=True, buyable=False, sellable=False),
            _flag("H", D2),
        ],
    )
    result = db.get_factor_data("F", D1.isoformat(), D2.isoformat(), env="test")
    report = result.report

    assert report.total_rows == 8
    assert report.kept_rows == 1
    assert report.dropped_missing_flag == 1
    assert report.dropped_out_of_universe == 1
    assert report.dropped_suspended == 1
    assert report.dropped_limit_up == 1
    assert report.dropped_invalid_factor == 1
    assert report.dropped_missing_forward_return == 1
    assert report.dropped_no_next_trade_date == 1
    assert report.dropped_rows == 7
    categorized = (
        report.dropped_missing_flag
        + report.dropped_out_of_universe
        + report.dropped_suspended
        + report.dropped_limit_up
        + report.dropped_not_buyable
        + report.dropped_invalid_factor
        + report.dropped_no_next_trade_date
        + report.dropped_missing_forward_return
    )
    assert categorized == report.dropped_rows
    assert report.kept_rows + categorized == report.total_rows


def test_db_cache_token_tracks_file(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "market.duckdb"
    path.write_bytes(b"a")
    monkeypatch.setattr(db, "MARKET_DB_PATH", path)
    first = db.db_cache_token("prod")
    assert first == path.stat().st_mtime_ns
    os.utime(path, ns=(first, first + 1_000_000_000))
    assert db.db_cache_token("prod") != first


def test_db_cache_token_missing_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(db, "MARKET_DB_PATH", tmp_path / "missing.duckdb")
    assert db.db_cache_token("prod") == 0


def test_db_cache_token_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="未知库类型"):
        db.db_cache_token("prod", "other")

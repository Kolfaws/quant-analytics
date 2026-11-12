"""backtest_repo：只读、显式列、批量查询与视图回退。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from quant_analytics.repositories import backtest_repo


def _make_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE bt_runs (run_id VARCHAR, strategy_id VARCHAR, strategy_name VARCHAR, "
        "benchmark_id VARCHAR, params_json VARCHAR, start_date DATE, end_date DATE, "
        "initial_capital DECIMAL(20,4), cost_model VARCHAR, status VARCHAR, "
        "engine_version VARCHAR, config_hash VARCHAR, created_at TIMESTAMPTZ, finished_at TIMESTAMPTZ)"
    )
    con.execute(
        "CREATE TABLE bt_nav (run_id VARCHAR, trade_date DATE, nav DOUBLE, ret_1d DOUBLE, "
        "cum_ret DOUBLE, drawdown DOUBLE, benchmark_id VARCHAR)"
    )
    con.execute(
        "CREATE TABLE bt_run_metrics (run_id VARCHAR, metric_name VARCHAR, "
        "metric_value DOUBLE, computed_at TIMESTAMPTZ)"
    )
    con.execute(
        "CREATE TABLE bt_benchmarks (benchmark_id VARCHAR, name VARCHAR, kind VARCHAR, "
        "category VARCHAR, is_active BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE bt_baselines (benchmark_id VARCHAR, trade_date DATE, close DOUBLE, "
        "ret_1d DOUBLE, nav DOUBLE, base_date DATE, base_value DOUBLE)"
    )
    con.execute(
        "INSERT INTO bt_runs VALUES ('r1','s1','Demo','BM','{}',DATE '2024-01-01',"
        "DATE '2024-01-03',1000000,'x','success','1','h',now(),now())"
    )
    con.execute(
        "INSERT INTO bt_runs VALUES ('r2','s2','Bad','BM','{}',NULL,NULL,1000000,'x','failed','1','h',now(),now())"
    )
    for day, nav in ((date(2024, 1, 2), 1.0), (date(2024, 1, 3), 1.1)):
        con.execute("INSERT INTO bt_nav VALUES ('r1', ?, ?, 0.1, 0.1, 0.0, 'BM')", [day, nav])
        con.execute(
            "INSERT INTO bt_baselines VALUES ('BM', ?, ?, 0.1, ?, DATE '2024-01-02', 1.0)",
            [day, nav, nav],
        )
    con.execute("INSERT INTO bt_run_metrics VALUES ('r1','annualized_return',0.2,now())")
    con.execute("INSERT INTO bt_benchmarks VALUES ('BM','Bench','synthetic','basket',TRUE)")
    con.close()


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "backtest.duckdb"
    _make_db(path)
    monkeypatch.setattr(backtest_repo, "BACKTEST_DB_PATH", path)
    monkeypatch.setattr(backtest_repo, "BACKTEST_TEST_DB_PATH", path)
    return path


def test_repo_reads_and_shapes(db: Path) -> None:
    runs = backtest_repo.list_success_runs("prod", refresh_token=1)
    assert runs["run_id"].to_list() == ["r1"]
    metrics = backtest_repo.get_run_metrics("prod", ("r1",), refresh_token=2)
    assert metrics["annualized_return"][0] == pytest.approx(0.2)
    nav = backtest_repo.get_nav_by_runs("prod", ("r1",), refresh_token=3)
    assert nav.height == 2
    assert nav["ret_1d"].to_list() == pytest.approx([0.1, 0.1])
    benchmark = backtest_repo.get_benchmark_nav("prod", ("BM",), refresh_token=4)
    assert benchmark["benchmark_id"].unique().to_list() == ["BM"]


def test_repo_excess_fallback_and_missing_rank_ic(db: Path) -> None:
    excess = backtest_repo.get_excess_nav("prod", "r1", "BM", refresh_token=5)
    assert excess.source == "fallback"
    assert excess.frame.is_empty()
    ic = backtest_repo.get_rank_ic_series("prod", ("r1",), refresh_token=6)
    assert ic.is_empty()
    assert set(ic.columns) == {"run_id", "trade_date", "rank_ic"}

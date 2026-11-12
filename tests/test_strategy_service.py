"""策略服务：日期对齐、归一、超额回退和主基准隔离。"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_analytics.db import DataAccessError
from quant_analytics.repositories.backtest_repo import ExcessNav
from quant_analytics.services import strategy_service


def _nav() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["r1"] * 3,
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "nav": [1.0, 1.05, 1.10],
            "ret_1d": [0.0, 0.05, 0.0476],
        }
    )


def _bench() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "benchmark_id": ["BM_MAIN"] * 3 + ["BM_OTHER"] * 3,
            "trade_date": [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 5),
            ]
            * 2,
            "nav": [1.0, 1.02, 1.03, 1.0, 1.01, 1.02],
            "ret_1d": [0.0, 0.02, 0.01, 0.0, 0.01, 0.01],
        }
    )


def _patch(monkeypatch, *, excess_source: str = "fallback") -> None:
    monkeypatch.setattr(
        strategy_service,
        "get_run",
        lambda env, run_id, refresh_token=0: {
            "run_id": run_id,
            "strategy_id": "demo",
            "strategy_name": "Demo",
            "benchmark_id": "BM_MAIN",
            "status": "success",
        },
    )
    monkeypatch.setattr(
        strategy_service,
        "get_nav_by_runs",
        lambda *args, **kwargs: _nav(),
    )
    monkeypatch.setattr(
        strategy_service,
        "get_benchmark_nav",
        lambda *args, **kwargs: _bench(),
    )
    monkeypatch.setattr(
        strategy_service,
        "get_benchmark_defs",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "benchmark_id": ["BM_MAIN", "BM_OTHER"],
                "name": ["主基准", "对比基准"],
            }
        ),
    )
    monkeypatch.setattr(
        strategy_service,
        "get_run_metrics",
        lambda *args, **kwargs: pl.DataFrame(
            {"run_id": ["r1"], "annualized_return": [0.10], "sharpe_ratio": [1.2]}
        ),
    )
    if excess_source == "view":
        frame = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
                "excess_nav": [1.0, 1.02],
                "excess_ret_1d": [0.0, 0.02],
            }
        )
        monkeypatch.setattr(
            strategy_service,
            "get_excess_nav",
            lambda *args, **kwargs: ExcessNav(frame, "view"),
        )
    else:
        monkeypatch.setattr(
            strategy_service,
            "get_excess_nav",
            lambda *args, **kwargs: ExcessNav(
                pl.DataFrame(schema={"trade_date": pl.Date()}),
                "fallback",
                "视图不可用",
            ),
        )
    monkeypatch.setattr(
        strategy_service,
        "get_rank_ic_series",
        lambda *args, **kwargs: pl.DataFrame(
            schema={"run_id": pl.Utf8(), "trade_date": pl.Date(), "rank_ic": pl.Float64()}
        ),
    )


def test_strategy_inner_join_normalizes_and_uses_main_only(monkeypatch) -> None:
    _patch(monkeypatch)
    result = strategy_service.build_strategy_comparison(
        "prod", "r1", "BM_MAIN", compare_benchmark_ids=("BM_OTHER",)
    )
    assert result.nav.height == 2  # 1/5 没有策略，内连接丢弃
    assert result.nav["strategy_nav"][0] == pytest.approx(1.0)
    assert result.nav["benchmark_nav"][0] == pytest.approx(1.0)
    assert result.kpis["annualized_return"] == pytest.approx(0.10)
    assert result.ic_kpis is None
    assert "内存回退" in " ".join(result.notes)


def test_strategy_excess_view_source_and_compare_lines(monkeypatch) -> None:
    _patch(monkeypatch, excess_source="view")
    result = strategy_service.build_strategy_comparison(
        "prod", "r1", "BM_MAIN", compare_benchmark_ids=("BM_OTHER",)
    )
    assert "v_bt_nav_excess" in " ".join(result.notes)
    assert set(result.benchmark_lines["benchmark_id"].to_list()) == {"BM_MAIN", "BM_OTHER"}
    assert result.excess["excess_nav"].to_list() == pytest.approx([1.0, 1.02])


def test_strategy_missing_run_raises(monkeypatch) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(strategy_service, "get_run", lambda *args, **kwargs: None)
    with pytest.raises(DataAccessError):
        strategy_service.build_strategy_comparison("prod", "missing", "BM_MAIN")


def test_strategy_without_benchmark_has_no_excess(monkeypatch) -> None:
    _patch(monkeypatch)
    result = strategy_service.build_strategy_comparison("prod", "r1", None)
    assert result.excess.is_empty()
    assert result.kpis.get("excess_total_return") is None
    assert any("未选择主基准" in note for note in result.notes)

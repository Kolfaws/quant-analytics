"""tests.test_metrics — 因子/绩效指标的 Polars 实现基础用例。"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_analytics.metrics import (
    calc_cumulative_returns,
    calc_daily_ic,
    calc_drawdown,
    calc_ic_summary,
    calc_performance_metrics,
    calc_quantile_returns,
    calc_turnover,
)


def _factor_frame(rows: list[tuple[date, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "trade_date": pl.Date,
            "ts_code": pl.Utf8,
            "factor_value": pl.Float64,
            "forward_return": pl.Float64,
        },
        orient="row",
    )


def test_cumulative_returns_and_drawdown() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "strategy_nav": [1.0, 1.1, 1.21],
            "baseline_nav": [1.0, 1.05, 1.1025],
        }
    )
    result = calc_cumulative_returns(frame)
    assert result["strategy_cum"].to_list() == pytest.approx([1.0, 1.1, 1.21])
    assert result["baseline_cum"].to_list() == pytest.approx([1.0, 1.05, 1.1025])
    assert result["excess_cum"].to_list() == pytest.approx([1.0, 1.1 / 1.05, 1.21 / 1.1025])

    with_dd = calc_drawdown(result, "strategy_nav")
    assert with_dd["drawdown"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_performance_metrics() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "strategy_ret": [0.0, 0.10, -0.05],
            "baseline_ret": [0.0, 0.05, 0.0],
            "strategy_nav": [1.0, 1.10, 1.045],
            "baseline_nav": [1.0, 1.05, 1.05],
        }
    )
    metrics = calc_performance_metrics(frame)
    assert metrics["strategy_annual_return"] is not None
    assert metrics["baseline_annual_return"] is not None
    assert metrics["sharpe_ratio"] is not None
    assert metrics["max_drawdown"] == pytest.approx(1.045 / 1.10 - 1.0)
    assert metrics["beta"] == pytest.approx(2.5, abs=1e-6)


def test_daily_ic_perfect_monotonic() -> None:
    rows = []
    for day in (date(2024, 1, 2), date(2024, 1, 3)):
        for index in range(10):
            rows.append((day, f"{index:06d}.SZ", float(index), float(index) / 1000))
    result = calc_daily_ic(_factor_frame(rows))
    assert result.height == 2
    assert result["ic"].to_list() == pytest.approx([1.0, 1.0])
    assert result["rank_ic"].to_list() == pytest.approx([1.0, 1.0])
    summary = calc_ic_summary(result)
    assert summary["ic_mean"] == pytest.approx(1.0)
    assert summary["ic_win_rate"] == pytest.approx(1.0)


def test_quantile_returns_top_group_wins() -> None:
    rows = []
    for day in (date(2024, 1, 2), date(2024, 1, 3)):
        for index in range(10):
            rows.append((day, f"{index:06d}.SZ", float(index), float(index) / 1000))
    result = calc_quantile_returns(_factor_frame(rows), n_groups=5)
    assert set(result["group"].unique().to_list()) == {1, 2, 3, 4, 5}
    latest = result.filter(pl.col("trade_date") == date(2024, 1, 3))
    top = latest.filter(pl.col("group") == 1)["group_ret"][0]
    bottom = latest.filter(pl.col("group") == 5)["group_ret"][0]
    assert top > bottom
    ls = latest["long_short_ret"][0]
    assert ls == pytest.approx(top - bottom)
    assert latest["long_short_cum"][0] > 1.0


def test_turnover_full_rebalance() -> None:
    rows = []
    # 第 1 天 Top 2 = C/D；第 2 天 Top 2 = A/B，完全换手
    for day, values in (
        (date(2024, 1, 2), {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}),
        (date(2024, 1, 3), {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    ):
        for code, factor in values.items():
            rows.append((day, code, factor, 0.01))
    result = calc_turnover(_factor_frame(rows), n_groups=2)
    assert result["turnover"].to_list() == pytest.approx([0.0, 1.0])


def test_metrics_is_vectorized_on_empty() -> None:
    empty = _factor_frame([])
    assert calc_daily_ic(empty).is_empty()
    assert calc_quantile_returns(empty).is_empty()
    assert calc_turnover(empty).is_empty()

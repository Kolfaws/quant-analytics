"""tests.test_metrics — 因子/绩效指标的 Polars 实现基础用例。"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_analytics.metrics import (
    annual_return,
    annual_vol,
    beta,
    calc_cumulative_returns,
    calc_daily_ic,
    calc_drawdown,
    calc_ic_summary,
    calc_performance_metrics,
    calc_quantile_returns,
    calc_turnover,
    calmar,
    cum_nav,
    drawdown,
    excess_nav,
    information_ratio,
    max_drawdown,
    rank_ic_summary,
    sharpe,
    tracking_error,
    win_rate,
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


def test_new_nav_and_scalar_metrics() -> None:
    returns = pl.Series("ret_1d", [0.01, -0.02, 0.03, 0.0])
    frame = pl.DataFrame({"trade_date": [date(2024, 1, 2 + i) for i in range(4)]}).with_columns(
        returns.alias("ret_1d")
    )
    nav = cum_nav(frame)
    assert nav["cum_nav"].to_list() == pytest.approx([1.01, 0.9898, 1.019494, 1.019494])
    assert annual_return(returns) is not None
    assert annual_vol(returns) is not None
    assert sharpe(returns) is not None
    assert max_drawdown(nav["cum_nav"]) == pytest.approx(0.9898 / 1.01 - 1.0)
    assert calmar(returns) is not None
    assert win_rate(returns) == pytest.approx(0.5)
    assert information_ratio(returns) is not None
    assert tracking_error(returns) == pytest.approx(annual_vol(returns))


def test_excess_nav_and_drawdown_alignment() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "strategy_nav": [1.0, 1.1, 1.2],
            "benchmark_nav": [1.0, 1.05, 1.08],
        }
    )
    result = excess_nav(frame)
    assert result["excess_nav"].to_list() == pytest.approx([1.0, 1.1 / 1.05, 1.2 / 1.08])
    with_dd = drawdown(frame, nav_col="strategy_nav")
    assert with_dd["drawdown"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_metrics_empty_single_and_zero_variance() -> None:
    empty = pl.Series("ret", [], dtype=pl.Float64)
    assert annual_return(empty) is None
    assert annual_vol(empty) is None
    assert sharpe(empty) is None
    assert max_drawdown(empty) is None
    assert calmar(empty) is None
    assert win_rate(empty) is None
    assert information_ratio(empty) is None
    assert beta(empty, empty) is None

    one = pl.Series("ret", [0.01])
    assert annual_vol(one) is None
    assert sharpe(one) is None
    assert annual_return(one) == pytest.approx(1.01**252 - 1.0)

    zero = pl.Series("ret", [0.0, 0.0, 0.0])
    assert sharpe(zero) is None
    assert beta(zero, zero) is None


def test_rank_ic_summary_and_empty() -> None:
    ic = pl.DataFrame({"rank_ic": [0.1, -0.05, 0.2]})
    summary = rank_ic_summary(ic)
    assert summary["rank_ic_mean"] == pytest.approx(0.25 / 3)
    assert summary["rank_ic_win_rate"] == pytest.approx(2 / 3)
    assert summary["n_days"] == 3
    empty = pl.DataFrame(schema={"rank_ic": pl.Float64()})
    assert rank_ic_summary(empty)["n_days"] == 0

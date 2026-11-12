"""策略图表：不下库、主题和点数上限。"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from quant_analytics.charts import (
    MAX_CHART_POINTS,
    drawdown_chart,
    multi_run_drawdown_chart,
    multi_run_nav_chart,
    rank_ic_cum_chart,
    strategy_benchmark_nav_chart,
)


def _long_frame(days: int = 6000) -> pl.DataFrame:
    start = date(2010, 1, 1)
    return pl.DataFrame(
        {
            "trade_date": [start + timedelta(days=i) for i in range(days)],
            "strategy_nav": [1.0 + i * 1e-5 for i in range(days)],
            "benchmark_nav": [1.0 + i * 5e-6 for i in range(days)],
        }
    )


def test_charts_are_plotly_gl_and_downsampled() -> None:
    frame = _long_frame()
    wide = frame.select(
        "trade_date",
        pl.col("strategy_nav").alias("r1"),
        pl.col("benchmark_nav").alias("r2"),
    )
    fig = multi_run_nav_chart(wide, dark=True)
    assert all(trace.type == "scattergl" for trace in fig.data)
    assert all(len(trace.x) <= MAX_CHART_POINTS for trace in fig.data)
    assert fig.layout.paper_bgcolor == "#0E1117"

    strategy_fig = strategy_benchmark_nav_chart(frame, dark=False)
    assert strategy_fig.layout.paper_bgcolor == "#FFFFFF"

    dd = frame.with_columns((pl.col("strategy_nav") / 1.01).alias("drawdown"))
    assert drawdown_chart(dd).data
    wide_dd = dd.select("trade_date", pl.col("drawdown").alias("r1"))
    assert multi_run_drawdown_chart(wide_dd).data
    ic = pl.DataFrame({"trade_date": [date(2024, 1, 2)], "rank_ic_cum": [0.1]})
    assert rank_ic_cum_chart(ic).data

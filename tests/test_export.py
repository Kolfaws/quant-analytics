"""HTML 快照导出：结构、离线依赖和敏感信息检查。"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from quant_analytics.export import render_strategy_html


def test_export_contains_snapshot_and_no_sensitive_info() -> None:
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1.0, 1.1], name="策略"))
    table = pl.DataFrame({"指标": ["年化", "回撤"], "值": [0.1, -0.05]})
    rendered = render_strategy_html(
        title="策略快照",
        env="test",
        run_id="run-1",
        benchmark_id="BI_TEST",
        compare_benchmark_ids=("BI_OTHER",),
        kpis={"annualized_return": 0.1, "max_drawdown": -0.05},
        figures=[fig],
        table=table,
        notes=["超额来源：测试"],
    )
    assert "策略快照" in rendered
    assert "run-1" in rendered
    assert "BI_TEST" in rendered
    assert "plotly-2.35.2.min.js" in rendered
    assert "dataTables" in rendered
    assert "snapshot-table" in rendered
    assert "annualized_return" not in rendered or "年化收益" in rendered
    assert ".duckdb" not in rendered
    assert "BACKTEST_DB_PATH" not in rendered
    assert "SELECT " not in rendered.upper()
    assert "年化收益" in rendered


def test_export_empty_table_is_safe() -> None:
    rendered = render_strategy_html(
        title="空",
        env="prod",
        run_id="none",
        benchmark_id=None,
        figures=[],
        table=pl.DataFrame(),
    )
    assert "无数据" in rendered
    assert "none" in rendered

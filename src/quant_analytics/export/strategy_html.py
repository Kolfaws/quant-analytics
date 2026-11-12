"""strategy_html — 当前筛选快照的独立 HTML 报告。

只接收页面已经算好的 KPI、图表和表格；不连接数据库、不写 SQL、不包含
数据库路径。Plotly 和 DataTables 使用 CDN。
"""

from __future__ import annotations

import html as html_lib
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import plotly.graph_objects as go
import polars as pl


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, date | datetime):
        return str(value)
    return html_lib.escape(str(value))


def _kpi_cards(kpis: Mapping[str, Any]) -> str:
    labels = {
        "total_return": "总收益",
        "annualized_return": "年化收益",
        "annualized_volatility": "年化波动",
        "sharpe_ratio": "Sharpe",
        "max_drawdown": "最大回撤",
        "calmar_ratio": "Calmar",
        "win_rate": "胜率",
        "excess_total_return": "超额总收益",
        "tracking_error": "跟踪误差",
        "information_ratio": "IR",
        "beta": "Beta",
        "run_count": "run 数",
        "mean_annualized_return": "平均年化",
        "mean_sharpe_ratio": "平均 Sharpe",
        "mean_max_drawdown": "平均最大回撤",
    }
    cards = []
    for key, label in labels.items():
        if key in kpis:
            cards.append(
                f'<div class="kpi"><span>{html_lib.escape(label)}</span>'
                f"<strong>{_fmt(kpis[key])}</strong></div>"
            )
    return '<section class="kpis">' + "".join(cards) + "</section>"


def _table_html(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return '<p class="muted">无数据</p>'
    columns = frame.columns
    head = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in columns)
    rows = []
    for row in frame.iter_rows(named=True):
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in columns)
        rows.append(f"<tr>{cells}</tr>")
    return (
        '<table id="snapshot-table" class="display compact">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_strategy_html(
    *,
    title: str,
    env: str,
    run_id: str,
    benchmark_id: str | None,
    compare_benchmark_ids: Sequence[str] = (),
    start: date | None = None,
    end: date | None = None,
    generated_at: datetime | None = None,
    kpis: Mapping[str, Any] | None = None,
    figures: Sequence[go.Figure] = (),
    table: pl.DataFrame | None = None,
    notes: Sequence[str] = (),
    dark: bool = False,
) -> str:
    """生成离线可打开的筛选快照 HTML；不包含连接串或数据库路径。"""
    generated = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    period = f"{start or '—'} ~ {end or '—'}"
    filters = [
        ("环境", "测试" if env == "test" else "正式"),
        ("run", run_id),
        ("主基准", benchmark_id or "—"),
        ("对比基准", ", ".join(compare_benchmark_ids) or "—"),
        ("区间", period),
        ("生成时间", generated),
    ]
    filter_html = "".join(
        f"<div><span>{html_lib.escape(k)}</span><strong>{html_lib.escape(v)}</strong></div>"
        for k, v in filters
    )
    charts = []
    for index, fig in enumerate(figures):
        charts.append(
            '<section class="chart">'
            f'<div class="chart-title">图 {index + 1}</div>'
            + fig.to_html(
                include_plotlyjs=False,
                full_html=False,
                config={
                    "displayModeBar": False,
                    "displaylogo": False,
                    "responsive": True,
                },
            )
            + "</section>"
        )
    notes_html = ""
    if notes:
        notes_html = (
            '<ul class="notes">'
            + "".join(f"<li>{html_lib.escape(str(note))}</li>" for note in notes)
            + "</ul>"
        )
    bg = "#0B0F17" if dark else "#F8FAFC"
    card = "#111827" if dark else "#FFFFFF"
    text = "#E5E7EB" if dark else "#111827"
    muted = "#94A3B8" if dark else "#64748B"
    border = "#253041" if dark else "#E2E8F0"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/2.1.8/css/dataTables.dataTables.min.css">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/2.1.8/js/dataTables.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: {bg}; color: {text};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 32px 22px 48px; }}
  h1 {{ font-size: 24px; margin: 0 0 6px; }}
  .meta {{ color: {muted}; font-size: 12px; margin-bottom: 18px; }}
  .filters {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px; background: {card}; border: 1px solid {border}; border-radius: 8px;
    padding: 14px; margin-bottom: 18px; }}
  .filters span {{ display: block; color: {muted}; font-size: 11px; }}
  .filters strong {{ display: block; margin-top: 3px; font-size: 13px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px; margin-bottom: 18px; }}
  .kpi {{ background: {card}; border: 1px solid {border}; border-radius: 8px; padding: 12px; }}
  .kpi span {{ display: block; color: {muted}; font-size: 11px; }}
  .kpi strong {{ display: block; margin-top: 4px; font-size: 18px; }}
  .chart {{ background: {card}; border: 1px solid {border}; border-radius: 8px;
    padding: 8px; margin-bottom: 14px; }}
  .chart-title {{ color: {muted}; font-size: 12px; padding: 4px 6px 0; }}
  table.display {{ width: 100% !important; }}
  .notes {{ color: {muted}; font-size: 12px; }}
  .muted {{ color: {muted}; }}
</style>
</head>
<body>
<main>
  <h1>{html_lib.escape(title)}</h1>
  <div class="meta">筛选快照；仅含当前页面已计算的数据，不含数据库连接或 SQL。</div>
  <section class="filters">{filter_html}</section>
  {_kpi_cards(kpis or {})}
  {"".join(charts)}
  <section class="chart">
    <div class="chart-title">绩效表</div>
    {_table_html(table) if table is not None else '<p class="muted">无数据</p>'}
  </section>
  {notes_html}
</main>
<script>
$(function () {{
  if ($.fn.DataTable) {{
    $('#snapshot-table').DataTable({{ pageLength: 15, lengthMenu: [10, 15, 25, 50] }});
  }}
}});
</script>
</body>
</html>"""


__all__ = ["render_strategy_html"]

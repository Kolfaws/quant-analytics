"""charts — Plotly 图表封装。

所有计算在 Polars 完成；仅在交给 Plotly 前短暂 ``to_pandas()``。
图表配色跟随 Streamlit 当前 Light/Dark 主题；导出的独立 HTML 使用同一主题。
"""

from __future__ import annotations

import html as html_lib

import plotly.graph_objects as go
import polars as pl

_COLORS = {
    "strategy": "#2563EB",
    "baseline": "#94A3B8",
    "excess": "#F59E0B",
    "top": "#16A34A",
    "long_short": "#DC2626",
    "turnover": "#7C3AED",
}

_LIGHT = {
    "paper": "#FFFFFF",
    "plot": "#FFFFFF",
    "font": "#111827",
    "grid": "#E5E7EB",
    "hover": "#FFFFFF",
    "zero": "#94A3B8",
}
_DARK = {
    "paper": "#0E1117",
    "plot": "#0E1117",
    "font": "#E5E7EB",
    "grid": "#2B313E",
    "hover": "#1F2937",
    "zero": "#64748B",
}


def _palette(dark: bool) -> dict[str, str]:
    return _DARK if dark else _LIGHT


def _apply_theme(fig: go.Figure, dark: bool = False) -> go.Figure:
    palette = _palette(dark)
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=palette["paper"],
        plot_bgcolor=palette["plot"],
        font=dict(color=palette["font"], size=12),
        hoverlabel=dict(
            bgcolor=palette["hover"],
            font=dict(color=palette["font"]),
            bordercolor=palette["grid"],
        ),
    )
    fig.update_xaxes(gridcolor=palette["grid"], zerolinecolor=palette["zero"])
    fig.update_yaxes(gridcolor=palette["grid"], zerolinecolor=palette["zero"])
    return fig


def _layout(fig: go.Figure, *, title: str, y_title: str = "", dark: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        hovermode="x unified",
        margin=dict(l=12, r=12, t=56, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
        height=420,
    )
    fig.update_xaxes(title_text="")
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return _apply_theme(fig, dark)


def plot_equity_curves(
    df: pl.DataFrame, *, dark: bool = False, show_excess: bool = True
) -> go.Figure:
    """策略 / 基准 / 超额累计净值。"""
    data = df.sort("trade_date").to_pandas()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["strategy_cum"],
            name="策略",
            mode="lines",
            line=dict(color=_COLORS["strategy"], width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["baseline_cum"],
            name="基准",
            mode="lines",
            line=dict(color=_COLORS["baseline"], width=1.8, dash="dot"),
        )
    )
    if show_excess:
        fig.add_trace(
            go.Scatter(
                x=data["trade_date"],
                y=data["excess_cum"],
                name="超额（策略/基准）",
                mode="lines",
                line=dict(color=_COLORS["excess"], width=1.6),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(title="超额净值", overlaying="y", side="right", showgrid=False)
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color=_palette(dark)["zero"], line_width=1)
    title = "累计净值与超额" if show_excess else "累计净值"
    return _layout(fig, title=title, y_title="累计净值", dark=dark)


def plot_drawdown(df: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """策略回撤序列。"""
    data = df.sort("trade_date").to_pandas()
    fig = go.Figure(
        go.Scatter(
            x=data["trade_date"],
            y=data["drawdown"],
            name="回撤",
            mode="lines",
            line=dict(color=_COLORS["long_short"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.12)",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title="策略回撤", y_title="回撤", dark=dark)


def plot_ic_bars(ic_df: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """每日 Rank IC 柱状图。"""
    data = ic_df.sort("trade_date").to_pandas()
    colors = [
        _COLORS["top"] if value and value > 0 else _COLORS["long_short"]
        for value in data["rank_ic"]
    ]
    fig = go.Figure(
        go.Bar(
            x=data["trade_date"],
            y=data["rank_ic"],
            name="Rank IC",
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d}<br>Rank IC=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color=_palette(dark)["zero"], line_width=1)
    return _layout(fig, title="每日 Rank IC", y_title="IC", dark=dark)


def plot_cumulative_ic(ic_df: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """IC 累积曲线。"""
    data = (
        ic_df.sort("trade_date")
        .with_columns(
            pl.col("rank_ic").fill_null(0.0).cum_sum().alias("rank_ic_cum"),
            pl.col("ic").fill_null(0.0).cum_sum().alias("ic_cum"),
        )
        .to_pandas()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["rank_ic_cum"],
            name="Rank IC 累积",
            mode="lines",
            line=dict(color=_COLORS["strategy"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["ic_cum"],
            name="Pearson IC 累积",
            mode="lines",
            line=dict(color=_COLORS["baseline"], width=1.5, dash="dot"),
        )
    )
    return _layout(fig, title="IC 累积曲线", y_title="累积 IC", dark=dark)


def plot_group_cumulative(daily: pl.DataFrame, n_groups: int, *, dark: bool = False) -> go.Figure:
    """各分组累计净值（G1 为因子值最高组）。"""
    fig = go.Figure()
    groups = daily["group"].unique().sort().to_list()
    for group in groups:
        part = daily.filter(pl.col("group") == group).sort("trade_date").to_pandas()
        hue = int(210 - 160 * (int(group) - 1) / max(n_groups - 1, 1))
        color = f"hsl({hue}, 70%, {'62%' if dark else '50%'})"
        label = f"G{group}" + (
            "（Top）" if group == 1 else "（Bottom）" if group == n_groups else ""
        )
        fig.add_trace(
            go.Scatter(
                x=part["trade_date"],
                y=part["group_cum"],
                name=label,
                mode="lines",
                line=dict(color=color, width=1.8 if group in (1, n_groups) else 1.2),
            )
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color=_palette(dark)["zero"], line_width=1)
    return _layout(fig, title="分组累计净值", y_title="净值", dark=dark)


def plot_long_short(daily: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """多空对冲（Top - Bottom）与仅多头（Top Only）净值。"""
    data = (
        daily.select("trade_date", "top_only_cum", "long_short_cum")
        .unique(subset=["trade_date"], keep="first")
        .sort("trade_date")
        .to_pandas()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["top_only_cum"],
            name="Top Only",
            mode="lines",
            line=dict(color=_COLORS["top"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["long_short_cum"],
            name="Long - Short",
            mode="lines",
            line=dict(color=_COLORS["long_short"], width=2),
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=_palette(dark)["zero"], line_width=1)
    return _layout(fig, title="多空 / 仅多头净值", y_title="净值", dark=dark)


def plot_turnover(turnover: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """Top 组日度换手率 + 20 日均线。"""
    data = (
        turnover.sort("trade_date")
        .with_columns(pl.col("turnover").rolling_mean(20).alias("turnover_ma20"))
        .to_pandas()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["turnover"],
            name="日换手率",
            mode="lines",
            line=dict(color=_COLORS["turnover"], width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["trade_date"],
            y=data["turnover_ma20"],
            name="20 日均值",
            mode="lines",
            line=dict(color=_palette(dark)["font"], width=1.6, dash="dot"),
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title="Top 组换手率", y_title="换手率", dark=dark)


def plot_ic_distribution(ic_df: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """Rank IC 分布直方图（辅助洞察）。"""
    data = ic_df.select(pl.col("rank_ic").drop_nulls()).to_pandas()
    fig = go.Figure(
        go.Histogram(
            x=data["rank_ic"],
            nbinsx=40,
            marker_color=_COLORS["strategy"],
            opacity=0.8,
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color=_palette(dark)["zero"])
    return _layout(fig, title="Rank IC 分布", y_title="天数", dark=dark)


def figure_to_html(fig: go.Figure, *, dark: bool = False, subtitle: str = "") -> str:
    """导出独立、简洁的 HTML 页面。

    只保留标题、可选副标题和图表本身；隐藏 Plotly 工具栏和默认页面框架。
    """
    export = go.Figure(fig)
    _apply_theme(export, dark)
    title_obj = getattr(export.layout, "title", None)
    title = str(getattr(title_obj, "text", None) or "Quant Analytics").strip()
    export.update_layout(title_text="", margin=dict(l=12, r=12, t=16, b=12))
    title_html = html_lib.escape(title)
    subtitle_html = f'<p class="subtitle">{html_lib.escape(subtitle)}</p>' if subtitle else ""
    chart_html = export.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
    )

    if dark:
        page_bg, card_bg, text, muted, border = (
            "#0B0F17",
            "#111827",
            "#E5E7EB",
            "#94A3B8",
            "#253041",
        )
    else:
        page_bg, card_bg, text, muted, border = (
            "#F8FAFC",
            "#FFFFFF",
            "#111827",
            "#64748B",
            "#E2E8F0",
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="{"dark" if dark else "light"}">
<title>{title_html}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: {page_bg};
    color: {text};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  main {{ max-width: 1120px; margin: 0 auto; padding: 38px 24px 48px; }}
  header {{ border-bottom: 1px solid {border}; padding-bottom: 14px; margin-bottom: 20px; }}
  h1 {{ margin: 0; font-size: 22px; font-weight: 600; letter-spacing: 0; }}
  .subtitle {{ margin: 7px 0 0; color: {muted}; font-size: 13px; }}
  .chart {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 8px 8px 0;
    overflow: hidden;
  }}
  .chart .plotly-graph-div {{ width: 100% !important; }}
</style>
</head>
<body>
<main>
  <header>
    <h1>{title_html}</h1>
    {subtitle_html}
  </header>
  <section class="chart">
    {chart_html}
  </section>
</main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 新服务层页面用的批量/多 run 图
# ---------------------------------------------------------------------------
MAX_CHART_POINTS = 5000


def _downsample(df: pl.DataFrame, max_points: int = MAX_CHART_POINTS) -> pl.DataFrame:
    """按步长降采样，不逐行循环；保留首尾附近趋势。"""
    if df.height <= max_points:
        return df
    step = max(1, int(df.height / max_points) + 1)
    return (
        df.with_row_index("_i")
        .filter((pl.col("_i") % step == 0) | (pl.col("_i") == df.height - 1))
        .drop("_i")
    )


def _scatter_line(
    frame: pl.DataFrame,
    *,
    x: str,
    y: str,
    name: str,
    color: str,
    width: float = 1.8,
    dash: str | None = None,
    max_points: int = MAX_CHART_POINTS,
) -> go.Scattergl:
    data = _downsample(frame.select(x, y).drop_nulls(), max_points).to_pandas()
    return go.Scattergl(
        x=data[x],
        y=data[y],
        name=name,
        mode="lines",
        line=dict(color=color, width=width, dash=dash),
    )


def strategy_benchmark_nav_chart(
    nav: pl.DataFrame,
    *,
    excess: pl.DataFrame | None = None,
    benchmark_lines: pl.DataFrame | None = None,
    dark: bool = False,
) -> go.Figure:
    """主策略 + 主基准 + 可选超额 + 可选对比基准线。"""
    fig = go.Figure()
    if "strategy_nav" in nav.columns:
        fig.add_trace(
            _scatter_line(
                nav,
                x="trade_date",
                y="strategy_nav",
                name="策略",
                color=_COLORS["strategy"],
                width=2.2,
            )
        )
    if "benchmark_nav" in nav.columns:
        fig.add_trace(
            _scatter_line(
                nav,
                x="trade_date",
                y="benchmark_nav",
                name="主基准",
                color=_COLORS["baseline"],
                width=1.8,
                dash="dot",
            )
        )
    if excess is not None and not excess.is_empty():
        data = _downsample(excess.select("trade_date", "excess_nav").drop_nulls()).to_pandas()
        fig.add_trace(
            go.Scattergl(
                x=data["trade_date"],
                y=data["excess_nav"],
                name="超额（策略/主基准）",
                mode="lines",
                line=dict(color=_COLORS["excess"], width=1.5),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(title="超额净值", overlaying="y", side="right", showgrid=False)
        )
    if benchmark_lines is not None and not benchmark_lines.is_empty():
        for row in benchmark_lines["benchmark_id"].unique(maintain_order=True).to_list():
            part = benchmark_lines.filter(pl.col("benchmark_id") == row)
            label = str(part["name"][0]) if "name" in part.columns else str(row)
            fig.add_trace(
                _scatter_line(
                    part,
                    x="trade_date",
                    y="nav",
                    name=f"对比 · {label}",
                    color=_COLORS["turnover"],
                    width=1.0,
                    dash="dash",
                )
            )
    fig.add_hline(y=1.0, line_dash="dash", line_color=_palette(dark)["zero"], line_width=1)
    return _layout(fig, title="策略、基准与对比基准", y_title="归一净值", dark=dark)


def drawdown_chart(drawdown_frame: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """策略/基准回撤；缺失基准时只画策略。"""
    fig = go.Figure()
    fig.add_trace(
        _scatter_line(
            drawdown_frame,
            x="trade_date",
            y="drawdown",
            name="策略回撤",
            color=_COLORS["long_short"],
            width=1.5,
        )
    )
    if "benchmark_drawdown" in drawdown_frame.columns:
        fig.add_trace(
            _scatter_line(
                drawdown_frame,
                x="trade_date",
                y="benchmark_drawdown",
                name="基准回撤",
                color=_COLORS["baseline"],
                width=1.2,
                dash="dot",
            )
        )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title="回撤", y_title="回撤", dark=dark)


def rank_ic_cum_chart(ic_cum: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """RankIC 累积曲线；空表显示空图。"""
    fig = go.Figure()
    if not ic_cum.is_empty():
        fig.add_trace(
            _scatter_line(
                ic_cum,
                x="trade_date",
                y="rank_ic_cum",
                name="RankIC 累积",
                color=_COLORS["strategy"],
                width=2.0,
            )
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color=_palette(dark)["zero"])
    return _layout(fig, title="RankIC 累积", y_title="累积 RankIC", dark=dark)


def multi_run_nav_chart(
    nav_wide: pl.DataFrame,
    *,
    dark: bool = False,
    title: str = "多 run 归一净值",
    y_title: str = "净值",
) -> go.Figure:
    """多 run 归一净值；第一列为 trade_date，其余列是 run_id。"""
    fig = go.Figure()
    columns = [name for name in nav_wide.columns if name != "trade_date"]
    per_trace = max(200, MAX_CHART_POINTS // max(1, len(columns)))
    for index, column in enumerate(columns):
        fig.add_trace(
            _scatter_line(
                nav_wide,
                x="trade_date",
                y=column,
                name=str(column),
                color=f"hsl({210 - 24 * index}, 70%, {'62%' if dark else '45%'})",
                width=1.8,
                max_points=per_trace,
            )
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color=_palette(dark)["zero"])
    return _layout(fig, title=title, y_title=y_title, dark=dark)


def multi_run_drawdown_chart(drawdown_wide: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """多 run 回撤。"""
    fig = go.Figure()
    columns = [name for name in drawdown_wide.columns if name != "trade_date"]
    per_trace = max(200, MAX_CHART_POINTS // max(1, len(columns)))
    for index, column in enumerate(columns):
        fig.add_trace(
            _scatter_line(
                drawdown_wide,
                x="trade_date",
                y=column,
                name=str(column),
                color=f"hsl({8 + 24 * index}, 70%, {'62%' if dark else '45%'})",
                width=1.4,
                max_points=per_trace,
            )
        )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title="多 run 回撤", y_title="回撤", dark=dark)


def rank_ic_comparison_chart(ic_df: pl.DataFrame, *, dark: bool = False) -> go.Figure:
    """多 run 的平均 RankIC 对比条形图。"""
    fig = go.Figure()
    if not ic_df.is_empty():
        summary = (
            ic_df.group_by("run_id")
            .agg(pl.col("rank_ic").mean().alias("rank_ic_mean"))
            .sort("run_id")
        )
        data = summary.to_pandas()
        fig.add_trace(
            go.Bar(
                x=data["run_id"],
                y=data["rank_ic_mean"],
                name="平均 RankIC",
                marker_color=_COLORS["strategy"],
            )
        )
    fig.add_hline(y=0.0, line_color=_palette(dark)["zero"], line_width=1)
    return _layout(fig, title="平均 RankIC 对比", y_title="RankIC", dark=dark)


__all__ = [
    "MAX_CHART_POINTS",
    "drawdown_chart",
    "figure_to_html",
    "multi_run_drawdown_chart",
    "multi_run_nav_chart",
    "plot_cumulative_ic",
    "plot_drawdown",
    "plot_equity_curves",
    "plot_group_cumulative",
    "plot_ic_bars",
    "plot_ic_distribution",
    "plot_long_short",
    "plot_turnover",
    "rank_ic_comparison_chart",
    "rank_ic_cum_chart",
    "strategy_benchmark_nav_chart",
]

"""策略与基准对比页面。"""

from __future__ import annotations

import streamlit as st
from loguru import logger

from quant_analytics.charts import figure_to_html, plot_drawdown, plot_equity_curves
from quant_analytics.db import DataAccessError, get_available_options, get_nav_data
from quant_analytics.logging_utils import configure_logging
from quant_analytics.metrics import (
    calc_cumulative_returns,
    calc_drawdown,
    calc_performance_metrics,
)
from quant_analytics.theme import is_dark, theme_caption

st.set_page_config(page_title="策略与基准对比", page_icon=":material/monitoring:", layout="wide")
configure_logging()
theme_caption()
dark = is_dark()

st.title("策略与基准对比")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


env_label = st.sidebar.radio("数据环境", ["正式", "测试"], horizontal=True, key="data_env")
env = "test" if env_label == "测试" else "prod"

strategies, baselines, _ = get_available_options(env=env)

with st.sidebar:
    st.header("策略对比")
    if not strategies or not baselines:
        st.error("暂无可用的策略或基准数据。")
        st.stop()
    strategy = st.selectbox(
        "策略",
        strategies,
        format_func=lambda item: item["label"],
    )
    baseline = st.selectbox(
        "基准",
        baselines,
        format_func=lambda item: item["label"],
    )
    show_excess = st.checkbox("显示超额曲线", value=True)
    show_drawdown = st.checkbox("显示回撤图", value=True)
    st.divider()
    if st.button("刷新数据", icon=":material/refresh:", width="stretch"):
        get_nav_data.clear()
        get_available_options.clear()
        st.rerun()
    st.caption("查询缓存 1 小时；刷新后重新读取数据库。")

try:
    with st.spinner("加载数据…"):
        raw = get_nav_data(strategy["run_id"], baseline["benchmark_id"], env=env)
        frame = calc_cumulative_returns(raw)
        frame = calc_drawdown(frame, "strategy_nav")
        metrics = calc_performance_metrics(frame)
except DataAccessError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    logger.exception("策略页面计算失败")
    st.error(f"计算失败：{exc}")
    st.stop()

period = f"{frame['trade_date'].min()} ~ {frame['trade_date'].max()}"
st.caption(f"{period} · {frame.height:,} 个交易日")

kpi = st.columns(5)
kpi[0].metric("策略年化", _pct(metrics["strategy_annual_return"]))
kpi[1].metric("基准年化", _pct(metrics["baseline_annual_return"]))
kpi[2].metric("超额年化", _pct(metrics["excess_annual_return"]))
kpi[3].metric("夏普比率", _num(metrics["sharpe_ratio"]))
kpi[4].metric("最大回撤", _pct(metrics["max_drawdown"]))

equity_fig = plot_equity_curves(frame, dark=dark, show_excess=show_excess)
st.plotly_chart(equity_fig, theme=None, width="stretch")

st.caption(
    f"年化波动 {_pct(metrics['annual_volatility'])} · "
    f"跟踪误差 {_pct(metrics['tracking_error'])} · "
    f"信息比率 {_num(metrics['information_ratio'])} · "
    f"Beta {_num(metrics['beta'], 3)}"
)
if show_drawdown:
    st.plotly_chart(plot_drawdown(frame, dark=dark), theme=None, width="stretch")

st.download_button(
    "导出净值图 HTML",
    data=figure_to_html(
        equity_fig,
        dark=dark,
        subtitle=f"{period} · {frame.height:,} 个交易日",
    ),
    file_name=f"strategy_vs_baseline_{strategy['run_id']}.html",
    mime="text/html",
    width="content",
)

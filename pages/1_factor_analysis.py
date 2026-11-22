"""因子深度分析页面"""

from __future__ import annotations

from datetime import date
from typing import Final

import streamlit as st
from loguru import logger

from quant_analytics.charts import (
    figure_to_html,
    plot_cumulative_ic,
    plot_group_cumulative,
    plot_ic_bars,
    plot_long_short,
    plot_turnover,
)
from quant_analytics.db import (
    DataAccessError,
    FactorData,
    db_cache_token,
    get_available_options,
    get_factor_data,
    get_factor_date_range,
)
from quant_analytics.logging_utils import configure_logging
from quant_analytics.services.factor_service import load_factor_analysis
from quant_analytics.theme import is_dark, theme_caption

#: 因子页默认起点：2010 研究库从 2010-01-04 开始，不再硬编码 2015。
DEFAULT_START_DATE: Final[date] = date(2010, 1, 1)

st.set_page_config(page_title="因子深度分析", page_icon=":material/science:", layout="wide")
configure_logging()
theme_caption()
dark = is_dark()

st.title("因子深度分析")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _factor_label(item: dict[str, str]) -> str:
    direction = item.get("direction", "1")
    suffix = " · 负向" if direction == "-1" else ""
    return f"{item['label']}{suffix}"


def _render_sample_notes(data: FactorData) -> None:
    st.caption(f"因子方向：{data.direction_label}")
    for line in data.report.describe():
        st.caption(line)


env_label = st.sidebar.radio("数据环境", ["正式", "测试"], horizontal=True, key="data_env")
env = "test" if env_label == "测试" else "prod"
cache_token = db_cache_token(env) ^ db_cache_token(env, "backtest")

_, _, factors = get_available_options(env=env, cache_token=cache_token)

with st.sidebar:
    st.header("因子分析")
    if not factors:
        st.error("暂无可用因子数据")
        st.stop()
    factor = st.selectbox(
        "因子",
        factors,
        format_func=_factor_label,
    )
    factor_name = factor["factor_name"]
    factor_range = get_factor_date_range(factor_name, env=env, cache_token=cache_token)

    if factor_range:
        min_date, max_date = factor_range
        default_start = max(min_date, DEFAULT_START_DATE)
        default_end = max_date
        if default_start > default_end:
            default_start = min_date
    else:
        min_date, max_date = DEFAULT_START_DATE, date.today()
        default_start, default_end = min_date, max_date

    start_date = st.date_input(
        "开始日期", value=default_start, min_value=min_date, max_value=max_date
    )
    end_date = st.date_input("结束日期", value=default_end, min_value=min_date, max_value=max_date)
    n_groups = st.select_slider("分组数", options=[5, 10], value=5)
    show_ic_curve = st.checkbox("显示 IC 累积曲线", value=True)
    show_turnover = st.checkbox("显示换手率", value=True)
    st.divider()
    if st.button("刷新数据", icon=":material/refresh:", width="stretch"):
        get_factor_data.clear()
        get_factor_date_range.clear()
        get_available_options.clear()
        st.rerun()
    st.caption("查询缓存 1 小时；刷新后重新读取数据库")

if start_date > end_date:
    st.error("开始日期不能晚于结束日期")
    st.stop()

try:
    with st.spinner("计算因子指标…"):
        analysis = load_factor_analysis(
            factor_name,
            start_date.isoformat(),
            end_date.isoformat(),
            n_groups=n_groups,
            env=env,
            cache_token=cache_token,
        )
        factor_data = analysis.factor_data
        ic_df = analysis.ic_df
        ic_summary = analysis.ic_summary
        daily = analysis.daily
        turnover = analysis.turnover
        ls_stats = analysis.ls_stats
except DataAccessError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    logger.exception("因子页面计算失败")
    st.error(f"计算失败：{exc}")
    st.stop()

if ic_df.is_empty():
    st.warning("该区间没有可用的 IC")
    st.stop()

st.caption(f"{start_date} ~ {end_date} · {ic_df.height:,} 个交易日")
_render_sample_notes(factor_data)

kpi = st.columns(6)
kpi[0].metric("Rank IC 均值", _num(ic_summary["rank_ic_mean"], 4))
kpi[1].metric("Rank ICIR", _num(ic_summary["rank_icir"], 3))
kpi[2].metric("IC 胜率", _pct(ic_summary["rank_ic_win_rate"]))
kpi[3].metric("多空夏普", _num(ls_stats["sharpe"]))
kpi[4].metric("多空年化", _pct(ls_stats["annual_return"]))
kpi[5].metric("多空最大回撤", _pct(ls_stats["max_drawdown"]))
st.caption("以上为描述性统计，不构成因子有效性结论；单年/小样本结果只用于流程检查。")

if show_ic_curve:
    ic_left, ic_right = st.columns(2)
    with ic_left:
        st.plotly_chart(plot_ic_bars(ic_df, dark=dark), theme=None, width="stretch")
    with ic_right:
        st.plotly_chart(plot_cumulative_ic(ic_df, dark=dark), theme=None, width="stretch")
else:
    st.plotly_chart(plot_ic_bars(ic_df, dark=dark), theme=None, width="stretch")

group_fig = plot_group_cumulative(daily, n_groups, dark=dark)
st.plotly_chart(group_fig, theme=None, width="stretch")
if show_turnover:
    perf_left, perf_right = st.columns(2)
    with perf_left:
        st.plotly_chart(plot_long_short(daily, dark=dark), theme=None, width="stretch")
    with perf_right:
        st.plotly_chart(plot_turnover(turnover, dark=dark), theme=None, width="stretch")
else:
    st.plotly_chart(plot_long_short(daily, dark=dark), theme=None, width="stretch")

st.download_button(
    "导出分组图 HTML",
    data=figure_to_html(
        group_fig,
        dark=dark,
        subtitle=f"{start_date} ~ {end_date} · {ic_df.height:,} 个交易日",
    ),
    file_name=f"factor_{factor_name}_groups.html",
    mime="text/html",
    width="content",
)

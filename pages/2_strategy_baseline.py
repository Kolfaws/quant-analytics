"""策略与基准对比页面（服务层版本）"""

from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st
from loguru import logger

from quant_analytics.charts import (
    drawdown_chart,
    rank_ic_cum_chart,
    strategy_benchmark_nav_chart,
)
from quant_analytics.db import DataAccessError
from quant_analytics.export import render_strategy_html
from quant_analytics.logging_utils import configure_logging
from quant_analytics.repositories.backtest_repo import (
    get_benchmark_defs,
    list_success_runs,
)
from quant_analytics.services.strategy_service import (
    StrategyComparisonResult,
    build_strategy_comparison,
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


def _refresh() -> None:
    st.session_state["strategy_refresh"] = int(st.session_state.get("strategy_refresh", 0)) + 1


def _env() -> str:
    return "test" if st.session_state.get("strategy_env_label") == "测试" else "prod"


with st.sidebar:
    st.radio(
        "数据环境",
        ["正式", "测试"],
        horizontal=True,
        key="strategy_env_label",
    )
    if st.button("刷新数据", icon=":material/refresh:", width="stretch"):
        _refresh()
        st.rerun()
    st.caption("缓存 1 小时；刷新只影响本页")

env = _env()
token = int(st.session_state.get("strategy_refresh", 0))
st.markdown(
    f":{'orange' if env == 'test' else 'blue'}-badge[环境：{'测试' if env == 'test' else '正式'}]"
)

try:
    runs = list_success_runs(env, refresh_token=token)
    benchmarks = get_benchmark_defs(env, refresh_token=token)
except DataAccessError as exc:
    st.error(str(exc))
    st.stop()

if runs.is_empty():
    st.info("当前环境没有成功 run")
    st.stop()

run_labels = {
    str(row["run_id"]): f"{row['strategy_name'] or row['strategy_id']} · {row['run_id']}"
    for row in runs.iter_rows(named=True)
}
benchmark_labels = {
    str(row["benchmark_id"]): f"{row['name']} ({row['benchmark_id']})"
    for row in benchmarks.iter_rows(named=True)
}
default_run = next(iter(run_labels))
default_benchmark = next(
    (
        str(row["benchmark_id"])
        for row in runs.iter_rows(named=True)
        if row["run_id"] == default_run
    ),
    next(iter(benchmark_labels), None),
)

with st.sidebar, st.form("strategy_filters"):
    run_id = st.selectbox(
        "run",
        options=list(run_labels),
        format_func=lambda value: run_labels[value],
        key="strategy_run_id",
    )
    selected_meta = runs.filter(pl.col("run_id") == run_id).row(0, named=True)
    benchmark_options = list(benchmark_labels)
    if benchmark_options:
        main_benchmark = st.selectbox(
            "主 benchmark",
            options=benchmark_options,
            index=(
                benchmark_options.index(default_benchmark)
                if default_benchmark in benchmark_options
                else 0
            ),
            format_func=lambda value: benchmark_labels[value],
            key="strategy_main_benchmark",
        )
        compare_benchmarks = st.multiselect(
            "对比 benchmark（只画线，不影响主 KPI）",
            options=[item for item in benchmark_options if item != main_benchmark],
            format_func=lambda value: benchmark_labels[value],
            key="strategy_compare_benchmarks",
        )
    else:
        st.caption("没有可用基准，超额指标显示 N/A")
        main_benchmark = None
        compare_benchmarks = []
    min_date = selected_meta.get("start_date") or date(2010, 1, 1)
    max_date = selected_meta.get("end_date") or date.today()
    start = st.date_input(
        "开始日期",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="strategy_start",
    )
    end = st.date_input(
        "结束日期",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="strategy_end",
    )
    show_excess = st.checkbox("显示超额", value=True, key="strategy_show_excess")
    show_drawdown = st.checkbox("显示回撤", value=True, key="strategy_show_drawdown")
    show_ic = st.checkbox("显示 RankIC", value=True, key="strategy_show_ic")
    st.form_submit_button("应用筛选", width="stretch")

if start > end:
    st.error("开始日期不能晚于结束日期")
    st.stop()


@st.fragment
def _render_result(result: StrategyComparisonResult) -> None:
    st.caption(
        f"{result.nav['trade_date'].min()} ~ {result.nav['trade_date'].max()} · "
        f"{result.nav.height:,} 个交易日"
    )
    kpi = st.columns(6)
    kpi[0].metric("总收益", _pct(result.kpis.get("total_return")))
    kpi[1].metric("年化收益", _pct(result.kpis.get("annualized_return")))
    kpi[2].metric("夏普", _num(result.kpis.get("sharpe_ratio")))
    kpi[3].metric("最大回撤", _pct(result.kpis.get("max_drawdown")))
    kpi[4].metric("超额年化", _pct(result.kpis.get("excess_annualized_return")))
    kpi[5].metric("IR", _num(result.kpis.get("information_ratio")))

    ic = result.ic_kpis or {}
    if any(ic.get(key) is not None for key in ("rank_ic_mean", "rank_icir", "rank_ic_win_rate")):
        ic_cols = st.columns(4)
        ic_cols[0].metric("RankIC 均值", _num(ic.get("rank_ic_mean"), 4))
        ic_cols[1].metric("RankICIR", _num(ic.get("rank_icir"), 3))
        ic_cols[2].metric("RankIC 胜率", _pct(ic.get("rank_ic_win_rate")))
        ic_cols[3].metric(
            "IC 天数",
            str(ic.get("n_days")) if ic.get("n_days") is not None else "—",
        )
    else:
        st.caption("RankIC：N/A（无已存 RankIC 结果）")

    comparison_lines = (
        result.benchmark_lines.filter(
            __import__("polars").col("benchmark_id").is_in(compare_benchmarks)
        )
        if compare_benchmarks
        else None
    )
    nav_fig = strategy_benchmark_nav_chart(
        result.nav,
        excess=result.excess if show_excess else None,
        benchmark_lines=comparison_lines,
        dark=dark,
    )
    st.plotly_chart(nav_fig, theme=None, width="stretch")
    if show_drawdown:
        st.plotly_chart(drawdown_chart(result.drawdown, dark=dark), theme=None, width="stretch")
    if show_ic and not result.ic_cum.is_empty():
        st.plotly_chart(rank_ic_cum_chart(result.ic_cum, dark=dark), theme=None, width="stretch")

    table = __import__("polars").DataFrame(
        {
            "指标": list(result.kpis),
            "值": [None if v is None else float(v) for v in result.kpis.values()],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    for note in result.notes:
        st.caption(note)

    figures = [nav_fig]
    if show_drawdown:
        figures.append(drawdown_chart(result.drawdown, dark=dark))
    if show_ic and not result.ic_cum.is_empty():
        figures.append(rank_ic_cum_chart(result.ic_cum, dark=dark))
    st.download_button(
        "导出当前快照 HTML",
        data=render_strategy_html(
            title="策略与基准对比",
            env=env,
            run_id=run_id,
            benchmark_id=main_benchmark,
            compare_benchmark_ids=tuple(compare_benchmarks),
            start=start,
            end=end,
            kpis=result.kpis,
            figures=figures,
            table=table,
            notes=result.notes,
            dark=dark,
        ),
        file_name=f"strategy_{run_id}.html",
        mime="text/html",
        width="content",
    )


try:
    with st.spinner("加载策略与基准…"):
        result = build_strategy_comparison(
            env,
            run_id,
            main_benchmark,
            compare_benchmark_ids=tuple(compare_benchmarks),
            start=start,
            end=end,
            refresh_token=token,
        )
except DataAccessError as exc:
    st.warning(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    logger.exception("策略页面失败")
    st.warning(f"加载失败：{exc}")
    st.stop()

_render_result(result)

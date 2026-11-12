"""页面 3：按因子/模型比较多 run 的策略表现"""

from __future__ import annotations

from datetime import date

import streamlit as st
from loguru import logger

from quant_analytics.charts import (
    multi_run_drawdown_chart,
    multi_run_nav_chart,
    rank_ic_comparison_chart,
)
from quant_analytics.db import DataAccessError
from quant_analytics.export import render_strategy_html
from quant_analytics.logging_utils import configure_logging
from quant_analytics.repositories.backtest_repo import (
    get_benchmark_defs,
    list_success_runs,
)
from quant_analytics.services.factor_strategy_service import (
    FactorStrategyComparisonResult,
    build_factor_strategy_comparison,
)
from quant_analytics.theme import is_dark, theme_caption

st.set_page_config(page_title="因子策略对比", page_icon=":material/compare_arrows:", layout="wide")
configure_logging()
theme_caption()
dark = is_dark()
st.title("因子策略对比")


def _refresh() -> None:
    st.session_state["factor_strategy_refresh"] = (
        int(st.session_state.get("factor_strategy_refresh", 0)) + 1
    )


with st.sidebar:
    st.radio(
        "数据环境",
        ["正式", "测试"],
        horizontal=True,
        key="factor_strategy_env_label",
    )
    if st.button("刷新数据", icon=":material/refresh:", width="stretch"):
        _refresh()
        st.rerun()
    st.caption("缓存 1 小时；刷新只影响本页")

env = "test" if st.session_state.get("factor_strategy_env_label") == "测试" else "prod"
token = int(st.session_state.get("factor_strategy_refresh", 0))
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

factor_labels: dict[str, str] = {}
for row in runs.iter_rows(named=True):
    factor_id = row.get("factor_id")
    if factor_id is None:
        continue
    key = str(factor_id)
    factor_labels.setdefault(
        key,
        str(row.get("factor_name") or factor_id),
    )
model_labels = sorted({str(value) for value in runs["model_type"].to_list() if value is not None})
benchmark_labels = {
    str(row["benchmark_id"]): f"{row['name']} ({row['benchmark_id']})"
    for row in benchmarks.iter_rows(named=True)
}

with st.sidebar, st.form("factor_strategy_filters"):
    selected_factors = st.multiselect(
        "因子",
        options=list(factor_labels),
        format_func=lambda value: factor_labels[value],
        key="factor_strategy_factors",
    )
    selected_models = st.multiselect(
        "模型",
        options=model_labels,
        key="factor_strategy_models",
    )
    benchmark_options = ["__none__"] + list(benchmark_labels)
    benchmark_id = st.selectbox(
        "统一 benchmark（可选）",
        options=benchmark_options,
        format_func=lambda value: ("无" if value == "__none__" else benchmark_labels[value]),
        key="factor_strategy_benchmark",
    )
    benchmark_id = None if benchmark_id == "__none__" else benchmark_id
    start_default = min(
        (row["start_date"] for row in runs.iter_rows(named=True) if row["start_date"]),
        default=date(2010, 1, 1),
    )
    end_default = max(
        (row["end_date"] for row in runs.iter_rows(named=True) if row["end_date"]),
        default=date.today(),
    )
    start = st.date_input("开始日期", value=start_default, key="factor_strategy_start")
    end = st.date_input("结束日期", value=end_default, key="factor_strategy_end")
    compare_mode = st.selectbox(
        "对比模式",
        options=["factor", "model", "factor_model"],
        format_func=lambda value: {
            "factor": "按因子",
            "model": "按模型",
            "factor_model": "因子 × 模型",
        }[value],
        key="factor_strategy_compare_mode",
    )
    st.form_submit_button("应用筛选", width="stretch")

if start > end:
    st.error("开始日期不能晚于结束日期")
    st.stop()


@st.fragment
def _render_result(result: FactorStrategyComparisonResult) -> None:
    st.caption(f"{result.nav_wide.height:,} 个交易日 · {result.compare_table.height} 个 run")
    summary = st.columns(4)
    mean_annual = result.compare_table["annualized_return"].mean()
    mean_sharpe = result.compare_table["sharpe_ratio"].mean()
    mean_dd = result.compare_table["max_drawdown"].mean()
    summary[0].metric("run 数", str(result.compare_table.height))
    summary[1].metric("平均年化", "—" if mean_annual is None else f"{mean_annual:.2%}")
    summary[2].metric("平均 Sharpe", "—" if mean_sharpe is None else f"{mean_sharpe:.2f}")
    summary[3].metric("平均最大回撤", "—" if mean_dd is None else f"{mean_dd:.2%}")
    fig_nav = multi_run_nav_chart(result.nav_wide, dark=dark)
    st.plotly_chart(fig_nav, theme=None, width="stretch")
    if benchmark_id and result.excess_wide.width > 1:
        st.plotly_chart(
            multi_run_nav_chart(
                result.excess_wide,
                dark=dark,
                title="多 run 超额净值",
                y_title="超额净值",
            ),
            theme=None,
            width="stretch",
        )
    st.plotly_chart(
        multi_run_drawdown_chart(result.drawdown_wide, dark=dark),
        theme=None,
        width="stretch",
    )
    if not result.rank_ic.is_empty():
        st.plotly_chart(
            rank_ic_comparison_chart(result.rank_ic, dark=dark),
            theme=None,
            width="stretch",
        )
    else:
        st.caption("RankIC：N/A（无日度 RankIC 表）")

    st.dataframe(result.compare_table, width="stretch", hide_index=True)
    for note in result.notes:
        st.caption(note)

    figures = [fig_nav, multi_run_drawdown_chart(result.drawdown_wide, dark=dark)]
    if not result.rank_ic.is_empty():
        figures.append(rank_ic_comparison_chart(result.rank_ic, dark=dark))
    st.download_button(
        "导出当前快照 HTML",
        data=render_strategy_html(
            title="因子策略对比",
            env=env,
            run_id=", ".join(result.runs_meta["run_id"].to_list()),
            benchmark_id=benchmark_id,
            compare_benchmark_ids=(),
            start=start,
            end=end,
            kpis={
                "run_count": float(result.compare_table.height),
                "mean_annualized_return": mean_annual,
                "mean_sharpe_ratio": mean_sharpe,
                "mean_max_drawdown": mean_dd,
            },
            figures=figures,
            table=result.compare_table,
            notes=result.notes,
            dark=dark,
        ),
        file_name="factor_strategy_compare.html",
        mime="text/html",
        width="content",
    )


try:
    with st.spinner("加载多 run 结果…"):
        result = build_factor_strategy_comparison(
            env,
            tuple(selected_factors),
            tuple(selected_models),
            benchmark_id=benchmark_id,
            start=start,
            end=end,
            compare_mode=compare_mode,
            refresh_token=token,
        )
except DataAccessError as exc:
    st.info(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    logger.exception("因子策略页面失败")
    st.warning(f"加载失败：{exc}")
    st.stop()

_render_result(result)

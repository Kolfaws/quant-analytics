"""app.py — quant-analytics 首页。

运行：
    uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.db import get_available_options  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.theme import theme_caption  # noqa: E402

st.set_page_config(
    page_title="Quant Analytics",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)
configure_logging()
theme_caption()

with st.sidebar:
    st.header("数据")
    if st.button("刷新数据", icon=":material/refresh:", width="stretch"):
        get_available_options.clear()
        st.rerun()
    st.caption("重新读取可用策略、基准与因子。")

st.title("Quant Analytics")
st.caption("策略、基准与因子分析")

strategies, baselines, factors = get_available_options()

st.subheader("分析模块")
st.page_link(
    "pages/2_strategy_baseline.py",
    label="策略与基准对比",
    icon=":material/monitoring:",
)
st.caption("净值、超额收益、回撤与风险指标。")

st.page_link(
    "pages/1_factor_analysis.py",
    label="因子深度分析",
    icon=":material/science:",
)
st.caption("IC、分层收益、多空组合与换手率。")

if not strategies or not baselines:
    st.warning("暂无可用的策略与基准数据。")
if not factors:
    st.warning("暂无可用因子数据。")

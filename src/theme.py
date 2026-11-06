"""theme — 读取当前 Streamlit 主题并统一图表/导出配色。

Streamlit 原生主题由右上角 Settings -> Theme 选择（Light / Dark / System）。
本模块只负责让 Plotly 图表和导出的独立 HTML 跟随该选择，避免图表与界面不一致。
"""

from __future__ import annotations

import streamlit as st


def is_dark() -> bool:
    """当前 Streamlit 会话是否为 dark 主题。"""
    try:
        theme = st.context.theme
        return str(getattr(theme, "type", "") or "").lower() == "dark"
    except Exception:  # pragma: no cover - 无脚本上下文时按 light 处理
        return False


def theme_caption() -> None:
    """在侧边栏提示如何切换主题。"""
    st.sidebar.caption("主题：右上角 Settings → Theme，可切换 Light / Dark / System。")

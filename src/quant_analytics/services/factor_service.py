"""factor_service — 因子页的加载 + 指标编排。

页面只负责渲染；数据读取在 ``db`` / ``repositories.factor_repo``，指标计算在
``metrics``。本模块把两者串起来，便于测试和未来替换数据源。
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quant_analytics.db import FactorData, get_factor_data
from quant_analytics.metrics import (
    calc_daily_ic,
    calc_ic_summary,
    calc_quantile_returns,
    calc_return_statistics,
    calc_turnover,
)


@dataclass(frozen=True, slots=True)
class FactorAnalysis:
    factor_data: FactorData
    ic_df: pl.DataFrame
    ic_summary: dict[str, float | None]
    daily: pl.DataFrame
    turnover: pl.DataFrame
    long_short: pl.DataFrame
    ls_stats: dict[str, float | None]


def analyze_factor_data(factor_data: FactorData, *, n_groups: int = 5) -> FactorAnalysis:
    """对已加载的因子样本计算 IC/分层/换手/多空统计。"""
    raw = factor_data.frame
    ic_df = calc_daily_ic(raw)
    daily = calc_quantile_returns(raw, n_groups=n_groups)
    turnover = calc_turnover(raw, n_groups=n_groups)
    long_short = (
        daily.select("trade_date", "long_short_ret")
        .unique(subset=["trade_date"], keep="first")
        .sort("trade_date")
    )
    return FactorAnalysis(
        factor_data=factor_data,
        ic_df=ic_df,
        ic_summary=calc_ic_summary(ic_df),
        daily=daily,
        turnover=turnover,
        long_short=long_short,
        ls_stats=calc_return_statistics(long_short["long_short_ret"]),
    )


def load_factor_analysis(
    factor_name: str,
    start_date: str,
    end_date: str,
    *,
    n_groups: int = 5,
    env: str = "prod",
    cache_token: int = 0,
) -> FactorAnalysis:
    """读取因子样本并计算指标；页面入口。"""
    factor_data = get_factor_data(
        factor_name, start_date, end_date, env=env, cache_token=cache_token
    )
    return analyze_factor_data(factor_data, n_groups=n_groups)


__all__ = ["FactorAnalysis", "analyze_factor_data", "load_factor_analysis"]

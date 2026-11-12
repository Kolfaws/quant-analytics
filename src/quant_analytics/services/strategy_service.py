"""strategy_service — 单 run 的策略/基准比较服务。

页面只调用这里，服务只调用 repositories；不在页面拼 SQL，也不查库计算
KPI。超额优先使用 ``v_bt_nav_excess``，不可用时在 Polars 内存里回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from quant_analytics.db import DataAccessError
from quant_analytics.metrics import (
    annual_return,
    annual_vol,
    beta,
    calmar,
    drawdown,
    information_ratio,
    max_drawdown,
    rank_ic_summary,
    sharpe,
    tracking_error,
    win_rate,
)
from quant_analytics.repositories.backtest_repo import (
    ExcessNav,
    get_benchmark_defs,
    get_benchmark_nav,
    get_excess_nav,
    get_nav_by_runs,
    get_rank_ic_series,
    get_run,
    get_run_metrics,
)
from quant_analytics.repositories.backtest_schema import metric_value


@dataclass(frozen=True, slots=True)
class StrategyComparisonResult:
    """页面 2 的完整展示对象；不含 SQL/连接。"""

    run_meta: dict[str, Any]
    nav: pl.DataFrame
    excess: pl.DataFrame
    drawdown: pl.DataFrame
    benchmark_lines: pl.DataFrame
    kpis: dict[str, float | None]
    ic_kpis: dict[str, float | None] | None
    ic_cum: pl.DataFrame
    notes: list[str] = field(default_factory=list)


def _normalize_nav(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.with_columns((pl.col(column) / pl.col(column).first()).alias(column))


def _wide_metrics_to_dict(wide: pl.DataFrame, run_id: str) -> dict[str, float | None]:
    if wide.is_empty() or "run_id" not in wide.columns:
        return {}
    row = wide.filter(pl.col("run_id") == run_id)
    if row.is_empty():
        return {}
    return {
        name: (None if value is None else float(value))
        for name, value in row.drop("run_id").row(0, named=True).items()
    }


def _fallback_kpis(nav: pl.DataFrame) -> dict[str, float | None]:
    strategy_ret = nav["strategy_ret"].drop_nulls()
    if strategy_ret.is_empty():
        return {}
    strategy_nav = nav["strategy_nav"]
    total = float(strategy_nav[-1] - 1.0) if strategy_nav.len() else None
    annual = annual_return(strategy_ret)
    vol = annual_vol(strategy_ret)
    dd = max_drawdown(strategy_nav)
    metrics: dict[str, float | None] = {
        "total_return": total,
        "annualized_return": annual,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe(strategy_ret),
        "max_drawdown": dd,
        "calmar_ratio": calmar(strategy_ret),
        "win_rate": win_rate(strategy_ret),
    }
    if "benchmark_ret" in nav.columns:
        aligned = nav.select("strategy_ret", "benchmark_ret").drop_nulls()
        active = (
            aligned["strategy_ret"] - aligned["benchmark_ret"]
            if not aligned.is_empty()
            else pl.Series("active", [], dtype=pl.Float64)
        )
        metrics.update(
            {
                "excess_total_return": (
                    float(nav["strategy_nav"][-1] / nav["benchmark_nav"][-1] - 1.0)
                    if nav["benchmark_nav"].len()
                    else None
                ),
                "excess_annualized_return": annual_return(active),
                "tracking_error": tracking_error(active),
                "information_ratio": information_ratio(active),
                "beta": beta(aligned["strategy_ret"], aligned["benchmark_ret"])
                if not aligned.is_empty()
                else None,
            }
        )
    return metrics


def _ic_from_metrics(raw: dict[str, float | None]) -> dict[str, float | None] | None:
    """bt_run_metrics 若已存 RankIC 汇总，直接展示，不现场重算。"""
    mapping = {
        "rank_ic_mean": ("rank_ic_mean", "rank_ic"),
        "rank_ic_std": ("rank_ic_std",),
        "rank_icir": ("rank_icir", "rank_ic_ir"),
        "rank_ic_win_rate": ("rank_ic_win_rate", "rank_ic_win"),
        "n_days": ("rank_ic_n", "rank_ic_days", "n_days"),
    }
    result: dict[str, float | None] = {}
    for target, names in mapping.items():
        value = next((raw[name] for name in names if raw.get(name) is not None), None)
        result[target] = value
    return result if any(value is not None for value in result.values()) else None


def _kpis(
    metrics_wide: pl.DataFrame,
    nav: pl.DataFrame,
    run_id: str,
) -> dict[str, float | None]:
    raw = _wide_metrics_to_dict(metrics_wide, run_id)
    if not raw:
        return _fallback_kpis(nav)
    names = (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "excess_total_return",
        "excess_annualized_return",
        "tracking_error",
        "information_ratio",
        "beta",
    )
    result = {name: metric_value(raw, name) for name in names}
    fallback = _fallback_kpis(nav)
    for key, value in result.items():
        if value is None:
            result[key] = fallback.get(key)
    return result


def build_strategy_comparison(
    env: str,
    run_id: str,
    benchmark_id: str | None,
    *,
    compare_benchmark_ids: tuple[str, ...] = (),
    start: date | None = None,
    end: date | None = None,
    refresh_token: int = 0,
) -> StrategyComparisonResult:
    """构建单个 run 的净值、超额、回撤、KPI、RankIC 和对比基准线。"""
    run_meta = get_run(env, run_id, refresh_token=refresh_token)
    if run_meta is None:
        raise DataAccessError(f"找不到 run：{run_id}")

    raw_nav = get_nav_by_runs(env, (run_id,), start, end, refresh_token=refresh_token)
    if raw_nav.is_empty():
        raise DataAccessError(f"run {run_id} 在日期区间内没有净值")

    strategy = raw_nav.filter(pl.col("run_id") == run_id).select(
        "trade_date",
        pl.col("nav").alias("strategy_nav"),
        pl.col("ret_1d").alias("strategy_ret"),
    )
    strategy = _normalize_nav(strategy.sort("trade_date"), "strategy_nav")

    benchmark_ids = tuple(
        dict.fromkeys(([benchmark_id] if benchmark_id else []) + list(compare_benchmark_ids))
    )
    benchmark_nav = get_benchmark_nav(env, benchmark_ids, start, end, refresh_token=refresh_token)
    benchmark_defs = get_benchmark_defs(env, refresh_token=refresh_token)
    names = {
        str(row["benchmark_id"]): str(row["name"]) for row in benchmark_defs.iter_rows(named=True)
    }

    notes: list[str] = []
    main_benchmark = (
        benchmark_nav.filter(pl.col("benchmark_id") == benchmark_id)
        if benchmark_id
        else pl.DataFrame()
    )
    if benchmark_id and main_benchmark.is_empty():
        notes.append(f"主基准 {benchmark_id} 在日期区间内无数据")
        nav = strategy
    elif benchmark_id:
        bench = main_benchmark.select(
            "trade_date",
            pl.col("nav").alias("benchmark_nav"),
            pl.col("ret_1d").alias("benchmark_ret"),
        )
        nav = strategy.join(bench, on="trade_date", how="inner")
        if nav.is_empty():
            raise DataAccessError(f"策略 {run_id} 与基准 {benchmark_id} 没有共同日期")
        nav = _normalize_nav(nav.sort("trade_date"), "benchmark_nav")
        notes.append(f"交易日取策略与主基准 {benchmark_id} 的交集")
    else:
        nav = strategy
        notes.append("未选择主基准，超额指标为 N/A")

    if "benchmark_nav" in nav.columns:
        drawdown_frame = drawdown(nav, nav_col="strategy_nav")
        drawdown_frame = drawdown_frame.with_columns(
            (pl.col("benchmark_nav") / pl.col("benchmark_nav").cum_max() - 1.0).alias(
                "benchmark_drawdown"
            )
        ).select("trade_date", "drawdown", "benchmark_drawdown")
    else:
        drawdown_frame = drawdown(nav, nav_col="strategy_nav").select("trade_date", "drawdown")

    excess_result = (
        get_excess_nav(env, run_id, benchmark_id, start, end, refresh_token=refresh_token)
        if benchmark_id
        else ExcessNav(
            pl.DataFrame(
                schema={
                    "trade_date": pl.Date(),
                    "excess_nav": pl.Float64(),
                    "excess_ret_1d": pl.Float64(),
                }
            ),
            "fallback",
            "未选择主基准",
        )
    )
    if excess_result.source == "view" and not excess_result.frame.is_empty():
        excess = excess_result.frame.select("trade_date", "excess_nav", "excess_ret_1d")
        if "benchmark_nav" in nav.columns:
            excess = excess.join(
                nav.select("trade_date", "benchmark_nav"),
                on="trade_date",
                how="inner",
            )
        source_note = "超额来源：v_bt_nav_excess"
    else:
        source_note = "超额来源：内存回退（v_bt_nav_excess 不可用）"
        if "benchmark_nav" in nav.columns:
            excess = (
                nav.select(
                    "trade_date",
                    "strategy_nav",
                    "strategy_ret",
                    "benchmark_nav",
                    "benchmark_ret",
                )
                .with_columns(
                    (pl.col("strategy_nav") / pl.col("benchmark_nav")).alias("excess_nav"),
                    (pl.col("strategy_ret") - pl.col("benchmark_ret")).alias("excess_ret_1d"),
                )
                .select("trade_date", "excess_nav", "excess_ret_1d")
            )
        else:
            excess = pl.DataFrame(
                schema={
                    "trade_date": pl.Date(),
                    "excess_nav": pl.Float64(),
                    "excess_ret_1d": pl.Float64(),
                }
            )
        if excess_result.note:
            notes.append(excess_result.note)
    notes.append(source_note)

    if benchmark_ids:
        lines = (
            benchmark_nav.select("benchmark_id", "trade_date", "nav")
            .with_columns((pl.col("nav") / pl.col("nav").first().over("benchmark_id")).alias("nav"))
            .with_columns(
                pl.col("benchmark_id")
                .replace_strict(names, default=pl.col("benchmark_id"))
                .alias("name")
            )
            .select("benchmark_id", "name", "trade_date", "nav")
        )
    else:
        lines = pl.DataFrame(
            schema={
                "benchmark_id": pl.Utf8(),
                "name": pl.Utf8(),
                "trade_date": pl.Date(),
                "nav": pl.Float64(),
            }
        )

    metrics_wide = get_run_metrics(env, (run_id,), refresh_token=refresh_token)
    kpis = _kpis(metrics_wide, nav, run_id)

    ic = get_rank_ic_series(env, (run_id,), refresh_token=refresh_token)
    if ic.is_empty():
        ic_kpis = _ic_from_metrics(_wide_metrics_to_dict(metrics_wide, run_id))
        ic_cum = pl.DataFrame(schema={"trade_date": pl.Date(), "rank_ic_cum": pl.Float64()})
        if ic_kpis is None:
            notes.append("未找到 RankIC 结果，RankIC 显示 N/A")
        else:
            notes.append("RankIC 来源：bt_run_metrics")
    else:
        ic_run = ic.filter(pl.col("run_id") == run_id).sort("trade_date")
        ic_kpis = rank_ic_summary(ic_run)
        ic_cum = ic_run.select(
            "trade_date",
            pl.col("rank_ic").fill_null(0.0).cum_sum().alias("rank_ic_cum"),
        )

    return StrategyComparisonResult(
        run_meta=run_meta,
        nav=nav,
        excess=excess,
        drawdown=drawdown_frame,
        benchmark_lines=lines,
        kpis=kpis,
        ic_kpis=ic_kpis,
        ic_cum=ic_cum,
        notes=notes,
    )


__all__ = ["StrategyComparisonResult", "build_strategy_comparison"]

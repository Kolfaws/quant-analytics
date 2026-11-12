"""factor_strategy_service — 按因子/模型批量比较多个 run。

约束：

* run 列表最多 6 个；
* ``bt_nav``、``bt_run_metrics``、``bt_benchmarks``/``bt_baselines`` 各一次批量查询；
* 每个 run 的 KPI 独立计算，不用 outer join 后的填充值；
* 对比基准只做展示线，不参与主 KPI。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from quant_analytics.db import DataAccessError
from quant_analytics.metrics import (
    annual_return,
    information_ratio,
    max_drawdown,
    sharpe,
    tracking_error,
    win_rate,
)
from quant_analytics.repositories.backtest_repo import (
    get_benchmark_defs,
    get_benchmark_nav,
    get_nav_by_runs,
    get_rank_ic_series,
    get_run_metrics,
    list_success_runs,
)
from quant_analytics.repositories.backtest_schema import metric_value

MAX_RUNS = 6


@dataclass(frozen=True, slots=True)
class FactorStrategyComparisonResult:
    """页面 3 的展示对象。"""

    runs_meta: pl.DataFrame
    nav_wide: pl.DataFrame
    excess_wide: pl.DataFrame
    drawdown_wide: pl.DataFrame
    compare_table: pl.DataFrame
    benchmark_meta: pl.DataFrame
    rank_ic: pl.DataFrame
    notes: list[str] = field(default_factory=list)


def _wide_metrics(wide: pl.DataFrame, run_id: str) -> dict[str, float | None]:
    if wide.is_empty() or "run_id" not in wide.columns:
        return {}
    row = wide.filter(pl.col("run_id") == run_id)
    if row.is_empty():
        return {}
    return {
        name: (None if value is None else float(value))
        for name, value in row.drop("run_id").row(0, named=True).items()
    }


def _metric_or_none(raw: dict[str, float | None], name: str) -> float | None:
    return metric_value(raw, name)


def _filter_runs(
    runs: pl.DataFrame,
    factor_ids: tuple[str, ...],
    model_types: tuple[str, ...],
) -> pl.DataFrame:
    if runs.is_empty():
        return runs
    filtered = runs
    if factor_ids:
        filtered = filtered.filter(pl.col("factor_id").cast(pl.Utf8).is_in(list(factor_ids)))
    if model_types:
        filtered = filtered.filter(pl.col("model_type").cast(pl.Utf8).is_in(list(model_types)))
    return filtered.sort("created_at", descending=True, nulls_last=True)


def build_factor_strategy_comparison(
    env: str,
    factor_ids: tuple[str, ...] = (),
    model_types: tuple[str, ...] = (),
    *,
    benchmark_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
    compare_mode: str = "factor",
    refresh_token: int = 0,
) -> FactorStrategyComparisonResult:
    """按因子/模型筛选 run，返回多曲线与独立 KPI 对比表。"""
    notes: list[str] = []
    runs = list_success_runs(env, refresh_token=refresh_token)
    selected = _filter_runs(runs, factor_ids, model_types)
    if selected.is_empty():
        raise DataAccessError("没有匹配的成功 run")
    if selected.height > MAX_RUNS:
        notes.append(f"匹配 {selected.height} 个 run，仅展示最新 {MAX_RUNS} 个")
        selected = selected.head(MAX_RUNS)
    run_ids = tuple(str(value) for value in selected["run_id"].to_list())

    nav_long = get_nav_by_runs(env, run_ids, start, end, refresh_token=refresh_token)
    if nav_long.is_empty():
        raise DataAccessError("筛选的 run 在日期区间内没有净值")
    nav_long = nav_long.with_columns(
        (pl.col("nav") / pl.col("nav").first().over("run_id")).alias("nav")
    )
    nav_wide = nav_long.pivot(index="trade_date", on="run_id", values="nav").sort("trade_date")

    drawdown_long = nav_long.select("run_id", "trade_date", "nav").with_columns(
        (pl.col("nav") / pl.col("nav").cum_max().over("run_id") - 1.0).alias("drawdown")
    )
    drawdown_wide = drawdown_long.pivot(index="trade_date", on="run_id", values="drawdown").sort(
        "trade_date"
    )

    metrics_wide = get_run_metrics(env, run_ids, refresh_token=refresh_token)
    benchmark_nav = get_benchmark_nav(
        env, (benchmark_id,) if benchmark_id else (), start, end, refresh_token=refresh_token
    )
    benchmark_defs = get_benchmark_defs(env, refresh_token=refresh_token)
    benchmark_meta = (
        benchmark_defs.filter(pl.col("benchmark_id") == benchmark_id)
        if benchmark_id
        else pl.DataFrame()
    )

    excess_wide = pl.DataFrame(schema={"trade_date": pl.Date()})
    if benchmark_id and not benchmark_nav.is_empty():
        bench = benchmark_nav.select("trade_date", pl.col("nav").alias("benchmark_nav"))
        excess_long = (
            nav_long.select("run_id", "trade_date", "nav")
            .join(bench, on="trade_date", how="inner")
            .with_columns((pl.col("nav") / pl.col("benchmark_nav")).alias("excess_nav"))
        )
        if excess_long.is_empty():
            notes.append(f"基准 {benchmark_id} 与所选 run 没有共同日期")
        else:
            excess_wide = excess_long.pivot(
                index="trade_date", on="run_id", values="excess_nav"
            ).sort("trade_date")
    elif benchmark_id:
        notes.append(f"主基准 {benchmark_id} 无数据，超额为 N/A")
    else:
        notes.append("未选择基准，超额列为 N/A")

    rows: list[dict[str, Any]] = []
    for run in selected.iter_rows(named=True):
        run_id = str(run["run_id"])
        own = nav_long.filter(pl.col("run_id") == run_id).sort("trade_date")
        returns = own["ret_1d"].drop_nulls()
        raw = _wide_metrics(metrics_wide, run_id)
        annual_metric = _metric_or_none(raw, "annualized_return")
        sharpe_metric = _metric_or_none(raw, "sharpe_ratio")
        dd_metric = _metric_or_none(raw, "max_drawdown")
        win_metric = _metric_or_none(raw, "win_rate")
        kpi = {
            "total_return": (float(own["nav"][-1] - 1.0) if not own.is_empty() else None),
            "annualized_return": (
                annual_metric if annual_metric is not None else annual_return(returns)
            ),
            "sharpe_ratio": (sharpe_metric if sharpe_metric is not None else sharpe(returns)),
            "max_drawdown": (dd_metric if dd_metric is not None else max_drawdown(own["nav"])),
            "win_rate": (win_metric if win_metric is not None else win_rate(returns)),
        }
        active = None
        if benchmark_id and not benchmark_nav.is_empty():
            bench = benchmark_nav.select("trade_date", pl.col("ret_1d").alias("benchmark_ret"))
            aligned = (
                own.select("trade_date", "ret_1d")
                .join(bench, on="trade_date", how="inner")
                .drop_nulls()
            )
            if not aligned.is_empty():
                active = aligned["ret_1d"] - aligned["benchmark_ret"]
        kpi["excess_total_return"] = (
            float(excess_wide[run_id][-1] - 1.0)
            if run_id in excess_wide.columns and not excess_wide.is_empty()
            else None
        )
        kpi["tracking_error"] = tracking_error(active) if active is not None else None
        kpi["information_ratio"] = information_ratio(active) if active is not None else None
        rows.append(
            {
                "run_id": run_id,
                "strategy_name": run.get("strategy_name"),
                "factor_id": run.get("factor_id"),
                "factor_name": run.get("factor_name"),
                "model_type": run.get("model_type"),
                "benchmark_id": benchmark_id,
                "compare_mode": compare_mode,
                **kpi,
            }
        )
    compare_table = pl.DataFrame(rows) if rows else pl.DataFrame()
    if compare_table.is_empty():
        compare_table = pl.DataFrame(
            schema={
                "run_id": pl.Utf8(),
                "strategy_name": pl.Utf8(),
                "factor_id": pl.Utf8(),
                "factor_name": pl.Utf8(),
                "model_type": pl.Utf8(),
                "benchmark_id": pl.Utf8(),
                "compare_mode": pl.Utf8(),
                "total_return": pl.Float64(),
                "annualized_return": pl.Float64(),
                "sharpe_ratio": pl.Float64(),
                "max_drawdown": pl.Float64(),
                "win_rate": pl.Float64(),
                "excess_total_return": pl.Float64(),
                "tracking_error": pl.Float64(),
                "information_ratio": pl.Float64(),
            }
        )

    rank_ic = get_rank_ic_series(env, run_ids, refresh_token=refresh_token)
    if rank_ic.is_empty():
        notes.append("无日度 RankIC 表，RankIC 对比显示 N/A")
    else:
        rank_ic = rank_ic.filter(pl.col("run_id").is_in(list(run_ids)))

    runs_meta = selected.select(
        "run_id",
        "strategy_id",
        "strategy_name",
        "factor_id",
        "factor_name",
        "model_type",
        "benchmark_id",
        "start_date",
        "end_date",
        "status",
    )
    return FactorStrategyComparisonResult(
        runs_meta=runs_meta,
        nav_wide=nav_wide,
        excess_wide=excess_wide,
        drawdown_wide=drawdown_wide,
        compare_table=compare_table,
        benchmark_meta=benchmark_meta,
        rank_ic=rank_ic,
        notes=notes,
    )


__all__ = [
    "MAX_RUNS",
    "FactorStrategyComparisonResult",
    "build_factor_strategy_comparison",
]

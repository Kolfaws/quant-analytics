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


def _last_value(series: pl.Series) -> float | None:
    """末值：先 ``drop_nulls()`` 再 ``last()``；短 run / 末尾缺失不崩。"""
    cleaned = series.drop_nulls()
    if cleaned.is_empty():
        return None
    return float(cleaned.last())


def _stored_kpis_usable(
    run: dict[str, Any],
    start: date | None,
    end: date | None,
) -> bool:
    """已存 ``bt_run_metrics`` 是否与所选窗口一致。

    只有所选窗口完整覆盖 run 区间时才可直接复用已存的策略侧指标；窗口被
    收窄时必须按窗口重算，避免把全区间旧指标贴到子窗口上。超额/跟踪误差/
    IR 始终按所选主基准在窗口内重算，不读已存值。
    """
    run_start = run.get("start_date")
    run_end = run.get("end_date")
    if not isinstance(run_start, date) or not isinstance(run_end, date):
        return False
    if start is not None and run_start < start:
        return False
    return not (end is not None and run_end > end)


def _kpi_missing_note(missing: dict[str, list[str]]) -> str | None:
    if not missing:
        return None
    detail = "；".join(f"{name}=[{', '.join(run_ids)}]" for name, run_ids in missing.items())
    return f"KPI 缺失置空（窗口内数据不足或无基准交集）：{detail}"


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
    nav_long = nav_long.filter(pl.col("nav").is_not_null() & pl.col("nav").is_finite())
    if nav_long.is_empty():
        raise DataAccessError("筛选的 run 在日期区间内没有有效净值")
    nav_long = nav_long.sort(["run_id", "trade_date"]).with_columns(
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
    if not benchmark_nav.is_empty():
        benchmark_nav = benchmark_nav.filter(
            pl.col("nav").is_not_null() & pl.col("nav").is_finite()
        )
    benchmark_defs = get_benchmark_defs(env, refresh_token=refresh_token)
    benchmark_meta = (
        benchmark_defs.filter(pl.col("benchmark_id") == benchmark_id)
        if benchmark_id
        else pl.DataFrame()
    )

    excess_wide = pl.DataFrame(schema={"trade_date": pl.Date()})
    excess_aligned = pl.DataFrame(
        schema={"run_id": pl.Utf8(), "trade_date": pl.Date(), "excess_nav": pl.Float64()}
    )
    if benchmark_id and not benchmark_nav.is_empty():
        bench = benchmark_nav.select("trade_date", pl.col("nav").alias("benchmark_nav"))
        excess_aligned = (
            nav_long.select("run_id", "trade_date", "nav")
            .join(bench, on="trade_date", how="inner")
            .sort(["run_id", "trade_date"])
        )
        if excess_aligned.is_empty():
            notes.append(f"基准 {benchmark_id} 与所选 run 没有共同日期，超额为 N/A")
        else:
            # 策略与基准各自以「窗口内首个共同交易日」为 1 重新归一，
            # 超额起点恒为 1，不受基准自身 base_date 影响。
            excess_aligned = excess_aligned.with_columns(
                (
                    pl.col("nav")
                    / pl.col("nav").first().over("run_id")
                    / (pl.col("benchmark_nav") / pl.col("benchmark_nav").first().over("run_id"))
                ).alias("excess_nav")
            )
            excess_wide = excess_aligned.pivot(
                index="trade_date", on="run_id", values="excess_nav"
            ).sort("trade_date")
            notes.append(f"超额：策略与主基准 {benchmark_id} 在窗口首个共同交易日归一为 1")
    elif benchmark_id:
        notes.append(f"主基准 {benchmark_id} 无数据，超额为 N/A")
    else:
        notes.append("未选择基准，超额列为 N/A")

    benchmark_returns = (
        benchmark_nav.select("trade_date", pl.col("ret_1d").alias("benchmark_ret"))
        if not benchmark_nav.is_empty() and "ret_1d" in benchmark_nav.columns
        else pl.DataFrame(schema={"trade_date": pl.Date(), "benchmark_ret": pl.Float64()})
    )

    kpi_names = (
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "excess_total_return",
        "tracking_error",
        "information_ratio",
    )
    rows: list[dict[str, Any]] = []
    missing_kpi: dict[str, list[str]] = {}
    stored_used = False
    window_recomputed: list[str] = []
    for run in selected.iter_rows(named=True):
        run_id = str(run["run_id"])
        own = nav_long.filter(pl.col("run_id") == run_id).sort("trade_date")
        returns = (
            own["ret_1d"].drop_nulls()
            if "ret_1d" in own.columns
            else pl.Series("ret_1d", [], dtype=pl.Float64)
        )
        nav_last = _last_value(own["nav"])
        kpi: dict[str, float | None] = {
            "total_return": None if nav_last is None else nav_last - 1.0,
            "annualized_return": annual_return(returns),
            "sharpe_ratio": sharpe(returns),
            "max_drawdown": max_drawdown(own["nav"]),
            "win_rate": win_rate(returns),
        }

        raw = _wide_metrics(metrics_wide, run_id)
        if raw and _stored_kpis_usable(run, start, end):
            for name in ("annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"):
                value = _metric_or_none(raw, name)
                if value is not None:
                    kpi[name] = value
                    stored_used = True
        elif raw and any(
            _metric_or_none(raw, name) is not None
            for name in ("annualized_return", "sharpe_ratio", "max_drawdown", "win_rate")
        ):
            window_recomputed.append(run_id)

        excess_series = (
            excess_aligned.filter(pl.col("run_id") == run_id)["excess_nav"]
            if not excess_aligned.is_empty()
            else None
        )
        excess_last = None if excess_series is None else _last_value(excess_series)
        kpi["excess_total_return"] = None if excess_last is None else excess_last - 1.0

        active = None
        if benchmark_id and not benchmark_returns.is_empty() and "ret_1d" in own.columns:
            aligned = (
                own.select("trade_date", "ret_1d")
                .join(benchmark_returns, on="trade_date", how="inner")
                .drop_nulls()
            )
            if not aligned.is_empty():
                active = aligned["ret_1d"] - aligned["benchmark_ret"]
        kpi["tracking_error"] = tracking_error(active) if active is not None else None
        kpi["information_ratio"] = information_ratio(active) if active is not None else None

        for name in kpi_names:
            if kpi.get(name) is None:
                missing_kpi.setdefault(name, []).append(run_id)

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
                **{name: pl.Float64() for name in kpi_names},
            }
        )
    else:
        compare_table = compare_table.with_columns(
            [pl.col(name).cast(pl.Float64) for name in kpi_names]
        )

    if stored_used:
        notes.append("部分 run 的 KPI 使用已存 bt_run_metrics（所选窗口完整覆盖 run 区间）")
    if window_recomputed:
        notes.append(
            "已存 bt_run_metrics 未覆盖所选窗口，已按窗口重算：" + "、".join(window_recomputed)
        )
    missing_note = _kpi_missing_note(missing_kpi)
    if missing_note:
        notes.append(missing_note)
    notes.append(
        "KPI 口径：所选窗口 "
        f"{start.isoformat() if start else '最早'} ~ {end.isoformat() if end else '最新'}，"
        f"主基准 {benchmark_id or '未选'}；数据不足的指标置空（N/A）"
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

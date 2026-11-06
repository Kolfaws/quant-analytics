"""metrics — Polars 向量化绩效与因子指标。

约束：

* 不写 Python ``for`` 循环、不用 ``apply``；
* join / group_by / rank / cum_prod 全部在 Polars 表达式里完成；
* 只有 beta 的协方差矩阵调用 numpy（底层数学聚合的例外）。
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import polars as pl

TRADING_DAYS: Final[int] = 252


def _to_float(value: object) -> float:
    """把 Polars 的数值聚合结果安全转成 float，并给类型检查器明确类型。"""
    if isinstance(value, bool) or not isinstance(value, int | float | np.floating):
        raise TypeError(f"期望数值，实际为 {type(value).__name__}")
    return float(value)


# ---------------------------------------------------------------------------
# 基础序列指标
# ---------------------------------------------------------------------------
def _clean(series: pl.Series) -> pl.Series:
    cleaned = series.drop_nulls().cast(pl.Float64)
    return cleaned.filter(cleaned.is_finite())


def _series_metrics(returns: pl.Series) -> dict[str, float | None]:
    """单条日收益序列的绩效指标。"""
    series = _clean(returns)
    if series.is_empty():
        return {
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
        }
    count = series.len()
    gross = _to_float((1.0 + series).product())
    annual = None if gross <= 0 else gross ** (TRADING_DAYS / count) - 1.0
    volatility = _to_float(series.std(ddof=1)) * math.sqrt(TRADING_DAYS) if count > 1 else None
    std = _to_float(series.std(ddof=1)) if count > 1 else 0.0
    sharpe = _to_float(series.mean()) / std * math.sqrt(TRADING_DAYS) if std and std > 0 else None
    nav = (1.0 + series).cum_prod()
    drawdown = nav / nav.cum_max() - 1.0
    max_dd = _to_float(drawdown.min())
    calmar = annual / abs(max_dd) if annual is not None and max_dd < 0 else None
    active = series.filter(series != 0)
    win = _to_float((active > 0).mean()) if not active.is_empty() else None
    return {
        "annual_return": annual,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win,
    }


def calc_cumulative_returns(df: pl.DataFrame) -> pl.DataFrame:
    """计算策略 / 基准 / 超额累计净值。

    输入列：``trade_date``、``strategy_nav``、``baseline_nav``（或 ret 列）。
    """
    frame = df.sort("trade_date")
    if "strategy_nav" not in frame.columns:
        frame = frame.with_columns((1.0 + pl.col("strategy_ret")).cum_prod().alias("strategy_nav"))
    if "baseline_nav" not in frame.columns:
        frame = frame.with_columns((1.0 + pl.col("baseline_ret")).cum_prod().alias("baseline_nav"))
    if "strategy_ret" not in frame.columns:
        frame = frame.with_columns(
            (pl.col("strategy_nav") / pl.col("strategy_nav").shift(1) - 1.0)
            .fill_null(0.0)
            .alias("strategy_ret")
        )
    if "baseline_ret" not in frame.columns:
        frame = frame.with_columns(
            (pl.col("baseline_nav") / pl.col("baseline_nav").shift(1) - 1.0)
            .fill_null(0.0)
            .alias("baseline_ret")
        )
    return frame.with_columns(
        (pl.col("strategy_nav") / pl.col("strategy_nav").first()).alias("strategy_cum"),
        (pl.col("baseline_nav") / pl.col("baseline_nav").first()).alias("baseline_cum"),
        (pl.col("strategy_ret") - pl.col("baseline_ret")).alias("excess_ret"),
    ).with_columns((pl.col("strategy_cum") / pl.col("baseline_cum")).alias("excess_cum"))


def calc_drawdown(df: pl.DataFrame, nav_col: str = "strategy_nav") -> pl.DataFrame:
    """在 DataFrame 上追加 ``drawdown`` 列（相对历史最高净值）。"""
    if nav_col not in df.columns:
        raise ValueError(f"缺少净值列：{nav_col}")
    return df.sort("trade_date").with_columns(
        (pl.col(nav_col) / pl.col(nav_col).cum_max() - 1.0).alias("drawdown")
    )


def calc_performance_metrics(df: pl.DataFrame) -> dict[str, float | None]:
    """策略 / 基准 / 超额的核心绩效指标。"""
    for column in ("strategy_ret", "baseline_ret"):
        if column not in df.columns:
            raise ValueError(f"缺少收益列：{column}")
    strategy = _series_metrics(df["strategy_ret"])
    baseline = _series_metrics(df["baseline_ret"])
    aligned = (
        df.select("strategy_ret", "baseline_ret")
        .drop_nulls()
        .filter(pl.col("strategy_ret").is_finite() & pl.col("baseline_ret").is_finite())
        .with_columns((pl.col("strategy_ret") - pl.col("baseline_ret")).alias("active_ret"))
    )
    active = _series_metrics(aligned["active_ret"]) if not aligned.is_empty() else {}
    beta = _beta(aligned)
    active_return = active.get("annual_return")
    active_volatility = active.get("annual_volatility")
    information_ratio = (
        active_return / active_volatility
        if active_return is not None and active_volatility
        else None
    )
    metrics: dict[str, float | None] = {
        "strategy_annual_return": strategy["annual_return"],
        "baseline_annual_return": baseline["annual_return"],
        "excess_annual_return": active.get("annual_return"),
        "sharpe_ratio": strategy["sharpe"],
        "baseline_sharpe": baseline["sharpe"],
        "max_drawdown": strategy["max_drawdown"],
        "baseline_max_drawdown": baseline["max_drawdown"],
        "calmar_ratio": strategy["calmar"],
        "win_rate": strategy["win_rate"],
        "annual_volatility": strategy["annual_volatility"],
        "tracking_error": active_volatility,
        "information_ratio": information_ratio,
        "beta": beta,
    }
    return metrics


def _beta(aligned: pl.DataFrame) -> float | None:
    if aligned.height < 2:
        return None
    strategy = aligned["strategy_ret"].to_numpy()
    benchmark = aligned["baseline_ret"].to_numpy()
    variance = float(np.var(benchmark, ddof=1))
    if variance == 0:
        return None
    covariance = float(np.cov(strategy, benchmark, ddof=1)[0, 1])
    return covariance / variance


# ---------------------------------------------------------------------------
# 因子指标
# ---------------------------------------------------------------------------
def _valid_factor_frame(df: pl.DataFrame, factor_col: str, return_col: str) -> pl.DataFrame:
    for column in ("trade_date", "ts_code", factor_col, return_col):
        if column not in df.columns:
            raise ValueError(f"缺少列：{column}")
    return df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(factor_col).is_finite()
        & pl.col(return_col).is_not_null()
        & pl.col(return_col).is_finite()
    )


def calc_daily_ic(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    return_col: str = "forward_return",
) -> pl.DataFrame:
    """按交易日计算 Pearson IC 与 Rank IC（Spearman）。"""
    frame = _valid_factor_frame(df, factor_col, return_col)
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date(),
                "ic": pl.Float64(),
                "rank_ic": pl.Float64(),
                "n": pl.UInt32(),
            }
        )
    ranked = frame.with_columns(
        pl.col(factor_col).rank(method="average").over("trade_date").alias("_factor_rank"),
        pl.col(return_col).rank(method="average").over("trade_date").alias("_return_rank"),
    )
    return (
        ranked.group_by("trade_date")
        .agg(
            pl.corr(pl.col(factor_col), pl.col(return_col)).alias("ic"),
            pl.corr(pl.col("_factor_rank"), pl.col("_return_rank")).alias("rank_ic"),
            pl.len().alias("n"),
        )
        .sort("trade_date")
        .with_columns(
            pl.col("ic").fill_nan(None),
            pl.col("rank_ic").fill_nan(None),
        )
    )


def calc_ic_summary(ic_df: pl.DataFrame) -> dict[str, float | None]:
    """IC 均值 / 标准差 / ICIR / IC>0 胜率。"""
    if ic_df.is_empty():
        return {
            "ic_mean": None,
            "ic_std": None,
            "icir": None,
            "ic_win_rate": None,
            "rank_ic_mean": None,
            "rank_ic_std": None,
            "rank_icir": None,
            "rank_ic_win_rate": None,
            "n_days": 0,
        }

    def _summary(
        column: str,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        series = _clean(ic_df[column])
        if series.is_empty():
            return None, None, None, None
        mean = _to_float(series.mean())
        std = _to_float(series.std(ddof=1)) if series.len() > 1 else 0.0
        icir = mean / std if std > 0 else None
        win = _to_float((series > 0).mean())
        return mean, std, icir, win

    ic_mean, ic_std, icir, ic_win = _summary("ic")
    rank_mean, rank_std, rank_icir, rank_win = _summary("rank_ic")
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_win_rate": ic_win,
        "rank_ic_mean": rank_mean,
        "rank_ic_std": rank_std,
        "rank_icir": rank_icir,
        "rank_ic_win_rate": rank_win,
        "n_days": ic_df.height,
    }


def _assign_groups(
    df: pl.DataFrame,
    factor_col: str,
    return_col: str,
    n_groups: int,
) -> pl.DataFrame:
    if n_groups < 2:
        raise ValueError("n_groups 必须 >= 2")
    frame = _valid_factor_frame(df, factor_col, return_col)
    if frame.is_empty():
        return frame
    ranked = frame.with_columns(
        pl.col(factor_col)
        .rank(method="ordinal", descending=True)
        .over("trade_date")
        .alias("_rank"),
        pl.col(factor_col).count().over("trade_date").alias("_count"),
    )
    return ranked.with_columns(
        ((((pl.col("_rank") - 1) * n_groups / pl.col("_count")).floor()) + 1)
        .cast(pl.Int32)
        .clip(1, n_groups)
        .alias("group")
    ).drop(["_rank", "_count"])


def calc_quantile_returns(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    return_col: str = "forward_return",
    n_groups: int = 5,
) -> pl.DataFrame:
    """按日分 N 组，返回分组收益、累计净值、多空与仅多头序列。"""
    assigned = _assign_groups(df, factor_col, return_col, n_groups)
    if assigned.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date(),
                "group": pl.Int32(),
                "group_ret": pl.Float64(),
                "group_cum": pl.Float64(),
                "top_only_ret": pl.Float64(),
                "long_short_ret": pl.Float64(),
                "top_only_cum": pl.Float64(),
                "long_short_cum": pl.Float64(),
                "n": pl.UInt32(),
            }
        )

    daily = (
        assigned.group_by(["trade_date", "group"])
        .agg(
            pl.col(return_col).mean().alias("group_ret"),
            pl.len().alias("n"),
        )
        .sort(["trade_date", "group"])
    )
    daily = daily.with_columns(
        ((1.0 + pl.col("group_ret")).cum_prod().over("group")).alias("group_cum")
    )

    top = daily.filter(pl.col("group") == 1).select(
        "trade_date",
        pl.col("group_ret").alias("top_only_ret"),
    )
    bottom = daily.filter(pl.col("group") == pl.col("group").max().over("trade_date")).select(
        "trade_date",
        pl.col("group_ret").alias("bottom_ret"),
    )
    long_short = (
        top.join(bottom, on="trade_date", how="inner")
        .with_columns((pl.col("top_only_ret") - pl.col("bottom_ret")).alias("long_short_ret"))
        .with_columns(
            (1.0 + pl.col("top_only_ret")).cum_prod().alias("top_only_cum"),
            (1.0 + pl.col("long_short_ret")).cum_prod().alias("long_short_cum"),
        )
        .select(
            "trade_date",
            "top_only_ret",
            "long_short_ret",
            "top_only_cum",
            "long_short_cum",
        )
    )
    return daily.join(long_short, on="trade_date", how="left").sort(["trade_date", "group"])


def calc_turnover(
    df: pl.DataFrame,
    factor_col: str = "factor_value",
    n_groups: int = 5,
) -> pl.DataFrame:
    """Top 组日度换手率 = 1 - 与上一交易日 Top 组的交集 / Top 组数量。"""
    assigned = _assign_groups(df, factor_col, "forward_return", n_groups)
    if assigned.is_empty():
        return pl.DataFrame(
            schema={
                "trade_date": pl.Date(),
                "turnover": pl.Float64(),
                "top_n": pl.UInt32(),
            }
        )

    top = assigned.filter(pl.col("group") == 1).select("trade_date", "ts_code")
    dates = (
        assigned.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_columns(pl.col("trade_date").shift(1).alias("prev_date"))
    )
    top_prev = top.select(pl.col("trade_date").alias("prev_date"), "ts_code").join(
        dates, on="prev_date", how="inner"
    )
    kept = (
        top.join(top_prev, on=["trade_date", "ts_code"], how="inner")
        .group_by("trade_date")
        .len()
        .rename({"len": "kept"})
    )
    counts = top.group_by("trade_date").len().rename({"len": "top_n"})
    first_date = dates["trade_date"].min()
    return (
        counts.join(kept, on="trade_date", how="left")
        .with_columns(pl.col("kept").fill_null(0))
        .with_columns(
            pl.when(pl.col("trade_date") == first_date)
            .then(0.0)
            .otherwise(1.0 - pl.col("kept") / pl.col("top_n"))
            .alias("turnover")
        )
        .select("trade_date", "turnover", "top_n")
        .sort("trade_date")
    )


def calc_return_statistics(returns: pl.Series) -> dict[str, float | None]:
    """给单条收益序列（如多空组合）直接算 KPI。"""
    return _series_metrics(returns)


__all__ = [
    "TRADING_DAYS",
    "calc_cumulative_returns",
    "calc_daily_ic",
    "calc_drawdown",
    "calc_ic_summary",
    "calc_performance_metrics",
    "calc_quantile_returns",
    "calc_return_statistics",
    "calc_turnover",
]

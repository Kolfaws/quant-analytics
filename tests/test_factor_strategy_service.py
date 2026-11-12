"""多 run 服务：批量查询、run 上限与独立 KPI。"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quant_analytics.db import DataAccessError
from quant_analytics.services import factor_strategy_service


def _runs(count: int = 8) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(count)],
            "strategy_id": ["s"] * count,
            "strategy_name": [f"S{i}" for i in range(count)],
            "factor_id": ["f1" if i % 2 == 0 else "f2" for i in range(count)],
            "factor_name": ["F1" if i % 2 == 0 else "F2" for i in range(count)],
            "model_type": ["ridge" if i < 4 else "lgbm" for i in range(count)],
            "benchmark_id": ["BM"] * count,
            "start_date": [date(2024, 1, 2)] * count,
            "end_date": [date(2024, 1, 4)] * count,
            "status": ["success"] * count,
            "created_at": [date(2024, 1, 1)] * count,
        }
    )


def _nav(run_ids: tuple[str, ...]) -> pl.DataFrame:
    rows = []
    for run_id in run_ids:
        step = 0.01 if run_id.endswith("0") else 0.02
        for offset, day in enumerate((date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))):
            rows.append((run_id, day, 1.0 + offset * step, step))
    return pl.DataFrame(
        rows,
        schema={
            "run_id": pl.Utf8(),
            "trade_date": pl.Date(),
            "nav": pl.Float64(),
            "ret_1d": pl.Float64(),
        },
        orient="row",
    )


def test_batch_query_limit_and_independent_kpis(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_nav(env, run_ids, *args, **kwargs):
        calls.append(tuple(run_ids))
        return _nav(tuple(run_ids))

    monkeypatch.setattr(factor_strategy_service, "list_success_runs", lambda *a, **k: _runs())
    monkeypatch.setattr(factor_strategy_service, "get_nav_by_runs", fake_nav)
    monkeypatch.setattr(
        factor_strategy_service,
        "get_run_metrics",
        lambda *a, **k: pl.DataFrame(schema={"run_id": pl.Utf8()}),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_benchmark_nav",
        lambda *a, **k: pl.DataFrame(
            schema={"benchmark_id": pl.Utf8(), "trade_date": pl.Date(), "nav": pl.Float64()}
        ),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_benchmark_defs",
        lambda *a, **k: pl.DataFrame(schema={"benchmark_id": pl.Utf8(), "name": pl.Utf8()}),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_rank_ic_series",
        lambda *a, **k: pl.DataFrame(
            schema={"run_id": pl.Utf8(), "trade_date": pl.Date(), "rank_ic": pl.Float64()}
        ),
    )

    result = factor_strategy_service.build_factor_strategy_comparison("prod")
    assert len(calls) == 1
    assert len(calls[0]) == 6
    assert result.compare_table.height == 6
    assert result.nav_wide.width == 7
    totals = dict(
        zip(
            result.compare_table["run_id"].to_list(),
            result.compare_table["total_return"].to_list(),
            strict=True,
        )
    )
    assert len(set(totals.values())) > 1
    assert result.excess_wide.width == 1


def test_factor_and_model_filters(monkeypatch) -> None:
    monkeypatch.setattr(factor_strategy_service, "list_success_runs", lambda *a, **k: _runs())
    monkeypatch.setattr(
        factor_strategy_service,
        "get_nav_by_runs",
        lambda env, run_ids, *a, **k: _nav(tuple(run_ids)),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_run_metrics",
        lambda *a, **k: pl.DataFrame(schema={"run_id": pl.Utf8()}),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_benchmark_nav",
        lambda *a, **k: pl.DataFrame(
            schema={"benchmark_id": pl.Utf8(), "trade_date": pl.Date(), "nav": pl.Float64()}
        ),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_benchmark_defs",
        lambda *a, **k: pl.DataFrame(schema={"benchmark_id": pl.Utf8(), "name": pl.Utf8()}),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_rank_ic_series",
        lambda *a, **k: pl.DataFrame(
            schema={"run_id": pl.Utf8(), "trade_date": pl.Date(), "rank_ic": pl.Float64()}
        ),
    )
    result = factor_strategy_service.build_factor_strategy_comparison("prod", ("f1",), ("ridge",))
    assert result.runs_meta.height == 2
    assert set(result.runs_meta["factor_id"].to_list()) == {"f1"}
    assert set(result.runs_meta["model_type"].to_list()) == {"ridge"}


def test_no_matching_runs_raises(monkeypatch) -> None:
    monkeypatch.setattr(factor_strategy_service, "list_success_runs", lambda *a, **k: _runs())
    with pytest.raises(DataAccessError):
        factor_strategy_service.build_factor_strategy_comparison("prod", ("missing",))

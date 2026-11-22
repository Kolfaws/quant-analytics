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


# ---------------------------------------------------------------------------
# P0-2：page3 超额起点、短 run / 末尾缺失、KPI 窗口一致性
# ---------------------------------------------------------------------------
PAGE3_D1 = date(2024, 1, 2)
PAGE3_D2 = date(2024, 1, 3)
PAGE3_D3 = date(2024, 1, 4)
PAGE3_D4 = date(2024, 1, 5)


def _window_runs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "strategy_id": ["s1", "s2"],
            "strategy_name": ["S1", "S2"],
            "factor_id": ["f1", "f1"],
            "factor_name": ["F1", "F1"],
            "model_type": ["ridge", "ridge"],
            "benchmark_id": ["BM", "BM"],
            "start_date": [PAGE3_D1, PAGE3_D1],
            "end_date": [PAGE3_D3, PAGE3_D4],
            "status": ["success", "success"],
            "created_at": [date(2024, 1, 1), date(2024, 1, 1)],
        }
    )


def _window_nav() -> pl.DataFrame:
    rows = [
        ("r1", PAGE3_D1, 1.00, 0.0),
        ("r1", PAGE3_D2, 1.01, 0.01),
        ("r1", PAGE3_D3, 1.03, 0.0198),
        ("r2", PAGE3_D1, 1.00, 0.0),
        ("r2", PAGE3_D2, 1.02, 0.02),
        ("r2", PAGE3_D3, 1.04, 0.0196),
        ("r2", PAGE3_D4, 1.05, 0.0096),
    ]
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


def _window_bench() -> pl.DataFrame:
    rows = [
        ("BM", PAGE3_D1, 100.0, 0.0),
        ("BM", PAGE3_D2, 101.0, 0.01),
        ("BM", PAGE3_D3, 102.0, 0.0099),
        ("BM", PAGE3_D4, 103.0, 0.0098),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "benchmark_id": pl.Utf8(),
            "trade_date": pl.Date(),
            "nav": pl.Float64(),
            "ret_1d": pl.Float64(),
        },
        orient="row",
    )


def _patch_page3(monkeypatch, *, runs, nav, bench, metrics=None) -> None:
    def fake_nav(env, run_ids, start=None, end=None, *, refresh_token=0):
        frame = nav.filter(pl.col("run_id").is_in(list(run_ids)))
        if start is not None:
            frame = frame.filter(pl.col("trade_date") >= start)
        if end is not None:
            frame = frame.filter(pl.col("trade_date") <= end)
        return frame

    def fake_bench(env, benchmark_ids, start=None, end=None, *, refresh_token=0):
        frame = bench.filter(pl.col("benchmark_id").is_in(list(benchmark_ids)))
        if start is not None:
            frame = frame.filter(pl.col("trade_date") >= start)
        if end is not None:
            frame = frame.filter(pl.col("trade_date") <= end)
        return frame

    monkeypatch.setattr(factor_strategy_service, "list_success_runs", lambda *a, **k: runs)
    monkeypatch.setattr(factor_strategy_service, "get_nav_by_runs", fake_nav)
    monkeypatch.setattr(factor_strategy_service, "get_benchmark_nav", fake_bench)
    monkeypatch.setattr(
        factor_strategy_service,
        "get_run_metrics",
        lambda *a, **k: pl.DataFrame(schema={"run_id": pl.Utf8()}) if metrics is None else metrics,
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_benchmark_defs",
        lambda *a, **k: pl.DataFrame({"benchmark_id": ["BM"], "name": ["Benchmark"]}),
    )
    monkeypatch.setattr(
        factor_strategy_service,
        "get_rank_ic_series",
        lambda *a, **k: pl.DataFrame(
            schema={"run_id": pl.Utf8(), "trade_date": pl.Date(), "rank_ic": pl.Float64()}
        ),
    )


def test_page3_excess_starts_at_one_for_different_end_dates(monkeypatch) -> None:
    _patch_page3(
        monkeypatch,
        runs=_window_runs(),
        nav=_window_nav(),
        bench=_window_bench(),
    )
    result = factor_strategy_service.build_factor_strategy_comparison(
        "prod", benchmark_id="BM", start=PAGE3_D1, end=PAGE3_D4
    )

    assert result.excess_wide.height == 4
    for run_id in ("r1", "r2"):
        values = result.excess_wide[run_id].drop_nulls()
        assert not values.is_empty()
        assert values.first() == pytest.approx(1.0)

    # 基准自 benchmark 自身 base_date（nav=100）起算，若不按窗口首日归一，
    # 超额起点会是 1/1.0 但末值口径错误；这里断言归一后末值。
    r2_last = result.excess_wide.filter(pl.col("trade_date") == PAGE3_D4)["r2"].item()
    assert r2_last == pytest.approx(1.05 / 1.0 / (103.0 / 100.0))
    r2_kpi = result.compare_table.filter(pl.col("run_id") == "r2")
    assert r2_kpi["excess_total_return"].item() == pytest.approx(r2_last - 1.0)
    assert any("归一为 1" in note for note in result.notes)


def test_page3_short_run_with_missing_tail_does_not_crash(monkeypatch) -> None:
    nav = pl.DataFrame(
        [
            ("r1", PAGE3_D1, 1.00, 0.0),
            ("r1", PAGE3_D2, 1.02, 0.02),
            ("r1", PAGE3_D3, None, None),  # 末行净值缺失：旧实现 None - 1 崩溃
            ("r2", PAGE3_D1, 1.00, 0.0),  # 单点短 run
        ],
        schema={
            "run_id": pl.Utf8(),
            "trade_date": pl.Date(),
            "nav": pl.Float64(),
            "ret_1d": pl.Float64(),
        },
        orient="row",
    )
    _patch_page3(monkeypatch, runs=_window_runs(), nav=nav, bench=_window_bench())
    result = factor_strategy_service.build_factor_strategy_comparison(
        "prod", benchmark_id="BM", start=PAGE3_D1, end=PAGE3_D4
    )

    r1_kpi = result.compare_table.filter(pl.col("run_id") == "r1")
    assert r1_kpi["total_return"].item() == pytest.approx(0.02)
    r2_kpi = result.compare_table.filter(pl.col("run_id") == "r2")
    assert r2_kpi["total_return"].item() == pytest.approx(0.0)
    for run_id in ("r1", "r2"):
        values = result.excess_wide[run_id].drop_nulls()
        assert not values.is_empty()
        assert values.first() == pytest.approx(1.0)
    assert any("KPI 缺失置空" in note for note in result.notes)


def test_page3_kpi_recomputed_when_window_narrowed(monkeypatch) -> None:
    stored_metrics = pl.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "annualized_return": [9.99, 9.99],
            "sharpe_ratio": [9.99, 9.99],
        }
    )
    _patch_page3(
        monkeypatch,
        runs=_window_runs(),
        nav=_window_nav(),
        bench=_window_bench(),
        metrics=stored_metrics,
    )
    narrowed = factor_strategy_service.build_factor_strategy_comparison(
        "prod", benchmark_id="BM", start=PAGE3_D2, end=PAGE3_D3
    )
    narrowed_kpis = narrowed.compare_table.filter(pl.col("run_id") == "r1")
    assert narrowed_kpis["annualized_return"].item() != pytest.approx(9.99)
    assert any("按窗口重算" in note for note in narrowed.notes)

    full = factor_strategy_service.build_factor_strategy_comparison(
        "prod", benchmark_id="BM", start=PAGE3_D1, end=PAGE3_D4
    )
    full_kpis = full.compare_table.filter(pl.col("run_id") == "r1")
    assert full_kpis["annualized_return"].item() == pytest.approx(9.99)
    assert any("已存 bt_run_metrics" in note for note in full.notes)


def test_page3_without_benchmark_excess_is_none_and_noted(monkeypatch) -> None:
    _patch_page3(
        monkeypatch,
        runs=_window_runs(),
        nav=_window_nav(),
        bench=_window_bench(),
    )
    result = factor_strategy_service.build_factor_strategy_comparison(
        "prod", start=PAGE3_D1, end=PAGE3_D4
    )
    assert result.excess_wide.width == 1
    assert all(value is None for value in result.compare_table["excess_total_return"].to_list())
    assert any("未选择基准" in note for note in result.notes)

"""backtest_schema：缺列兼容与指标别名。"""

from __future__ import annotations

from quant_analytics.repositories.backtest_schema import (
    metric_value,
    rank_ic_metric_names,
    run_metadata,
)


def test_run_metadata_parses_params_fallbacks() -> None:
    meta = run_metadata(
        {
            "run_id": "r1",
            "strategy_id": "s1",
            "status": "success",
            "params_json": '{"model_type":"lgbm","factor_id":"f1","factor_name":"mom"}',
        }
    )
    assert meta["model_type"] == "lgbm"
    assert meta["factor_id"] == "f1"
    assert meta["factor_name"] == "mom"


def test_run_metadata_missing_returns_none() -> None:
    meta = run_metadata({"run_id": "r1", "params_json": "not-json"})
    assert meta["model_type"] is None
    assert meta["factor_id"] is None
    assert meta["factor_name"] is None


def test_metric_alias_and_missing_rank_ic() -> None:
    assert metric_value({"annualized_return": 0.1}, "annualized_return") == 0.1
    assert metric_value({}, "rank_ic_mean") is None
    assert rank_ic_metric_names(("sharpe_ratio", "rank_ic_mean")) == ("rank_ic_mean",)

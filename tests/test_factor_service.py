"""factor_service — 页面编排层的纯函数测试（不连库）。"""

from __future__ import annotations

from datetime import date

import polars as pl

from quant_analytics.db import FactorData, FactorSampleReport
from quant_analytics.services.factor_service import analyze_factor_data


def _factor_data() -> FactorData:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    codes = [f"{i:06d}.SZ" for i in range(10)]
    rows = []
    for d in dates:
        for i, code in enumerate(codes):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": d,
                    "factor_value": float(i),
                    "forward_return": float(i) / 100.0,
                }
            )
    frame = pl.DataFrame(rows)
    return FactorData(
        factor_name="unit",
        direction=1,
        frame=frame,
        report=FactorSampleReport(total_rows=frame.height, kept_rows=frame.height),
    )


def test_analyze_factor_data_returns_all_sections() -> None:
    analysis = analyze_factor_data(_factor_data(), n_groups=5)
    assert analysis.factor_data.factor_name == "unit"
    assert analysis.ic_df.height == 2
    assert not analysis.daily.is_empty()
    assert not analysis.turnover.is_empty()
    assert not analysis.long_short.is_empty()
    assert "rank_ic_mean" in analysis.ic_summary
    assert "sharpe" in analysis.ls_stats

"""业务服务层：页面的数据编排与计算入口。"""

from __future__ import annotations

from .factor_strategy_service import (
    FactorStrategyComparisonResult,
    build_factor_strategy_comparison,
)
from .strategy_service import StrategyComparisonResult, build_strategy_comparison

__all__ = [
    "FactorStrategyComparisonResult",
    "StrategyComparisonResult",
    "build_factor_strategy_comparison",
    "build_strategy_comparison",
]

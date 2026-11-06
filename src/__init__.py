"""quant-analytics — Streamlit 量化分析看板。

纯展示端：
* 只读 DuckDB（backtest / market）；
* Polars 负责计算，Plotly 负责展示；
* 不包含回测、因子计算逻辑。
"""

__version__ = "0.1.0"

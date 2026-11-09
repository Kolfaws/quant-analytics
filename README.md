# quant-analytics

量化分析看板（Streamlit）：只读展示回测、基准与因子研究结果。

- 项目/文件夹：`quant-analytics`
- 技术栈：Python 3.13 / Streamlit / Plotly / Polars / DuckDB / Numpy
- 数据源：
  - `${BACKTEST_DB_PATH}`：`bt_runs`、`bt_nav`、`bt_baselines`、`bt_benchmarks`
  - `${MARKET_DB_PATH}`：`fact_factor_daily`、`fact_daily`、`meta_factor_registry`

> 纯展示端：不包含回测、因子计算或数据库写入逻辑。
> 所有 DuckDB 连接都是 `read_only=True`，不会锁定数据库；查询结果用
> `st.cache_data(ttl=3600)` 缓存。

## 1. 页面

| 页面 | 内容 |
| --- | --- |
| `app.py` | 首页：两个分析入口、正式/测试环境切换、数据刷新 |
| `pages/2_strategy_baseline.py` | 策略 vs 基准：策略/基准选择、超额与回撤开关、5 个 KPI、风险指标、刷新、HTML 导出 |
| `pages/1_factor_analysis.py` | 因子深度：因子/日期/分组控制、IC 累积与换手率开关、6 个 KPI、分层收益、刷新、HTML 导出 |

## 2. 目录

```text
quant-analytics/
├── app.py
├── pages/
│   ├── 1_factor_analysis.py
│   └── 2_strategy_baseline.py
├── src/quant_analytics/
│   ├── __init__.py
│   ├── db.py                # 只读 DuckDB + st.cache_data
│   ├── metrics.py           # Polars 向量化指标
│   ├── charts.py            # Plotly 封装
│   ├── theme.py             # Light / Dark 主题适配
│   ├── py.typed             # PEP 561 类型标记
│   └── logging_utils.py     # loguru
├── tests/test_metrics.py    # IC / 绩效 / 分层 / 换手
├── .streamlit/config.toml
├── .env.example
├── .gitignore / .gitattributes
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
└── README.md
```

`quant-analytics` 是正式的可安装包：

- distribution：`quant-analytics`
- import 包：`quant_analytics`
- `uv sync` 会以 editable 模式安装到 `.venv`；
- 页面只使用 `from quant_analytics...`，没有 `sys.path.insert`；
- pytest 通过 `pyproject.toml` 的 `pythonpath = ["src"]` 解析源码目录。

## 3. 安装

```bash
cd quant-analytics
uv sync
```

依赖：`streamlit`、`plotly`、`polars`、`duckdb`、`numpy`、`loguru`、`pyarrow`、`pytz`。
其中 `pyarrow` 是 Polars/Plotly 转换与 DuckDB Arrow 互操作的运行时依赖，
`pytz` 用于读取 DuckDB 的 `TIMESTAMPTZ` 列。

## 4. 启动、退出与主题

**必须先进入项目目录**。`uv run` 会读取当前目录的 `pyproject.toml` 和 `.venv`：

```bash
cd quant-analytics   # 从工作区根目录进入
uv run streamlit run app.py
```

如果直接在工作区根目录执行 `uv run streamlit run app.py`，`uv` 找不到
`quant-analytics` 的虚拟环境，会报 `Failed to spawn: streamlit` / `program not found`。
也可以显式指定项目目录：

```bash
uv run --project <workspace>/quant-analytics --directory <workspace>/quant-analytics streamlit run app.py
```

指定端口：

```bash
uv run streamlit run app.py --server.port 8501
```

或使用 Makefile（等价于先 cd 到项目目录）：

```bash
make run
```

### 退出

- 前台运行时：在启动终端按 `Ctrl + C`；
- 关闭浏览器标签页**不会**停止服务；
- 如果终端已关闭，按端口查找并结束进程（Windows）：

```powershell
Get-NetTCPConnection -LocalPort 8501 | Select-Object -ExpandProperty OwningProcess | Stop-Process
```

### Light / Dark 主题

- 点击右上角菜单 **Settings → Theme**，选择 **Light / Dark / System**；
- 页面、KPI、图表和导出的 HTML 都会跟随当前主题；
- 未做主题选择时默认使用 `.streamlit/config.toml` 的 `base = "light"`；
- 主题实现见 [src/quant_analytics/theme.py](src/quant_analytics/theme.py)。

### 环境变量

如果数据库路径不同，复制 `.env.example` 为 `.env`：

```env
MARKET_DB_PATH=/absolute/path/to/market.duckdb
BACKTEST_DB_PATH=/absolute/path/to/backtest.duckdb
MARKET_TEST_DB_PATH=/absolute/path/to/market_test.duckdb
BACKTEST_TEST_DB_PATH=/absolute/path/to/backtest_test.duckdb
```

## 5. 计算口径

### 策略/基准

- 收益：`bt_nav.ret_1d` 与 `bt_baselines.ret_1d`，按 `trade_date` 内连接
- 累计净值：`nav / first(nav)`
- 超额净值：`strategy_cum / baseline_cum`
- 绩效：年化收益、年化波动、Sharpe、Calmar、胜率、最大回撤、跟踪误差、IR、Beta
- 年化按 252 个交易日

### 因子

- 因子值：`fact_factor_daily.value`
- 下期收益：按 `ts_code` 分组的 `close.shift(-1) / close - 1`（在 Polars 中）
- Pearson IC / Rank IC：按 `trade_date` 分组，`pl.corr`
- 分层：按日对因子值降序 rank，切成 N 组，G1 = 因子值最高
- 多空：`G1 - 最低组`；仅多头：`G1`
- 换手率：`1 - 与上一交易日 Top 组交集 / Top 组数量`

## 6. 性能与缓存

- DuckDB 查询全部 `@st.cache_data(ttl=3600)`；
- 所有 join / group_by / rank / quantile / cum_prod 都在 Polars 表达式里完成；
- 不使用 Python `for` 循环做计算，不使用 `apply`；
- 仅在 Plotly 渲染前 `.to_pandas()` 做一次短转换；
- 切换侧边栏选项会自动复用缓存；每个页面都有“刷新数据”，会清除该页相关查询缓存并重新读取数据库。

### HTML 导出

页面上的“导出图表 HTML”生成独立文件，只包含：

- 图表标题与日期区间；
- 图表本身；
- 当前 Light/Dark 主题的页面背景、卡片和字体。

不包含 Plotly 默认页面框架、工具栏、导航或技术说明，适合直接归档或发送。
导出使用 Plotly CDN，打开 HTML 时需要网络加载图表脚本。

## 7. 测试

```bash
uv run --with pytest python -m pytest tests -q
# 或
make test
```

覆盖：

- 累计净值 / 超额 / 回撤；
- 年化、Sharpe、Beta；
- 完美单调因子的 Pearson IC / Rank IC；
- 分层收益与多空序列；
- Top 组全换手 / 无换手场景。

## 8. 测试环境

侧边栏“数据环境”可切换：

- 正式：`MARKET_DB_PATH` / `BACKTEST_DB_PATH`；
- 测试：`MARKET_TEST_DB_PATH` / `BACKTEST_TEST_DB_PATH`。

查询缓存按环境分别缓存；切换后自动读取对应数据库，刷新按钮会清空相关缓存。
测试库不存在时页面显示为空或错误提示，不影响正式环境。

## 9. 数据前置条件

- 策略页：`backtest.duckdb` 至少有 `status='success'` 的 `bt_runs`、对应的 `bt_nav`，
  以及 `bt_baselines`。
- 因子页：`market.duckdb` 的 `meta_factor_registry` 有 `is_active=TRUE` 的因子，
  且 `fact_factor_daily` 有数据。
- 若因子目录为空，因子页会给出提示而不是报错；首页会显示警告。

## 10. 部署建议

- Streamlit Community Cloud / 内网服务器均可；
- 生产环境建议把 `.streamlit/secrets.toml` 用于密钥（本项目已 gitignore）；
- 项目本身不写数据库，可安全地将 `<data-root>` 以只读方式挂载/共享。

**EXIT**:
Ctrl + C 退出 Streamlit 服务，关闭浏览器标签页不会停止服务。

```powershell
Get-NetTCPConnection -LocalPort 8501 | Select-Object -ExpandProperty OwningProcess | Stop-Process
```

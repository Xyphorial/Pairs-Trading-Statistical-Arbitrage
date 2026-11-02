> **GitHub 公开版本提示：** 本仓库没有附带原始 `data/adjusted_close.csv`，因此直接运行 `--prices` 前需要自行获取有授权的行情数据。重新下载数据不能保证与历史研究结果完全一致。

# S&P 500 配对交易研究

## 本次真实数据结果

请求 503 个代码，成功获得 305 个，其中 275 个满足 formation 完整历史要求，测试 37,675 个股票组合。下载缺口见 data/download_audit.csv。

最终选择 MSCI / PODD：lookback=126，entry=2.5，exit=0.5，z_stop=4.5，最大持仓 20 个交易日。2023–2025 年扣费后累计收益 28.09%，年化 8.65%，Sharpe 0.91，最大回撤 4.49%。只有 7 笔测试交易，收益全部集中在一对股票。样本外协整 p=0.0737，未通过 5% 水平检验。因此这是有限证据下的研究结果，不能称为稳健有效策略。默认股票池是当前成分股，存在幸存者偏差。

先打开 **Pairs_Trading_Report.html** 查看图表和结果，再读本说明和 Walkthrough.ipynb。

## 先运行

Python 3.10+。在 VS Code 打开本文件所在的整个文件夹，在终端运行：

```bash
python -m pip install -r requirements.txt
python project.py --prices data/adjusted_close.csv
```

GitHub 公开版本不包含原始历史行情 CSV；重新下载当前股票池与行情：

```bash
python project.py --download
```

自备股票池（CSV 必须有 Symbol 列）：

```bash
python project.py --download --universe data/universe_snapshot.csv
```

离线软件演示：`python project.py --demo`。模拟结果单独放在 results/synthetic_check，不能作为投资业绩。
运行核算检查：`python -m unittest discover -s tests -v`。

## 项目到底在做什么

假设两只股票长期有稳定关系，但短期偏离。先回归

`log(P_Y) = alpha + beta × log(P_X) + residual`。

残差偏高时做空 Y、做多 X；残差偏低时做多 Y、做空 X，期待残差回到正常范围。相关性只衡量共同变动；协整检验考察某种价格组合是否平稳。高相关不保证协整，协整也不保证未来盈利。

### 数据与时间划分

- Formation：2015–2019，筛选股票、检验协整、排序与建立候选池。
- Validation：2020–2022，比较参数，按扣费后 Sharpe 选出设置。
- Test：2023–2025，参数冻结后做一次最终评估。

不是随机 train_test_split，因为市场有时间先后。测试阶段滚动估计只用当时已经看到的历史数据，不能用测试结果重新挑股票、改参数。formation 绩效是事后展示，不是独立预测证据；validation 已参与选参，也存在选择偏差。

### 1. 股票池和清洗

默认下载运行时的 S&P 500 成分股快照，使用 Yahoo 复权收盘价。股票代码中的点改为横线。只根据 formation 是否有完整且足够长的历史决定资格。不会根据测试期表现或完整性删除股票。若最终选中股票在评估期缺价，程序直接报错，要求处理数据，而不是悄悄跳过。

重要限制：当前成分股回溯历史存在幸存者偏差，漏掉已退市或被剔除股票。并非严格的历史时点 S&P 500 回测。复权价格采用调整单位近似总回报，不是逐笔真实成交/分红现金流账本。

### 2. 统计检验

对每只股票展示价格水平和一阶差分 ADF p 值；候选需满足水平未拒绝单位根、差分拒绝单位根的启发式 I(1) 筛选。这不是对 I(1) 的证明。

对所有合格股票的无序对做一次 Engle–Granger 检验，按字母顺序固定回归方向，不在两个方向中挑更小 p 值。原假设是“不协整”。不能把普通残差 ADF 的 p 值当作正确的协整 p 值；这里使用 statsmodels.coint。

同时输出 Benjamini–Hochberg q 值，默认 q <= 0.10，减轻多重检验问题。股票对存在依赖，FDR 不是对全部数据挖掘风险的保证。通过协整、I(1) 和正 beta 筛选后，按 formation p 值挑最多五对，股票不重复出现，避免重复持仓。没有合格策略就明确返回，不强行挑一个。

### 3. 滚动 hedge ratio 和 z-score

每个信号日，用它之前 126 或 252 个交易日估计 alpha、beta 与残差标准差；当天残差除以该标准差得到 z。均值与标准差来自同一窗口的同一回归，不混用不同时点回归的残差。

beta 是对数回归斜率，因此用作美元敞口比率。多 spread 时权重是 `[1, -beta] / (1+abs(beta))`，不是直接按 beta 股数买卖。只有正 beta 会开仓。

### 4. 交易与成本

收盘 t 计算信号，最早收盘 t+1 成交。从成交后才开始计收益。每对初始资金一份，组合初始等权；持仓期间保持调整后单位数，不每日再平衡，避免遗漏换仓费用。各 sleeve 独立复利，无仓位时现金收益为零。

达到 entry 开仓；均值回归到 exit、穿越均值、达到 z 止损或最大持有期时退出。z 止损是统计偏离止损，不是账户亏损百分比止损。最后一天强制平仓并收费。每日先按原持仓计盈亏，再执行延迟信号。

每条腿按实际交易金额收取单边佣金 2 bp + 滑点 3 bp，1 bp = 0.01%。空头按年 100 bp 收融券费，跨周末按日历日累计。成本假设是研究设定，不代表券商报价。测试额外展示 0、1、2、4 倍执行成本，融券费保持不变。

未建模：动态可借券与召回、真实 bid/ask、冲击、税、融资保证金约束、现金利息与真实成交失败。gross exposure 建仓时为资金的 100%，持仓后会随价格漂移。不是 beta 中性或严格美元中性的保证。

### 5. 参数选择

验证网格：p 阈值 0.01/0.05/0.10；回归窗口 126/252；entry 1.5/2/2.5；exit 0/0.5；z 止损 3.5/4.5；持有上限 20/60。共 144 组（部分因同一候选池而等价）。至少十笔验证期平仓交易，再以净 Sharpe 最大为准；并列优先较低换手。FDR 阈值不优化；交易成本不作为可压低的收益优化参数。

### 6. 输出怎么读

| 文件 | 内容 |
|---|---|
| cointegration_tests.csv | 全部 formation 股票对的 p/q、beta、相关性 |
| adf_diagnostics.csv | 单股 ADF 检查 |
| validation_sensitivity.csv | 全部参数及验证表现，可检查 entry/exit 敏感性 |
| performance.csv | formation / validation / test 对比 |
| test_cost_stress.csv | 执行成本压力测试 |
| *_trades.csv | 每笔开平仓日、方向、费用后盈亏、退出原因 |
| *_portfolio.csv | 每日净收益、净值与换手 |
| *_daily.csv | 各股票对的每日仓位与费用 |
| test_cointegration_diagnostic.csv | 样本外协整，仅事后诊断，不参与选参 |
| run_metadata.json | 设置、选中股票对、数据哈希与环境版本 |
| equity_curve.png | 整体组合在各区间的净值 |
| spread_z_*.png | 每对 spread、z、开平仓阈值 |
| candidate_pvalues.png | 候选股票对统计图 |

Sharpe = 日均净收益 / 日收益标准差 × sqrt(252)，现金无风险利率设为零。年化收益用复利；最大回撤从含初始净值 1 的历史高点计算。单次交易回报以该 pair sleeve 开仓前资金为分母，胜率按净盈亏；不是整个组合每笔贡献。年化换手是单边买卖绝对金额之和 / 期初净值，往返都会计入。日度记录可用于复核成本与收益。

## 面试/演讲时的核心结论

这个项目的价值是检验一个假说，而不是保证 Sharpe 很高。最终亏钱也应如实展示，并分析结构变化、成本、选参偏差与幸存者偏差。只有可靠的样本外结果支持时，才能声称策略有效。

## 参考

- 项目要求：用户提供的 Pairs Trading Strategy.pdf。
- statsmodels Engle–Granger：https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html
- Yahoo 下载与复权参数：https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- 默认成分股快照：https://github.com/datasets/s-and-p-500-companies

行情用于学习研究，重新分发和使用应遵守数据源条款。此项目提供研究代码与数据来源记录。

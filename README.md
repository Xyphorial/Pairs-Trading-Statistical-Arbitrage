# Statistical Arbitrage | S&P 500 Pairs Trading Research

**Python • Engle–Granger Cointegration • Backtesting • Out-of-Sample Evaluation**

A research pipeline evaluating mean-reversion pairs across a snapshot of S&P 500 stocks, with separate formation, validation and held-out test periods, explicit signal-execution lag and trading-cost accounting. The archived study selected the **MSCI / PODD** pair; this repo is the full-universe research project, **not** the separate XOM/CVX practice example.

## Research design

- **Formation (2015–2019):** screen available stocks for approximate I(1), test pairs with Engle–Granger and calculate Benjamini–Hochberg q-values across a large candidate set.
- **Validation (2020–2022):** tune rolling regression window, entry/exit z-scores, z-stop and maximum holding duration using net Sharpe, subject to minimum-trade criteria.
- **Test (2023–2025):** freeze pair/parameters and assess the selected strategy after explicit commissions, slippage and borrowing assumptions.
- Daily hedge ratio/signal formation uses only historical information available by the signal date. Execution is deferred, so an observation's close cannot generate same-close simulated execution.

## Archived test-period performance (historical research, not a forecast)

| Metric | Result |
|---|---:|
| Selected pair | MSCI / PODD |
| Total net return | 28.09% |
| Annualised net return | 8.65% |
| Net Sharpe | 0.91 |
| Maximum drawdown | 4.49% |
| Closed test trades | 7 |

**Important:** Only seven test-period trades support this result, and a post-test cointegration diagnostic gave **p = 0.0737**, not significant at 5%. Selection over thousands of pairs, survivorship bias from using a current S&P 500 constituent snapshot, fragile relationships, execution frictions and non-investable adjusted-price data make the apparent returns **weak evidence** of a durable edge.

## View findings

Open [research report](Pairs_Trading_Report.html) or [Chinese research walkthrough](README_中文.md). See [performance summary](results/real_data/performance.csv), [equity curve](results/real_data/equity_curve.png), [z-score chart](results/real_data/spread_z_MSCI_PODD.png), [cost stress test](results/real_data/test_cost_stress.csv) and [out-of-sample diagnostics](results/real_data/test_cointegration_diagnostic.csv).

## Run and reproduce

1. Install Python 3.10+ and project dependencies in `requirements.txt` (tested historical versions in `requirements-lock.txt`).
2. To run with your own licensed adjusted-price matrix, save it to `data/adjusted_close.csv` (see [data/README.md](data/README.md)) and invoke `project.py --prices data/adjusted_close.csv`.
3. Alternatively, invoke `project.py --download` to get a **new current-constituent snapshot** and Yahoo adjusted prices. This does not guarantee reproduction of the archived metrics and requires internet access.
4. Run core accounting tests: `python -m unittest discover -s tests -v`.
5. `Walkthrough.ipynb` and the included report explain and display the original results without needing the market-data CSV.

Market price observations are **not included** because third-party redistribution terms may apply. Outputs include selected trades, charts, aggregate diagnostics, dated backtest summaries, configuration and provenance. The original input can only be exactly replicated if the same licensed historical data and universe coverage are available.

## Limitations

This is a retrospective statistical test, not a production or live-trading track record. It omits some realistic market impact, borrow constraints and execution failures. Training/validation selection and multiple testing can substantially inflate expected out-of-sample performance. Current index members applied backwards create survivorship bias; use historical constituent data before treating this as an investable strategy.

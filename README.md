# Do YouTube Trading Rules Actually Work?

Trading educators on YouTube teach plenty of indicator rules: "MACD golden cross above zero", "DeMark 9 means enter", "short the premarket-low retest". I turned the rules from a long-running technical-analysis channel and a US day-trading channel into code, then tested them the same way a systematic desk would. Each rule runs on a fixed universe, over 10 years, against buy-and-hold, with a pass/fail bar fixed in advance.

**Short answer:** none of the 19 daily-indicator rules passes. A few cut drawdowns, but none beats buy-and-hold on risk-adjusted terms. Combining all of them into a composite score doesn't help either.

## 1. Daily indicator rules — `tech_score/`

**Setup**
- **Rules:** 19 modules in `tech_score/rules/`: MACD, RSI, KDJ, CCI, Bollinger, ATR/Keltner, Ichimoku, GMMA, BBI, MA, SAR, OBV, WVAD, volume-price, rolling VWAP, Fibonacci, DeMark 9/13, RSI/MACD divergence, and 61 TA-Lib candlestick patterns. Each module's docstring records the rule *as the presenter states it*, including their caveats ("don't trade it below the 30-min chart", "the histogram is optional"). Rules are coded literally, not tuned.
- **Universe:** 27 liquid instruments listed in `tech_score/universe.yml` across US index and sector ETFs, China A-share index ETFs, Hang Seng indices, gold, silver, oil and bitcoin. 26 produced results (the CSI 1000 ETF `512100.SS` did not return usable data; `rvwap` has 25).
- **Backtest:** daily bars, `yfinance period=10y` (so up to 10 years ending late 2026; shorter for younger ETFs such as IBIT 2024-, 588000.SS 2020-, XLC 2018-), long-only (enter on +1, exit on −1), 5 bp fees per side, `vectorbt`.
- **Pass bar, fixed in advance** (encoded in `tech_score/verify.py` before any rule was run): median Sharpe ≥ 0.3 **and** positive alpha vs buy-and-hold on ≥ 50% of tickers **and** a median drawdown no deeper than buy-and-hold's (median rule-DD / B&H-DD ≤ 1).

**Leaderboard** (top and bottom; full table in [`tech_score/results/leaderboard.csv`](tech_score/results/leaderboard.csv))

| Rule | Family | Median Sharpe | B&H Sharpe | % tickers beating B&H | Median DD ratio (rule/B&H) |
|---|---|---:|---:|---:|---:|
| Rolling VWAP | volume | 0.72 | 0.73 | 36% | 1.00 |
| ATR / Keltner | volatility | 0.68 | 0.72 | 38% | 1.00 |
| RSI | momentum | 0.61 | 0.72 | 27% | 0.91 |
| Bollinger | volatility | 0.61 | 0.72 | 4% | 0.90 |
| … | | | | | |
| MACD | momentum | 0.39 | 0.72 | 8% | 0.99 |
| Candlestick patterns | pattern | 0.30 | 0.72 | 4% | 1.00 |
| GMMA | trend | 0.31 | 0.72 | 4% | 0.75 |

- **0 of 19 rules pass.** None beats buy-and-hold on more than 38% of tickers.
- Several rules (KDJ, Fibonacci, Ichimoku, SAR) do reduce drawdown (DD ratio 0.59–0.73), mostly by sitting out of the market rather than by timing it.
- **Composite score** (`tech_score/composite.py`): family sub-scores → weighted 0–100 score → buy/sell on a vote threshold. Median Sharpe 0.53 vs 0.72 for buy-and-hold, beating B&H on 12% of tickers (`results/composite_verification_norgm.csv`). A regime-aware version (ADX / MA200 / 60-day return classifier in `regime.py`) improves it to 0.61 and 19% of tickers, still below buy-and-hold (`results/composite_verification_regime.csv`). Median max drawdown is −33% (−35% with regime) vs −42% for B&H.

```bash
python -m tech_score verify macd          # one rule, prints the verdict
python -m tech_score verify-all           # all rules → leaderboard.csv
python -m tech_score verify-composite          # add --no-regime for the plain composite
```

## 2. Intraday: premarket-low breakdown short — `backtest_pml_*.py`

This is a day-trading setup from a US options channel: on a red day, wait for a 5-minute close below the premarket low, short the retest, stop on a close back above it, trim at the prior low of day.

- `backtest_pml_retest.py`: the rule exactly as described, on 17 liquid names with 5-minute bars, one trade per ticker per day.
- `backtest_pml_vwap_v2.py`: adds the presenter's VWAP and 8-EMA rules, then runs a grid over level (PML vs min(PML, VWAP)), entry (touch vs confirmed close) and exit (trim vs EMA trail).
- `backtest_pml_options.py`: re-tests the signals with **real option prices** (nearest-expiry put, strike nearest entry) from moomoo historical bars. It compares a premium-at-risk exit with a stock-stop exit, using explicit spread assumptions.

The data limits are written into each script: yfinance only serves about 60 days of 5-minute bars, and moomoo keeps expired contracts for about 2–3 weeks. So this part is a small-sample case study, not a verdict.

## 3. Bottom signals on A-share indices

`analyze_weekly_bottom_signals.py` and the `scan_*` scripts test three "market bottom" signals on daily, weekly and monthly bars of Chinese indices: RSI/MACD double divergence, DeMark Setup 9, and up/down candle-colour sequences. The rules come from Chinese retail lore. [`candlestick_bottom_signals_research.md`](candlestick_bottom_signals_research.md) (in Chinese) reviews the existing evidence first: Bulkowski, QuantifiedStrategies, and A-share studies in *Physica A* and *SAGE Open*.

## Also included

`signal_generator.py` (multi-timeframe daily + weekly signal score), `backtest.py` (signal-driven backtest vs buy-and-hold) and `risk_manager.py` (ATR trailing stops, full and half Kelly sizing, portfolio exposure and VaR checks). These were my first version of a personal trading toolkit.

## Limitations

- In `tech_score`, every signal is computed from bars up to and including day *t* and filled at day *t*'s close (`vectorbt.from_signals` default). That is slightly optimistic for the rules, so it does not weaken the negative result; filling at the next open would be stricter.
- Sharpe ratios use vectorbt's default 365-day year, which inflates absolute values by about 20% versus a 252-trading-day convention. Rule and buy-and-hold are affected equally, so the comparison and the pass bar are unchanged.
- Results in `tech_score/results/` were produced with pandas 2.x. The code now uses `shift(1, fill_value=False)` for the boolean "first bar of a state" checks so it behaves identically on pandas 3.
- Nothing here is a statistical test: 19 rules on 26 correlated instruments is a small sample, and the rules were chosen because a presenter teaches them, not at random.
- The universe is ETFs and indices only. Single stocks and shorting are not tested.
- How I read each rule is a judgement call. The docstrings quote the source wording so the translation can be checked.
- Code comments are partly in Chinese.

## Setup

```bash
pip install -r requirements.txt   # TA-Lib needs the C library (brew install ta-lib)
pytest -q                          # 38 smoke tests; need only pandas/numpy
```

The intraday scripts additionally need `moomoo-api` and a running moomoo OpenD gateway (see each script's docstring). No market data is redistributed in this repo; the CSVs under `tech_score/results/` are derived statistics only.

Research only, not investment advice.

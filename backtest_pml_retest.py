"""
Premarket-Low (PML) breakdown + retest short — intraday backtest on the underlying.

Source: PB "How I Made $3555 Today Trading Options" (2025-04-04).
Rules as described in the video:
  1. Pick the weakest names of the day (red market -> look for weakest).
  2. Wait for a 5-min candle to CLOSE below the premarket low (PML).
  3. Wait for price to come back and retest the PML; enter (puts) at the PML.
     No retest -> no trade (he skipped MU for that reason).
  4. Stop: a 5-min close back above the PML.
  5. Trim at the prior low of day, ride the rest; never let a green trade go red.

Modelled here on the stock price (no options pricing):
  entry  = PML (limit short filled when a bar's high >= PML after the break)
  stop   = first 5-min close > PML  -> exit at that close
  T1     = RTH low made before entry -> cover 50%, stop on rest moves to entry
  rest   = EOD (15:55 bar close) or breakeven stop
Signals only counted 09:30 - ENTRY_CUTOFF; one trade per ticker per day.

Data: yfinance 5m with prepost (only the last ~60 calendar days are available).

Usage:
    python backtest_pml_retest.py            # download + run
    python backtest_pml_retest.py --cache f.pkl
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["SMCI", "HOOD", "MU", "AMD", "TGT", "NVDA", "ARM", "AMZN", "GOOGL",
           "COIN", "IWM", "AAPL", "TSLA", "MSFT", "QQQ", "SPY", "META"]
ENTRY_CUTOFF = "11:30"


def load(cache: Path | None) -> dict[str, pd.DataFrame]:
    if cache and cache.exists():
        return pd.read_pickle(cache)
    raw = yf.download(TICKERS, period="60d", interval="5m", prepost=True,
                      progress=False, auto_adjust=False, group_by="ticker")
    data = {t: raw[t].dropna(subset=["Close"]).tz_convert("America/New_York")
            for t in TICKERS}
    if cache:
        pd.to_pickle(data, cache)
    return data


def day_change_series(df: pd.DataFrame) -> pd.Series:
    """Per-bar % change vs previous day's RTH close."""
    rth = df.between_time("09:30", "15:55")
    prev_close = rth["Close"].groupby(rth.index.date).last().shift(1)
    pc = pd.Series(df.index.date, index=df.index).map(prev_close)
    return df["Close"] / pc - 1


def simulate_day(day: pd.DataFrame, chg: pd.Series, spy_chg: pd.Series):
    pm = day.between_time("04:00", "09:25")
    rth = day.between_time("09:30", "15:55")
    if len(pm) < 10 or len(rth) < 70:
        return None
    pml = pm["Low"].min()
    gap = chg.loc[pm.index[-1]]

    # 1) first 5m close below PML within the entry window
    window = rth.between_time("09:30", ENTRY_CUTOFF)
    below = window[window["Close"] < pml]
    if below.empty:
        return None
    t_break = below.index[0]
    # 2) retest: a later bar trades back up to PML
    after = window.loc[window.index > t_break]
    retest = after[after["High"] >= pml]
    if retest.empty:
        return {"gap": gap, "traded": False, "t_break": t_break}
    t_entry = retest.index[0]

    # context at entry
    pre = rth.loc[:t_entry].iloc[:-1]
    t1 = pre["Low"].min()
    tp = (rth["High"] + rth["Low"] + rth["Close"]) / 3
    vwap = ((tp * rth["Volume"]).cumsum() / rth["Volume"].cumsum()).loc[t_entry]
    rel = chg.loc[t_break] - spy_chg.get(t_break, float("nan"))

    entry = pml
    path = rth.loc[t_entry:]
    pnl, size, stop_px, trimmed, exit_reason = 0.0, 1.0, None, False, "eod"
    mfe = 0.0
    for i, (ts, bar) in enumerate(path.iterrows()):
        mfe = max(mfe, (entry - bar["Low"]) / entry)
        if trimmed and bar["High"] >= stop_px:          # breakeven on runner
            exit_reason = "be"
            break
        if bar["Close"] > pml and not trimmed:           # close-based stop
            pnl += size * (entry - bar["Close"]) / entry
            size, exit_reason = 0.0, "stop"
            break
        if i > 0 and not trimmed and bar["Low"] <= t1:   # trim 50% at prior LOD (not on entry bar: order unknown)
            pnl += 0.5 * (entry - t1) / entry
            size, trimmed, stop_px = 0.5, True, entry
    else:
        pnl += size * (entry - path["Close"].iloc[-1]) / entry
    if exit_reason == "be":
        size = 0.0  # runner flat at entry
    # "simple" variant: full size, exit at T1 / stop / EOD
    simple = None
    for i, (ts, bar) in enumerate(path.iterrows()):
        if bar["Close"] > pml:
            simple = (entry - bar["Close"]) / entry
            break
        if i > 0 and bar["Low"] <= t1:
            simple = (entry - t1) / entry
            break
    if simple is None:
        simple = (entry - path["Close"].iloc[-1]) / entry

    return {"gap": gap, "traded": True, "t_break": t_break, "t_entry": t_entry,
            "pml": pml, "t1": t1, "t1_dist": (entry - t1) / entry,
            "below_vwap": entry < vwap, "rel_spy": rel,
            "chg_at_break": chg.loc[t_break], "ret": pnl, "ret_simple": simple,
            "mfe": mfe, "exit": exit_reason}


def summarize(df: pd.DataFrame, col: str = "ret") -> dict:
    if df.empty:
        return {"n": 0}
    r = df[col]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    return {"n": len(r), "win%": round(100 * len(wins) / len(r), 1),
            "avg%": round(100 * r.mean(), 3), "avgW%": round(100 * wins.mean(), 3) if len(wins) else 0,
            "avgL%": round(100 * losses.mean(), 3) if len(losses) else 0,
            "PF": round(pf, 2), "sum%": round(100 * r.sum(), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write trades CSV")
    args = ap.parse_args()

    data = load(args.cache)
    spy_chg = day_change_series(data["SPY"])
    rows = []
    for t, df in data.items():
        chg = day_change_series(df)
        for d, day in df.groupby(df.index.date):
            res = simulate_day(day, chg, spy_chg)
            if res:
                rows.append({"ticker": t, "date": d, **res})
    sig = pd.DataFrame(rows)
    days = sorted({d for df in data.values() for d in df.index.date})
    print(f"Period {days[0]} -> {days[-1]}  ({len(days)} sessions), {len(TICKERS)} tickers")
    print(f"Breaks below PML (5m close, by {ENTRY_CUTOFF}): {len(sig)}; "
          f"with retest/entry: {sig['traded'].sum()}")

    tr = sig[sig["traded"]].copy()
    # weakness rank among the 17 names by premarket gap that day
    all_gaps = pd.DataFrame({t: day_change_series(df).between_time("09:25", "09:25")
                             .groupby(lambda x: x.date()).last() for t, df in data.items()})
    rank = all_gaps.rank(axis=1)  # 1 = weakest gap
    tr["gap_rank"] = [rank.at[d, t] if d in rank.index else None
                      for d, t in zip(tr["date"], tr["ticker"])]

    filters = {
        "all retest entries": tr,
        "rel vs SPY < 0 at break": tr[tr["rel_spy"] < 0],
        "rel vs SPY < -1%": tr[tr["rel_spy"] < -0.01],
        "stock down >2% at break": tr[tr["chg_at_break"] < -0.02],
        "bottom-5 premarket gap": tr[tr["gap_rank"] <= 5],
        "rel<0 & below VWAP": tr[(tr["rel_spy"] < 0) & tr["below_vwap"]],
        "SPY red at break & rel<0": tr[(tr["rel_spy"] < 0) & (tr["chg_at_break"] - tr["rel_spy"] < 0)],
    }
    out = []
    for name, f in filters.items():
        out.append({"filter": name, **{f"trim_{k}": v for k, v in summarize(f, "ret").items()},
                    **{f"simple_{k}": v for k, v in summarize(f, "ret_simple").items() if k != "n"}})
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== Underlying % return per trade (short from PML) ==")
    print(pd.DataFrame(out).set_index("filter").to_string())
    print("\nExit mix (all):", tr["exit"].value_counts().to_dict())
    print(f"Median T1 distance: {100*tr['t1_dist'].median():.2f}%  median MFE: {100*tr['mfe'].median():.2f}%")

    # No-retest comparison: short at the first 5m close below PML
    print(f"\nNo-retest breaks (skipped, like MU on the video day): {(~sig['traded']).sum()}")
    if args.out:
        tr.to_csv(args.out, index=False)
        print(f"Trades -> {args.out}")


if __name__ == "__main__":
    main()

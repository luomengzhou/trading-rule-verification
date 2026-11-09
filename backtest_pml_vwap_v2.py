"""
v2 of the PML breakdown short, adding rules from PB's "The ONLY 2 Indicators" video:
  - Never take puts above VWAP.
  - If VWAP is below the premarket low, trade the VWAP retest, not the PML retest
    -> retest level L = min(PML, VWAP) at each bar.
  - Stop: 5-min close back above L.
  - Runners: hold while the 5-min stays under the 8 EMA (close above 8 EMA -> exit).
Plus a stricter "confirmed" entry: the retest bar must tag L and CLOSE back below it,
entry at that bar's close (vs v1 "touch": limit fill at L).

Grid:
  level : pml (v1) | min(pml,vwap) (v2)
  entry : touch (fill at L) | confirm (retest bar closes below L, fill at close)
  exit  : trim  = 50% at prior RTH low, runner exits on close > EMA8 or back to entry
          ema   = full size; stop close > L; once EMA8 < entry, exit on close > EMA8
Underlying % returns only, no option pricing. Shared data/cache with backtest_pml_retest.py.

Usage:
    python backtest_pml_vwap_v2.py --cache bars5m.pkl [--out trades.csv]
"""
import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from backtest_pml_retest import ENTRY_CUTOFF, day_change_series, load, summarize

TOL = 0.0005  # a high within 0.05% of L counts as a tag


def simulate(day, ema8, chg, spy_chg, level_mode, entry_mode, exit_mode):
    pm = day.between_time("04:00", "09:25")
    rth = day.between_time("09:30", "15:55")
    if len(pm) < 10 or len(rth) < 70:
        return None
    pml = pm["Low"].min()
    tp = (rth["High"] + rth["Low"] + rth["Close"]) / 3
    vwap = (tp * rth["Volume"]).cumsum() / rth["Volume"].cumsum()
    lvl = pd.Series(pml, index=rth.index) if level_mode == "pml" else vwap.clip(upper=pml)

    win = rth.between_time("09:30", ENTRY_CUTOFF)
    below = win[(win["Close"] < pml) & (win["Close"] < vwap.loc[win.index])]
    if below.empty:
        return None
    t_break = below.index[0]

    # find entry
    t_entry = entry = None
    for ts, bar in win.loc[win.index > t_break].iterrows():
        L = lvl[ts]
        if bar["High"] >= L * (1 - TOL):
            if entry_mode == "touch":
                t_entry, entry = ts, L
                break
            if bar["Close"] < L and bar["Close"] < vwap[ts]:
                t_entry, entry = ts, bar["Close"]
                break
    if t_entry is None:
        return None

    t1 = rth.loc[:t_entry]["Low"].iloc[:-1].min()
    path = rth.loc[t_entry:]
    pnl, size, trimmed, reason = 0.0, 1.0, False, "eod"
    for i, (ts, bar) in enumerate(path.iterrows()):
        c, e8 = bar["Close"], ema8[ts]
        if not trimmed and c > lvl[ts]:                       # initial stop
            pnl += size * (entry - c) / entry
            size, reason = 0.0, "stop"
            break
        if exit_mode == "trim":
            if trimmed and bar["High"] >= entry:              # runner back to entry
                size, reason = 0.0, "be"
                break
            if trimmed and c > e8:                            # runner trail
                pnl += size * (entry - c) / entry
                size, reason = 0.0, "ema"
                break
            if i > 0 and not trimmed and bar["Low"] <= t1:    # trim at prior LOD
                pnl += 0.5 * (entry - t1) / entry
                size, trimmed = 0.5, True
        else:  # ema
            if i > 0 and e8 < entry and c > e8:
                pnl += size * (entry - c) / entry
                size, reason = 0.0, "ema"
                break
    if size:
        pnl += size * (entry - path["Close"].iloc[-1]) / entry

    return {"t_break": t_break, "t_entry": t_entry, "entry": entry, "pml": pml,
            "vwap_lvl": lvl[t_entry] < pml - 1e-9,
            "confluence": abs(vwap[t_entry] - pml) / pml < 0.002,
            "rel_spy": chg[t_break] - spy_chg.get(t_break, float("nan")),
            "spy_chg": spy_chg.get(t_break, float("nan")),
            "ret": pnl, "exit": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.cache)
    spy_chg = day_change_series(data["SPY"])
    pre = {t: (df, df["Close"].ewm(span=8, adjust=False).mean(), day_change_series(df))
           for t, df in data.items()}

    rows = []
    for level_mode, entry_mode, exit_mode in product(["pml", "min_pml_vwap"],
                                                     ["touch", "confirm"], ["trim", "ema"]):
        for t, (df, ema8, chg) in pre.items():
            for d, day in df.groupby(df.index.date):
                r = simulate(day, ema8, chg, spy_chg, level_mode, entry_mode, exit_mode)
                if r:
                    rows.append({"level": level_mode, "entry_mode": entry_mode,
                                 "exit_mode": exit_mode, "ticker": t, "date": d, **r})
    tr = pd.DataFrame(rows)

    filters = {
        "all": lambda x: x,
        "rel<-1%": lambda x: x[x.rel_spy < -0.01],
        "SPY red & rel<0": lambda x: x[(x.spy_chg < 0) & (x.rel_spy < 0)],
        "SPY < -1%": lambda x: x[x.spy_chg < -0.01],
        "VWAP~PML confluence": lambda x: x[x.confluence],
    }
    out = []
    for (lv, en, ex), g in tr.groupby(["level", "entry_mode", "exit_mode"]):
        for fname, f in filters.items():
            s = summarize(f(g))
            if s["n"]:
                out.append({"level": lv, "entry": en, "exit": ex, "filter": fname, **s})
    pd.set_option("display.width", 250, "display.max_rows", 200)
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    if args.out:
        tr.to_csv(args.out, index=False)
        print(f"Trades -> {args.out}")


if __name__ == "__main__":
    main()

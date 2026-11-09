"""
Real-option test of the PML retest put: premium-at-risk vs stock-stop exits.

Signals: backtest_pml_vwap_v2 with level=pml, entry=touch (fill near PML, below VWAP).
Contract: put with the nearest expiry on/after the signal day, strike nearest entry.
Risk = premium. Reward = premium x N. Both exit rules take profit at +N x premium:
  nostop : hold until the take-profit or the 15:55 bar (no overnight)
  stop   : same, but sell when a 5-min stock close is back above PML

Data limits:
  - moomoo keeps expired contracts only ~2-3 weeks, so only recent signals are testable.
  - Bars are trade prices, no bid/ask. Spread is an assumption (SPREADS, round trip).
  - Entry price is the option's entry-bar close, moved to the PML stock price with
    delta 0.5. Bars without trades carry the last trade forward.

Usage:
    python backtest_pml_options.py --cache bars5m.pkl --opt-cache opt.pkl
"""
import argparse
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from backtest_pml_retest import day_change_series
from backtest_pml_vwap_v2 import simulate

SPREADS = [0.0, 0.03, 0.06]
MULTIPLES = [1, 2, 3]
DELTA = 0.5
STALE_BARS = 3  # entry needs a trade within this many bars
ETFS = {"SPY", "QQQ", "IWM"}


def signals(data):
    spy_chg = day_change_series(data["SPY"])
    rows = []
    for t, df in data.items():
        ema8, chg = df["Close"].ewm(span=8, adjust=False).mean(), day_change_series(df)
        for d, day in df.groupby(df.index.date):
            r = simulate(day, ema8, chg, spy_chg, "pml", "touch", "trim")
            if r:
                rows.append({"ticker": t, "date": d, **r})
    return pd.DataFrame(rows)


def strikes(px):
    grid = {round(px / s) * s for s in (0.5, 1, 2.5, 5, 10)}
    return sorted(grid, key=lambda k: abs(k - px))[:5]


def expiries(d, ticker):
    """Dates to try, nearest first. Single names: Friday weeklies. ETFs: any weekday."""
    days = [d + timedelta(n) for n in range(0, 8)]
    days = [x for x in days if x.weekday() < 5]
    return days if ticker in ETFS else [x for x in days if x.weekday() == 4]


def fetch(ctx, code, day):
    from moomoo import KLType, RET_OK, Session

    time.sleep(0.6)  # request_history_kline: 60 calls / 30s
    ret, df, _ = ctx.request_history_kline(code, start=str(day), end=str(day), ktype=KLType.K_5M,
                                           session=Session.ALL, max_count=500)
    if ret != RET_OK:
        return None
    df = df.set_index(pd.to_datetime(df["time_key"]) - pd.Timedelta(minutes=5))  # END -> START
    return df[["open", "high", "low", "close", "volume"]].tz_localize("America/New_York")


def find_contract(ctx, cache, s):
    """Return (code, expiry, strike, bars for the signal day), or None."""
    key = (s.ticker, s.date)
    if key in cache:
        return cache[key]
    found = None
    for exp in expiries(s.date, s.ticker):
        for k in strikes(s.entry):
            code = f"US.{s.ticker}{exp:%y%m%d}P{round(k * 1000)}"
            bars = fetch(ctx, code, s.date)
            if bars is not None and len(bars):
                found = (code, exp, k, bars)
                break
        if found:
            break
    cache[key] = found
    return found


def trade(s, stock, opt, multiple, spread, use_stop):
    rth = stock.between_time("09:30", "15:55")
    path = rth.loc[s.t_entry:]
    o = opt.reindex(rth.index)
    o["close"] = o["close"].ffill()
    traded = opt.index[(opt["volume"] > 0) & (opt.index <= s.t_entry)]
    if not len(traded) or (s.t_entry - traded[-1]) > pd.Timedelta(minutes=5 * STALE_BARS):
        return None
    mid = o.at[s.t_entry, "close"] + DELTA * (path["Close"].iloc[0] - s.entry)
    if mid <= 0:
        return None
    buy = mid * (1 + spread / 2)
    target = buy * (1 + multiple)
    for i, (ts, bar) in enumerate(path.iterrows()):
        if use_stop and bar["Close"] > s.pml:
            return o.at[ts, "close"] * (1 - spread / 2) / buy - 1, "stop"
        high = o.at[ts, "high"]
        if i > 0 and pd.notna(high) and o.at[ts, "volume"] > 0 and high * (1 - spread / 2) >= target:
            return multiple, "target"
    return o["close"].iloc[-1] * (1 - spread / 2) / buy - 1, "eod"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="stock 5m bars pickle")
    ap.add_argument("--opt-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--since", default="2026-10-12")
    args = ap.parse_args()

    data = pd.read_pickle(args.cache)
    sig = signals(data)
    sig = sig[sig["date"].astype(str) >= args.since].reset_index(drop=True)
    cache = pd.read_pickle(args.opt_cache) if args.opt_cache.exists() else {}

    from moomoo import OpenQuoteContext  # needs OpenD running; pip install moomoo-api

    ctx = OpenQuoteContext(host="127.0.0.1", port=11111, ai_type=1)
    try:
        contracts = [find_contract(ctx, cache, s) for s in sig.itertuples()]
    finally:
        ctx.close()
        pd.to_pickle(cache, args.opt_cache)

    rows = []
    for s, c in zip(sig.itertuples(), contracts):
        if c is None:
            continue
        code, exp, k, opt = c
        day = data[s.ticker]
        day = day[day.index.date == s.date]
        for m in MULTIPLES:
            for sp in SPREADS:
                for use_stop in (False, True):
                    r = trade(s, day, opt, m, sp, use_stop)
                    if r:
                        rows.append({"ticker": s.ticker, "date": s.date, "code": code,
                                     "dte": (exp - s.date).days, "strike": k, "rel_spy": s.rel_spy,
                                     "N": m, "spread": sp, "rule": "stop" if use_stop else "nostop",
                                     "ret": r[0], "exit": r[1]})
    tr = pd.DataFrame(rows)
    base = tr[(tr.N == 1) & (tr.spread == 0) & (tr.rule == "nostop")]
    print(f"signals {len(sig)}, with contract {sum(c is not None for c in contracts)}, "
          f"tradable {len(base)}, sessions {base.date.nunique()}, median DTE {base.dte.median():.0f}")

    def report(df, label):
        g = df.groupby(["rule", "N", "spread"])["ret"]
        out = g.agg(n="size", win=lambda r: round(100 * (r > 0).mean()),
                    avg=lambda r: round(r.mean(), 3), total=lambda r: round(r.sum(), 2))
        out["breakeven"] = [round(100 / (1 + n)) for n in out.index.get_level_values("N")]
        print(f"\n== {label} (returns in multiples of premium) ==")
        print(out.to_string())

    pd.set_option("display.width", 200, "display.max_rows", 200)
    report(tr, "all signals")
    report(tr[tr.rel_spy < -0.01], "rel vs SPY < -1%")
    if args.out:
        tr.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

"""Download 5m US bars from moomoo OpenD into the cache used by the PML backtests.

Output: pickle of {ticker: DataFrame[Open, High, Low, Close, Volume]}, indexed by
bar START time in America/New_York, 04:00-19:55 only, forward-adjusted (splits and
dividends). This is the format backtest_pml_retest.load() reads via --cache.

Requires OpenD running and logged in on 127.0.0.1:11111. Quote context only: this
script never opens a trade context.

Tickers already in the cache are skipped, so an interrupted run can be restarted.
Forward-adjusted history changes on each new dividend/split, so bars are never
appended: delete the cache and rerun to refresh.

Usage:
    ~/.venvs/moomoo/bin/python fetch_bars5m.py [--out bars5m.pkl] [--tickers NVDA AMD]
    ~/.venvs/moomoo/bin/python backtest_pml_retest.py --cache bars5m.pkl
"""
import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd
from moomoo import AuType, KLType, OpenQuoteContext, RET_OK, Session

from backtest_pml_retest import TICKERS

START = "2018-10-01"     # earliest 5m RTH history; real pre-market starts ~2020
COLUMNS = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
           "volume": "Volume"}


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert moomoo bars to the yfinance-style frame the backtests expect."""
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())]
    # moomoo stamps a bar with its END time; the backtests use START time.
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["time_key"])
                                - pd.Timedelta(minutes=5), name="Datetime")
    # Volume 0 bars are flat placeholders (all pre-market before ~2020). PML must
    # not see them. The clock filter drops overnight bars, including the DST-
    # ambiguous 01:00 hour, so tz_localize below cannot fail.
    df = df[(df.index.dayofweek < 5) & (df["Volume"] > 0)]
    return df.between_time("04:00", "19:55").sort_index().tz_localize("America/New_York")


def fetch(ctx: OpenQuoteContext, ticker: str) -> pd.DataFrame:
    pages, key = [], None
    while True:
        ret, page, key = ctx.request_history_kline(
            f"US.{ticker}", start=START, end=str(date.today()), ktype=KLType.K_5M,
            autype=AuType.QFQ, session=Session.ALL, max_count=1000, page_req_key=key)
        if ret != RET_OK:
            raise RuntimeError(f"{ticker}: {page}")
        pages.append(page)
        if key is None:
            return normalize(pd.concat(pages, ignore_index=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bars5m.pkl"))
    ap.add_argument("--tickers", nargs="+", default=TICKERS)
    args = ap.parse_args()

    data = pd.read_pickle(args.out) if args.out.exists() else {}
    tmp = args.out.with_suffix(".tmp")
    ctx = OpenQuoteContext(host="127.0.0.1", port=11111, ai_type=1)
    try:
        for ticker in args.tickers:
            if ticker in data:
                print(f"{ticker}: cached, skipped", flush=True)
                continue
            t0 = time.time()
            data[ticker] = df = fetch(ctx, ticker)
            pd.to_pickle(data, tmp)
            tmp.replace(args.out)
            print(f"{ticker}: {len(df):,} bars {df.index[0]} -> {df.index[-1]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        ctx.close()


if __name__ == "__main__":
    main()

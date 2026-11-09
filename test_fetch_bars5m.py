"""运行：~/.venvs/moomoo/bin/python -m unittest test_fetch_bars5m"""

import unittest

import pandas as pd

from fetch_bars5m import normalize


def raw(rows):
    return pd.DataFrame(rows, columns=["time_key", "open", "high", "low", "close", "volume"])


class NormalizeTest(unittest.TestCase):
    def test_stamps_bar_start_in_new_york_time(self):
        df = normalize(raw([("2026-11-05 09:35:00", 1, 2, 0.5, 1.5, 100)]))
        self.assertEqual(str(df.index[0]), "2026-11-05 09:30:00-05:00")
        self.assertEqual(df.columns.tolist(), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(df["Low"].iloc[0], 0.5)

    def test_keeps_first_and_last_extended_hours_bars(self):
        df = normalize(raw([("2026-11-05 04:05:00", 1, 1, 1, 1, 10),
                            ("2026-11-05 20:00:00", 1, 1, 1, 1, 10)]))
        self.assertEqual([t.strftime("%H:%M") for t in df.index], ["04:00", "19:55"])

    def test_drops_overnight_bars(self):
        df = normalize(raw([("2026-11-05 00:05:00", 1, 1, 1, 1, 10),
                            ("2026-11-05 04:00:00", 1, 1, 1, 1, 10),
                            ("2026-11-05 20:05:00", 1, 1, 1, 1, 10)]))
        self.assertTrue(df.empty)

    def test_drops_zero_volume_placeholders(self):
        df = normalize(raw([("2019-06-10 08:05:00", 9, 9, 9, 9, 0),
                            ("2019-06-10 09:35:00", 1, 2, 1, 2, 10)]))
        self.assertEqual(len(df), 1)
        self.assertEqual(df["Low"].iloc[0], 1)

    def test_drops_weekend_bars(self):
        self.assertTrue(normalize(raw([("2026-11-08 09:35:00", 1, 1, 1, 1, 10)])).empty)

    def test_dst_fall_back_night_does_not_raise(self):
        df = normalize(raw([("2026-11-01 01:30:00", 1, 1, 1, 1, 10),
                            ("2026-11-02 09:35:00", 1, 1, 1, 1, 10)]))
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()

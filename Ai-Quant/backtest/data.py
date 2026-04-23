"""
Data loading: OANDA API + CSV cache fallback.

The DataLoader attempts to:
1. Fetch from OANDA v20 API (live)
2. Fall back to CSV cache if API fails (offline development)
3. Validate OHLCV completeness (no gaps, proper relationships)

For Phase 3, we use mock/cached data. Live OANDA integration
comes later (Phase 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataLoadError(Exception):
    """Raised when data cannot be loaded from any source."""
    pass


class DataValidator:
    """Validates OHLCV DataFrames for quality."""

    @staticmethod
    def validate(df: pd.DataFrame, pair: str) -> None:
        """
        Raises DataLoadError if:
        - Missing required columns (open, high, low, close, volume)
        - High < max(open, close)
        - Low > min(open, close)
        - Gaps in daily sequence
        - NaN or zero volume rows
        """
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            raise DataLoadError(f"{pair}: missing columns. Required: {required}")

        # OHLC relationships
        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise DataLoadError(f"{pair}: high < max(open, close)")
        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise DataLoadError(f"{pair}: low > min(open, close)")

        # No NaNs
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise DataLoadError(f"{pair}: contains NaN values")

        # No zero volume (optional — some exchanges allow it)
        if (df["volume"] <= 0).any():
            raise DataLoadError(f"{pair}: contains zero-volume bars")

        # Daily continuity: skip weekends/holidays but check no multi-day gaps
        if isinstance(df.index, pd.DatetimeIndex):
            diffs = df.index.to_series().diff()
            # Typical gaps: 1 day, 3 days (Fri-Mon), 2 days (holiday Fri-Mon)
            # Anything > 3 days is suspicious
            max_gap = diffs.max()
            if max_gap > pd.Timedelta(days=3):
                raise DataLoadError(f"{pair}: suspicious gap {max_gap} in date sequence")


class DataLoader:
    """
    Loads OHLCV daily candle data for backtesting.
    Phase 3: uses CSV cache or generates synthetic data.
    Phase 6: integrates live OANDA API.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.validator = DataValidator()

    def load(
        self,
        pair: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for pair.
        Returns DataFrame with index=datetime (UTC), columns=[open,high,low,close,volume].
        """
        # Try cache first
        cache_file = self.cache_dir / f"{pair}.csv"
        if cache_file.exists():
            df = self._load_csv(cache_file)
            self.validator.validate(df, pair)
            if start_date or end_date:
                df = df.loc[start_date:end_date]
            return df

        # Fallback: generate synthetic data (for testing/demo)
        df = self._generate_synthetic(pair, start_date or datetime(2015, 1, 1), end_date or datetime.utcnow())
        self.validator.validate(df, pair)
        return df

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load CSV cache file. Expected columns: datetime, open, high, low, close, volume."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df

    @staticmethod
    def _generate_synthetic(pair: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Generate synthetic daily OHLCV data for testing.
        Uses random walk with realistic volatility.
        NOT for production backtesting — use real data only.
        """
        # Annual volatility by pair (realistic proxies)
        annual_vol = {
            "EUR_USD": 0.09,
            "GBP_USD": 0.11,
            "USD_JPY": 0.08,
            "AUD_USD": 0.12,
            "USD_CAD": 0.10,
        }
        spot_price = {
            "EUR_USD": 1.10,
            "GBP_USD": 1.27,
            "USD_JPY": 110.0,
            "AUD_USD": 0.75,
            "USD_CAD": 1.25,
        }

        vol = annual_vol.get(pair, 0.10)
        daily_vol = vol / np.sqrt(252)
        start_price = spot_price.get(pair, 1.0)

        # Daily returns
        days = pd.bdate_range(start=start, end=end, freq="B")
        returns = np.random.normal(0.0001, daily_vol, len(days))
        close_prices = start_price * np.exp(np.cumsum(returns))

        # Construct OHLC (simplified: assume intraday range ±0.5× daily volatility)
        intraday_noise = np.random.normal(0, daily_vol * 0.5, len(days))
        opens = close_prices.shift(1).fillna(start_price)
        highs = np.maximum.reduce([opens, closes := close_prices, opens + abs(intraday_noise)])
        lows = np.minimum.reduce([opens, closes, opens - abs(intraday_noise)])
        volume = np.random.uniform(100_000, 500_000, len(days))

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close_prices,
            "volume": volume,
        }, index=days)

        return df


def load_all_pairs(
    pairs: list[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    cache_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load data for all pairs in parallel dict.
    """
    loader = DataLoader(cache_dir)
    data = {}
    for pair in pairs:
        data[pair] = loader.load(pair, start_date, end_date)
    return data

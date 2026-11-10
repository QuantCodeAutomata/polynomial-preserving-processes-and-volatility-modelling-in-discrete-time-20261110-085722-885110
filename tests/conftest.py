"""Shared pytest fixtures: market-data marker registration and a cached
reduced slice of S&P 500 returns for market-data-dependent tests."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "market_data: requires DATALOADER_API_KEY/MASSIVE_API_KEY and network access")


@pytest.fixture(scope="session")
def sp500_slice(request):
    """A short cached slice of S&P 500 (SPY) daily bars for market_data tests."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")
    cache = request.config.cache
    key = "sp500_slice_2022_2023"
    cached = cache.get(key, None)
    if cached is not None:
        import pandas as pd
        return pd.DataFrame(cached)
    from data.data_loader import load_ohlcv

    bars, meta = load_ohlcv(["SPY"], "2022-01-01", "2023-08-31", timeframe="1d")
    cache.set(key, bars.to_dict(orient="list"))
    return bars

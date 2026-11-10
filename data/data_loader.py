"""OHLCV bars from the QuantX Dataloader REST API, one tidy table per load."""

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15  # tickers per request; batches are split by ticker, never by date
MAX_PARALLEL = 4  # concurrent requests
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]  # the Massive key goes to fetch-ohlcvs only
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(
            method, API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    """The wide table of one answer: inline rows, or one CSV export when stored."""
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",  # a file label: the export holds every ticker
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    # A signed link: plain GET without credentials; never log or store it.
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    """Wide columns (one ticker: open..; several: AAPL_open..; stored CSV export:
    aapl_open.. by alias) to rows of timestamp, date, ticker, open..volume."""
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))  # an absent bar is not a zero bar
    bars = pd.concat(frames, ignore_index=True)
    # Daily bars are stamped at midnight New York time, stored in UTC.
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
        "normalization_warnings": answer.get("normalization_warnings"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Bars for ``tickers`` over the full inclusive window, and load metadata.

    Returns rows of timestamp (UTC), date, ticker, open, high, low, close,
    volume, sorted by ticker and time, plus metadata with every ticker's
    ``ticker_details`` entry. Tickers that failed are listed under ``failed``;
    the load fails only when no ticker loads.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
        "batches": [notes for _, _, notes in results],
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def _massive_headers() -> dict[str, str]:
    load_dotenv()
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not massive_key:
        raise DataloaderError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {massive_key}"}


def _massive_get(path: str, params: dict) -> dict:
    """Direct Massive REST call (non-OHLCV data not served by the Dataloader)."""
    response = requests.get("https://api.massive.com" + path, params=params,
                             headers=_massive_headers(), timeout=60)
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(f"{path}: HTTP {response.status_code}: {error}")
    return response.json()


def load_risk_free_rate(start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily 3-month U.S. Treasury yield from Massive's Treasury Yields endpoint
    (``GET /fed/v1/treasury-yields``), the Federal Reserve H.15 series used in
    Section 4.1 to compute daily excess log returns.

    Returns a DataFrame with columns ``date`` and ``yield_3_month`` (percent,
    annualized, investment basis), one row per available calendar date.
    """
    key = hashlib.sha256(f"rf_{start_date}_{end_date}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"rf_{key}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    rows: list[dict] = []
    params = {
        "date.gte": start_date, "date.lte": end_date,
        "limit": 50000, "sort": "date.asc",
        "fields": "date,yield_3_month",
    }
    next_url = None
    while True:
        if next_url:
            response = requests.get(next_url, headers=_massive_headers(), timeout=60)
            response.raise_for_status()
            payload = response.json()
        else:
            payload = _massive_get("/fed/v1/treasury-yields", params)
        rows.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        if not next_url:
            break
    if not rows:
        raise DataloaderError("no treasury-yield rows returned for the requested window")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["yield_3_month"]).sort_values("date", ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_option_chain_snapshot(underlying: str = "SPY", use_cache: bool = True) -> pd.DataFrame:
    """Current option chain for ``underlying`` from Massive's Option Chain
    Snapshot (``GET /v3/snapshot/options/{underlyingAsset}``).

    The endpoint serves a live cross-section, not a point-in-time history: it
    is used to obtain one genuine market option cross-section for Section 4.3's
    IVRMSE calibration machinery, not the paper's 1,443-week OptionMetrics panel
    (see "Not reproduced" in results/RESULTS.md). Returns one row per contract
    with columns: contract_type, strike_price, expiration_date, implied_volatility,
    open_interest, volume, bid, ask, underlying_price.
    """
    key = hashlib.sha256(f"chain_{underlying}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"chain_{key}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["expiration_date"])
    rows: list[dict] = []
    next_url = None
    params = {"limit": 250}
    while True:
        if next_url:
            response = requests.get(next_url, headers=_massive_headers(), timeout=60)
            response.raise_for_status()
            payload = response.json()
        else:
            payload = _massive_get(f"/v3/snapshot/options/{underlying}", params)
        for item in payload.get("results", []):
            details = item.get("details", {})
            quote = item.get("last_quote", {}) or {}
            greeks = item.get("greeks", {}) or {}
            rows.append({
                "contract_type": details.get("contract_type"),
                "strike_price": details.get("strike_price"),
                "expiration_date": details.get("expiration_date"),
                "implied_volatility": item.get("implied_volatility"),
                "open_interest": item.get("open_interest"),
                "volume": (item.get("day") or {}).get("volume"),
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "underlying_price": (item.get("underlying_asset") or {}).get("price"),
                "delta": greeks.get("delta"),
            })
        next_url = payload.get("next_url")
        if not next_url:
            break
    if not rows:
        raise DataloaderError(f"no option-chain rows returned for {underlying}")
    df = pd.DataFrame(rows)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

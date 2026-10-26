import time
import requests
import pandas as pd
import yfinance as yf

import agent

HTTP = requests.Session()

_ORIG_BINANCE_KLINES = agent.binance_klines
_ORIG_LIVE_PRICE = agent.get_live_price
_ORIG_TWELVE_REQUEST = agent.twelve_request
_ORIG_YAHOO_FRAMES = agent.yahoo_frames
_ORIG_SEND_TELEGRAM = agent.send_telegram
_ORIG_PROCESS_WATCH = agent.process_watch

CRYPTO_YAHOO = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
}

OKX_BAR = {
    "1m": "1m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
}


def retry(label, fn, attempts=3, sleep_seconds=1.0):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            print(f"{label} attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                time.sleep(sleep_seconds * attempt)
    raise last_error


def okx_inst_id(symbol):
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USDT"
    return symbol


def okx_rows_to_df(data, interval):
    minutes = {"1m": 1, "15m": 15, "1h": 60, "4h": 240}[interval]
    rows = []
    for candle in data:
        if len(candle) < 6:
            continue
        ts = pd.to_datetime(int(candle[0]), unit="ms", utc=True)
        rows.append({
            "time": ts,
            "end": ts + pd.Timedelta(minutes=minutes),
            "o": float(candle[1]),
            "h": float(candle[2]),
            "l": float(candle[3]),
            "c": float(candle[4]),
            "v": float(candle[5]),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


def okx_klines(symbol, interval, limit=300, start_ms=None):
    inst_id = okx_inst_id(symbol)
    bar = OKX_BAR[interval]

    if start_ms is None:
        response = HTTP.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": inst_id, "bar": bar, "limit": min(int(limit), 300)},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX error: {payload}")
        return okx_rows_to_df(payload.get("data") or [], interval)

    cursor = int(start_ms)
    all_rows = []
    after = None

    for _ in range(20):
        params = {"instId": inst_id, "bar": bar, "limit": 300}
        if after is not None:
            params["after"] = str(after)

        response = HTTP.get(
            "https://www.okx.com/api/v5/market/history-candles",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX history error: {payload}")

        batch = payload.get("data") or []
        if not batch:
            break

        all_rows.extend(batch)
        timestamps = [int(row[0]) for row in batch if row]
        if not timestamps:
            break

        earliest = min(timestamps)
        if earliest <= cursor:
            break
        if after == earliest:
            break
        after = earliest

    df = okx_rows_to_df(all_rows, interval)
    if df.empty:
        return df

    cursor_ts = pd.to_datetime(cursor, unit="ms", utc=True)
    return df[df.index >= cursor_ts].copy()


def yahoo_crypto_klines(symbol, interval, limit=300, start_ms=None):
    ticker = CRYPTO_YAHOO[symbol]

    if interval == "1m":
        yf_interval, period, minutes = "1m", "7d", 1
    elif interval == "15m":
        yf_interval, period, minutes = "15m", "5d", 15
    elif interval == "1h":
        yf_interval, period, minutes = "1h", "60d", 60
    else:
        raise RuntimeError(f"Unsupported Yahoo interval: {interval}")

    df = agent.normalize_yf(yf.download(
        ticker,
        period=period,
        interval=yf_interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    ))
    df = df[["o", "h", "l", "c", "v"]].copy()
    df["end"] = df.index + pd.Timedelta(minutes=minutes)

    if start_ms is not None:
        cursor = pd.to_datetime(int(start_ms), unit="ms", utc=True)
        df = df[df.index >= cursor]

    return df.tail(max(300, int(limit))).copy()


def resilient_binance_klines(symbol, interval, limit=300, start_ms=None):
    try:
        return retry(
            f"Binance {symbol} {interval}",
            lambda: _ORIG_BINANCE_KLINES(symbol, interval, limit, start_ms),
            attempts=2,
            sleep_seconds=0.5,
        )
    except Exception as exc:
        print(f"{symbol} {interval}: Binance unavailable -> OKX ({exc})")

    try:
        return retry(
            f"OKX {symbol} {interval}",
            lambda: okx_klines(symbol, interval, limit, start_ms),
            attempts=2,
            sleep_seconds=0.5,
        )
    except Exception as exc:
        print(f"{symbol} {interval}: OKX unavailable -> Yahoo ({exc})")

    return retry(
        f"Yahoo crypto {symbol} {interval}",
        lambda: yahoo_crypto_klines(symbol, interval, limit, start_ms),
        attempts=2,
        sleep_seconds=1.0,
    )


def okx_live_price(symbol):
    response = HTTP.get(
        "https://www.okx.com/api/v5/market/ticker",
        params={"instId": okx_inst_id(symbol)},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "0" or not payload.get("data"):
        raise RuntimeError(f"OKX ticker error: {payload}")
    return float(payload["data"][0]["last"])


def yahoo_crypto_live(symbol):
    ticker = CRYPTO_YAHOO[symbol]
    df = agent.normalize_yf(yf.download(
        ticker,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        threads=False,
    ))
    return float(df["c"].iloc[-1])


def resilient_live_price(key, cfg):
    if cfg.get("source") != "binance":
        return retry(
            f"{key} live",
            lambda: _ORIG_LIVE_PRICE(key, cfg),
            attempts=3,
            sleep_seconds=0.75,
        )

    symbol = cfg["symbol"]

    try:
        return retry(
            f"Binance live {symbol}",
            lambda: _ORIG_LIVE_PRICE(key, cfg),
            attempts=2,
            sleep_seconds=0.5,
        )
    except Exception as exc:
        print(f"{key}: Binance live unavailable -> OKX ({exc})")

    try:
        return retry(
            f"OKX live {symbol}",
            lambda: okx_live_price(symbol),
            attempts=2,
            sleep_seconds=0.5,
        )
    except Exception as exc:
        print(f"{key}: OKX live unavailable -> Yahoo ({exc})")

    return retry(
        f"Yahoo live {symbol}",
        lambda: yahoo_crypto_live(symbol),
        attempts=2,
        sleep_seconds=1.0,
    )


def resilient_twelve_request(endpoint, params):
    return retry(
        f"Twelve Data {endpoint}",
        lambda: _ORIG_TWELVE_REQUEST(endpoint, params),
        attempts=3,
        sleep_seconds=1.0,
    )


def resilient_yahoo_frames(ticker):
    return retry(
        f"Yahoo frames {ticker}",
        lambda: _ORIG_YAHOO_FRAMES(ticker),
        attempts=3,
        sleep_seconds=1.0,
    )


def resilient_send_telegram(text):
    return retry(
        "Telegram send",
        lambda: _ORIG_SEND_TELEGRAM(text),
        attempts=4,
        sleep_seconds=1.0,
    )


def recover_watch_history(st, key, now):
    watch = st["watch_setup"].get(key)
    if not watch or watch.get("touched"):
        return None

    cfg = agent.MARKETS[key]
    now = agent.stamp(now)
    created = agent.stamp(watch["created_at"])
    expiry = agent.stamp(watch["expires_at"])
    history_end = min(now, expiry)

    if history_end <= created:
        return None

    fake_trade = {
        "opened_at": created.isoformat(),
        "tracking_initialized_at": created.isoformat(),
        "last_tracking_bar": None,
    }

    try:
        bars = agent.get_tracking_bars(key, cfg, fake_trade, history_end)
    except Exception as exc:
        print(f"{key} watch recovery error: {exc}")
        return None

    if bars.empty:
        return None

    plan = watch["plan"]
    side = int(watch["side"])

    for ts, bar in bars.iterrows():
        ts = agent.stamp(ts)
        bar_end = agent.stamp(bar.get("end", ts + pd.Timedelta(minutes=1)))
        high = float(bar["h"])
        low = float(bar["l"])

        invalid = low <= plan["stop"] if side == 1 else high >= plan["stop"]
        if invalid:
            st["watch_setup"].pop(key, None)
            return (
                f"❌ {key} | SETUP INVALIDATED DURING DATA GAP\n"
                f"SETUP: {watch['setup_name']}\n"
                f"GRADE: {watch['grade']} | SCORE: {watch['score']}/10\n"
                f"STATUS: CLOSED / NO ENTRY\n"
                f"SETUP ID: {watch['id']}"
            )

        touched = low <= plan["zone_high"] and high >= plan["zone_low"]
        if touched:
            watch["touched"] = True
            watch["touch_time"] = bar_end.isoformat()
            st["watch_setup"][key] = watch
            print(f"{key}: recovered missed retest touch at {bar_end}")
            break

    return None


def resilient_process_watch(st, key, live_price, now):
    recovery_message = recover_watch_history(st, key, now)
    if recovery_message:
        return recovery_message, False
    return _ORIG_PROCESS_WATCH(st, key, live_price, now)


def frames_at_close(frames, close_time):
    close_time = agent.stamp(close_time)
    d15 = frames["15m"][(frames["15m"].index + pd.Timedelta(minutes=15)) <= close_time].copy()
    d1 = frames["1h"][(frames["1h"].index + pd.Timedelta(hours=1)) <= close_time].copy()
    d4 = frames["4h"][(frames["4h"].index + pd.Timedelta(hours=4)) <= close_time].copy()

    if len(d15) < 80 or len(d1) < 60 or len(d4) < 20:
        return None
    return {"15m": d15, "1h": d1, "4h": d4}


def resilient_scan_market(st, key, cfg):
    now = agent.now_utc()
    if not agent.market_allowed(cfg, now):
        return

    if key in st["active_trade"]:
        try:
            live_price = agent.get_live_price(key, cfg)
        except Exception as exc:
            print(key, "live tracking error:", exc)
            return

        for message in agent.monitor_open_trade(st, key, cfg, live_price, now):
            agent.send_telegram(message)
        return

    if key in st["watch_setup"]:
        try:
            live_price = agent.get_live_price(key, cfg)
        except Exception as exc:
            print(key, "watch live error:", exc)
            return

        watch_message, _ = agent.process_watch(st, key, live_price, now)
        if watch_message:
            agent.send_telegram(watch_message)
        return

    expected_close = now.floor("15min")
    slot_id = expected_close.isoformat()

    if st["last_analysis_slot"].get(key) == slot_id:
        return

    try:
        frames = agent.get_frames(key, cfg)
    except Exception as exc:
        print(key, "frame data error:", exc)
        return

    latest_start = agent.stamp(frames["15m"].index[-1])
    latest_close = latest_start + pd.Timedelta(minutes=15)

    if latest_close < expected_close:
        print(key, "waiting for expected 15m close:", expected_close, "latest:", latest_close)
        return

    last_raw = st["last_analysis_slot"].get(key)
    if last_raw:
        first_close = agent.stamp(last_raw) + pd.Timedelta(minutes=15)
    else:
        first_close = expected_close

    oldest_relevant = expected_close - pd.Timedelta(minutes=agent.WATCH_EXPIRY_MINUTES)
    if first_close < oldest_relevant:
        first_close = oldest_relevant.ceil("15min")

    candidate = None
    close_time = first_close

    while close_time <= expected_close:
        snapshot = frames_at_close(frames, close_time)
        if snapshot is not None:
            signal = agent.detect_signal(key, snapshot)
            if signal and st["last_signal"].get(key) != signal["id"]:
                candidate = signal
        close_time += pd.Timedelta(minutes=15)

    st["last_analysis_slot"][key] = slot_id

    if not candidate:
        return

    age_seconds = max(0.0, (now - candidate["confirmation_end"]).total_seconds())
    if age_seconds > agent.WATCH_EXPIRY_MINUTES * 60:
        st["last_signal"][key] = candidate["id"]
        print(key, "expired recovered setup ignored:", candidate["id"])
        return

    try:
        live_price = agent.get_live_price(key, cfg)
    except Exception as exc:
        print(key, "signal live error:", exc)
        return

    st["last_signal"][key] = candidate["id"]
    status = agent.initial_entry_status(candidate, live_price, now)
    agent.send_telegram(agent.format_signal(candidate, live_price, status))

    if status["tradable"]:
        st["watch_setup"].pop(key, None)
        agent.open_trade(st, candidate, live_price, now)
        return

    if status["watch"]:
        st["watch_setup"][key] = agent.signal_to_watch(candidate, live_price, now)
        recovery_message = recover_watch_history(st, key, now)
        if recovery_message:
            agent.send_telegram(recovery_message)


agent.binance_klines = resilient_binance_klines
agent.get_live_price = resilient_live_price
agent.twelve_request = resilient_twelve_request
agent.yahoo_frames = resilient_yahoo_frames
agent.send_telegram = resilient_send_telegram
agent.process_watch = resilient_process_watch
agent.scan_market = resilient_scan_market

if __name__ == "__main__":
    agent.main()

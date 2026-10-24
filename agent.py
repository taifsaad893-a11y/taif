import os
import json
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

MARKETS = {
    "XAUUSD": {
        "ticker": "GC=F",
        "weekdays_only": True,
    },
    "NASDAQ": {
        "ticker": "NQ=F",
        "weekdays_only": True,
    },
    "BTCUSD": {
        "ticker": "BTC-USD",
        "weekdays_only": False,
    },
}

TIMEFRAME = "15m"
PERIOD = "5d"

ATR_PERIOD = 14
MIN_RR = 1.5

# ORIGINAL STRATEGY SETTINGS — UNCHANGED
BREAK_BUFFER_ATR = 0.08
SL_BUFFER_ATR = 0.15

# Alert / entry-zone setting only
ENTRY_ZONE_ATR = 0.10

MAX_CANDLE_AGE_MINUTES = 45

STATE_FILE = "state.json"

LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")
BAGHDAD_TZ = ZoneInfo("Asia/Baghdad")


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        },
        timeout=20,
    )

    response.raise_for_status()


# =========================================================
# STATE
# =========================================================

def default_state():
    return {
        "signals": {},
        "trades": {},
        "last_closed_trade": {},
    }


def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)

        # New state format
        if (
            isinstance(raw, dict)
            and "signals" in raw
            and "trades" in raw
        ):
            raw.setdefault("last_closed_trade", {})
            return raw

        # Automatically migrate old state.json
        # so previous signal IDs are not lost.
        if isinstance(raw, dict):
            return {
                "signals": raw,
                "trades": {},
                "last_closed_trade": {},
            }

    except Exception:
        pass

    return default_state()


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(
            state,
            f,
            indent=2,
        )


def signal_id(
    symbol,
    setup,
    direction,
    candle_time,
):
    return (
        f"{symbol}|"
        f"{setup}|"
        f"{direction}|"
        f"{candle_time.isoformat()}"
    )


# =========================================================
# MARKET DATA
# =========================================================

def get_data(ticker):
    df = yf.download(
        ticker,
        period=PERIOD,
        interval=TIMEFRAME,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required:
        if column not in df.columns:
            return None

    df = df[required].copy()

    df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ],
        inplace=True,
    )

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


# =========================================================
# CLOSED CANDLES
# =========================================================

def remove_active_candle(df):
    if df is None or df.empty:
        return df

    now = pd.Timestamp.now(tz="UTC")
    candle_duration = pd.Timedelta(minutes=15)

    mask = (
        df.index + candle_duration
        <= now
    )

    return df[mask].copy()


# =========================================================
# FRESHNESS
# =========================================================

def candle_is_fresh(candle_time):
    now = datetime.now(timezone.utc)

    candle_close_time = (
        candle_time
        + pd.Timedelta(minutes=15)
    )

    age = (
        now
        - candle_close_time.to_pydatetime()
    )

    age_minutes = (
        age.total_seconds()
        / 60
    )

    return (
        0
        <= age_minutes
        <= MAX_CANDLE_AGE_MINUTES
    )


# =========================================================
# WEEKDAY FILTER
# =========================================================

def market_allowed_today(weekdays_only):
    if not weekdays_only:
        return True

    weekday = datetime.now(
        timezone.utc
    ).weekday()

    return weekday <= 4


# =========================================================
# ATR
# =========================================================

def calculate_atr(
    df,
    period=14,
):
    high_low = (
        df["High"]
        - df["Low"]
    )

    high_close = abs(
        df["High"]
        - df["Close"].shift()
    )

    low_close = abs(
        df["Low"]
        - df["Close"].shift()
    )

    ranges = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    )

    true_range = (
        ranges.max(axis=1)
    )

    return (
        true_range
        .rolling(period)
        .mean()
    )


# =========================================================
# VOLUME
# =========================================================

def volume_score(df):
    if len(df) < 21:
        return False

    current_volume = float(
        df["Volume"].iloc[-1]
    )

    average_volume = float(
        df["Volume"]
        .iloc[-21:-1]
        .mean()
    )

    if (
        pd.isna(current_volume)
        or
        pd.isna(average_volume)
        or
        average_volume <= 0
    ):
        return False

    return (
        current_volume
        >
        average_volume * 1.15
    )


# =========================================================
# LONDON RANGE
# =========================================================

def london_range(df):
    local = df.copy()

    local.index = local.index.tz_convert(
        LONDON_TZ
    )

    now = datetime.now(
        timezone.utc
    ).astimezone(
        LONDON_TZ
    )

    today = now.date()

    if now.hour < 12:
        return None

    session = local[
        (local.index.date == today)
        &
        (local.index.hour >= 8)
        &
        (local.index.hour < 12)
    ]

    if session.empty:
        return None

    return {
        "name": "London",
        "high": float(
            session["High"].max()
        ),
        "low": float(
            session["Low"].min()
        ),
        "timezone": LONDON_TZ,
        "end_hour": 12,
        "end_minute": 0,
    }


# =========================================================
# NEW YORK RANGE
# =========================================================

def new_york_range(df):
    local = df.copy()

    local.index = local.index.tz_convert(
        NY_TZ
    )

    now = datetime.now(
        timezone.utc
    ).astimezone(
        NY_TZ
    )

    today = now.date()

    current_minutes = (
        now.hour * 60
        + now.minute
    )

    start_minutes = (
        9 * 60
        + 30
    )

    end_minutes = (
        11 * 60
    )

    if current_minutes < end_minutes:
        return None

    candle_minutes = (
        local.index.hour * 60
        + local.index.minute
    )

    session = local[
        (local.index.date == today)
        &
        (
            candle_minutes
            >= start_minutes
        )
        &
        (
            candle_minutes
            < end_minutes
        )
    ]

    if session.empty:
        return None

    return {
        "name": "New York",
        "high": float(
            session["High"].max()
        ),
        "low": float(
            session["Low"].min()
        ),
        "timezone": NY_TZ,
        "end_hour": 11,
        "end_minute": 0,
    }


# =========================================================
# CANDLE AFTER SESSION
# =========================================================

def candle_after_session(
    candle_time,
    session,
):
    local_time = candle_time.tz_convert(
        session["timezone"]
    )

    candle_minutes = (
        local_time.hour * 60
        + local_time.minute
    )

    end_minutes = (
        session["end_hour"] * 60
        + session["end_minute"]
    )

    return candle_minutes >= end_minutes


# =========================================================
# SIGNAL DETECTION
# ORIGINAL STRATEGY CONDITIONS UNCHANGED
# =========================================================

def detect_signal(
    df,
    session,
):
    if session is None:
        return None

    if len(df) < ATR_PERIOD + 5:
        return None

    candle_time = df.index[-1]

    if not candle_after_session(
        candle_time,
        session,
    ):
        return None

    atr_series = calculate_atr(
        df,
        ATR_PERIOD,
    )

    atr = float(
        atr_series.iloc[-1]
    )

    if (
        pd.isna(atr)
        or
        atr <= 0
    ):
        return None

    current = df.iloc[-1]
    previous = df.iloc[-2]

    open_price = float(
        current["Open"]
    )

    high = float(
        current["High"]
    )

    low = float(
        current["Low"]
    )

    close = float(
        current["Close"]
    )

    prev_close = float(
        previous["Close"]
    )

    session_high = float(
        session["high"]
    )

    session_low = float(
        session["low"]
    )

    break_buffer = (
        atr
        * BREAK_BUFFER_ATR
    )

    sl_buffer = (
        atr
        * SL_BUFFER_ATR
    )

    strong_volume = volume_score(
        df
    )

    # =====================================================
    # LONG BREAKOUT
    # =====================================================

    if (
        close
        >
        session_high
        + break_buffer

        and

        prev_close
        <=
        session_high
        + break_buffer
    ):
        entry = close

        stop = min(
            session_high - sl_buffer,
            low - sl_buffer,
        )

        risk = entry - stop

        if risk > 0:
            return {
                "setup":
                f"{session['name']} Breakout",

                "direction":
                "LONG",

                "entry":
                entry,

                "stop":
                stop,

                "tp1":
                entry + risk * 1.5,

                "tp2":
                entry + risk * 2.0,

                "rr":
                2.0,

                "volume":
                strong_volume,

                "atr":
                atr,

                "reason":
                (
                    "15m candle closed above "
                    f"{session['name']} High."
                ),
            }

    # =====================================================
    # SHORT BREAKOUT
    # =====================================================

    if (
        close
        <
        session_low
        - break_buffer

        and

        prev_close
        >=
        session_low
        - break_buffer
    ):
        entry = close

        stop = max(
            session_low + sl_buffer,
            high + sl_buffer,
        )

        risk = stop - entry

        if risk > 0:
            return {
                "setup":
                f"{session['name']} Breakout",

                "direction":
                "SHORT",

                "entry":
                entry,

                "stop":
                stop,

                "tp1":
                entry - risk * 1.5,

                "tp2":
                entry - risk * 2.0,

                "rr":
                2.0,

                "volume":
                strong_volume,

                "atr":
                atr,

                "reason":
                (
                    "15m candle closed below "
                    f"{session['name']} Low."
                ),
            }

    # =====================================================
    # HIGH SWEEP -> SHORT
    # =====================================================

    swept_high = (
        high
        >
        session_high
        + break_buffer

        and

        close
        <
        session_high
    )

    bearish_confirmation = (
        close < open_price
        or
        close < prev_close
    )

    if (
        swept_high
        and
        bearish_confirmation
    ):
        entry = close

        stop = (
            high
            + sl_buffer
        )

        risk = stop - entry

        if risk > 0:
            return {
                "setup":
                f"{session['name']} High Sweep",

                "direction":
                "SHORT",

                "entry":
                entry,

                "stop":
                stop,

                "tp1":
                entry - risk * 1.5,

                "tp2":
                entry - risk * 2.0,

                "rr":
                2.0,

                "volume":
                strong_volume,

                "atr":
                atr,

                "reason":
                (
                    f"{session['name']} High "
                    "was swept and price "
                    "closed back inside the range."
                ),
            }

    # =====================================================
    # LOW SWEEP -> LONG
    # =====================================================

    swept_low = (
        low
        <
        session_low
        - break_buffer

        and

        close
        >
        session_low
    )

    bullish_confirmation = (
        close > open_price
        or
        close > prev_close
    )

    if (
        swept_low
        and
        bullish_confirmation
    ):
        entry = close

        stop = (
            low
            - sl_buffer
        )

        risk = entry - stop

        if risk > 0:
            return {
                "setup":
                f"{session['name']} Low Sweep",

                "direction":
                "LONG",

                "entry":
                entry,

                "stop":
                stop,

                "tp1":
                entry + risk * 1.5,

                "tp2":
                entry + risk * 2.0,

                "rr":
                2.0,

                "volume":
                strong_volume,

                "atr":
                atr,

                "reason":
                (
                    f"{session['name']} Low "
                    "was swept and price "
                    "closed back inside the range."
                ),
            }

    return None


# =========================================================
# SCORE
# =========================================================

def get_score(signal):
    score = 70

    if signal["volume"]:
        score += 10

    if "Sweep" in signal["setup"]:
        score += 5

    if score >= 80:
        return "A"

    if score >= 70:
        return "B"

    return "C"


# =========================================================
# ENTRY ZONE
# =========================================================

def calculate_entry_zone(signal):
    entry = signal["entry"]
    atr = signal["atr"]

    zone_size = (
        atr
        * ENTRY_ZONE_ATR
    )

    zone_low = (
        entry
        - zone_size
    )

    zone_high = (
        entry
        + zone_size
    )

    return (
        zone_low,
        zone_high,
    )


# =========================================================
# CURRENT PRICE
# =========================================================

def get_current_price(ticker):
    try:
        live = yf.download(
            ticker,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=False,
        )

        if live.empty:
            return None

        if isinstance(
            live.columns,
            pd.MultiIndex,
        ):
            live.columns = (
                live.columns
                .get_level_values(0)
            )

        if "Close" not in live.columns:
            return None

        live = live.dropna(
            subset=["Close"]
        )

        if live.empty:
            return None

        return float(
            live["Close"].iloc[-1]
        )

    except Exception as e:
        print(
            f"Current price error: {e}"
        )

        return None


# =========================================================
# ENTRY STATUS
# =========================================================

def price_inside_entry_zone(
    current_price,
    zone_low,
    zone_high,
):
    if current_price is None:
        return False

    return (
        zone_low
        <= current_price
        <= zone_high
    )


def get_entry_status(
    current_price,
    zone_low,
    zone_high,
):
    if current_price is None:
        return (
            "⚠️ Current price unavailable\n"
            "Check market price before entry."
        )

    if price_inside_entry_zone(
        current_price,
        zone_low,
        zone_high,
    ):
        return (
            "✅ PRICE INSIDE ENTRY ZONE\n"
            "⚡ ENTER NOW"
        )

    return (
        "⛔ ENTRY MISSED / PRICE OUTSIDE ZONE\n"
        "➡️ DO NOT CHASE — WAIT"
    )


# =========================================================
# TRADE TRACKING
# =========================================================

def open_trade(
    state,
    symbol,
    signal,
    current_price,
    candle_time,
):
    zone_low, zone_high = (
        calculate_entry_zone(signal)
    )

    state["trades"][symbol] = {
        "status": "OPEN",
        "symbol": symbol,
        "direction": signal["direction"],
        "setup": signal["setup"],
        "entry_reference": signal["entry"],
        "actual_entry_reference": current_price,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "stop": signal["stop"],
        "tp1": signal["tp1"],
        "tp2": signal["tp2"],
        "tp1_hit": False,
        "opened_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "signal_candle": candle_time.isoformat(),
    }


def close_trade(
    state,
    symbol,
    result,
    price,
):
    trade = state["trades"].get(symbol)

    if not trade:
        return

    trade["status"] = "CLOSED"
    trade["result"] = result
    trade["closed_price"] = price
    trade["closed_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    state["last_closed_trade"][symbol] = (
        trade.copy()
    )

    del state["trades"][symbol]


def format_trade_event(
    symbol,
    trade,
    event,
    current_price,
):
    direction_icon = (
        "🟢"
        if trade["direction"] == "LONG"
        else "🔴"
    )

    if event == "SL":
        return (
            f"❌ {symbol} — STOP LOSS HIT\n\n"
            f"{direction_icon} Previous trade: "
            f"{trade['direction']}\n"
            f"Setup: {trade['setup']}\n\n"
            f"🛑 Stop Loss: "
            f"{trade['stop']:.2f}\n"
            f"Current price: "
            f"{current_price:.2f}\n\n"
            f"Trade closed."
        )

    if event == "TP1":
        return (
            f"✅ {symbol} — TP1 HIT\n\n"
            f"{direction_icon} "
            f"{trade['direction']}\n"
            f"Setup: {trade['setup']}\n\n"
            f"🎯 TP1: "
            f"{trade['tp1']:.2f}\n"
            f"Current price: "
            f"{current_price:.2f}\n\n"
            f"Trade remains active for TP2."
        )

    if event == "TP2":
        return (
            f"🎯 {symbol} — TP2 HIT\n\n"
            f"{direction_icon} "
            f"{trade['direction']}\n"
            f"Setup: {trade['setup']}\n\n"
            f"🎯 TP2: "
            f"{trade['tp2']:.2f}\n"
            f"Current price: "
            f"{current_price:.2f}\n\n"
            f"✅ Trade completed."
        )

    return ""


def monitor_open_trade(
    state,
    symbol,
    current_price,
):
    trade = state["trades"].get(symbol)

    if not trade:
        return

    if current_price is None:
        return

    direction = trade["direction"]

    # =====================================================
    # LONG TRADE
    # =====================================================

    if direction == "LONG":

        # Check stop first.
        # With snapshot polling we cannot know exact intrabar
        # order if both levels were crossed between scans.
        if current_price <= trade["stop"]:
            send_telegram(
                format_trade_event(
                    symbol,
                    trade,
                    "SL",
                    current_price,
                )
            )

            close_trade(
                state,
                symbol,
                "SL",
                current_price,
            )

            return

        if current_price >= trade["tp2"]:
            if not trade.get(
                "tp1_hit",
                False,
            ):
                send_telegram(
                    format_trade_event(
                        symbol,
                        trade,
                        "TP1",
                        current_price,
                    )
                )

            send_telegram(
                format_trade_event(
                    symbol,
                    trade,
                    "TP2",
                    current_price,
                )
            )

            close_trade(
                state,
                symbol,
                "TP2",
                current_price,
            )

            return

        if (
            current_price >= trade["tp1"]
            and
            not trade.get(
                "tp1_hit",
                False,
            )
        ):
            send_telegram(
                format_trade_event(
                    symbol,
                    trade,
                    "TP1",
                    current_price,
                )
            )

            trade["tp1_hit"] = True

            return

    # =====================================================
    # SHORT TRADE
    # =====================================================

    if direction == "SHORT":

        if current_price >= trade["stop"]:
            send_telegram(
                format_trade_event(
                    symbol,
                    trade,
                    "SL",
                    current_price,
                )
            )

            close_trade(
                state,
                symbol,
                "SL",
                current_price,
            )

            return

        if current_price <= trade["tp2"]:
            if not trade.get(
                "tp1_hit",
                False,
            ):
                send_telegram(
                    format_trade_event(
                        symbol,
                        trade,
                        "TP1",
                        current_price,
                    )
                )

            send_telegram(
                format_trade_event(
                    symbol,
                    trade,
                    "TP2",
                    current_price,
                )
            )

            close_trade(
                state,
                symbol,
                "TP2",
                current_price,
            )

            return

        if (
            current_price <= trade["tp1"]
            and
            not trade.get(
                "tp1_hit",
                False,
            )
        ):
            send_telegram(
                format_trade_event(
                    symbol,
                    trade,
                    "TP1",
                    current_price,
                )
            )

            trade["tp1_hit"] = True


# =========================================================
# REVERSAL CONTEXT
# =========================================================

def get_reversal_context(
    state,
    symbol,
    new_direction,
):
    last_trade = (
        state
        .get(
            "last_closed_trade",
            {}
        )
        .get(symbol)
    )

    if not last_trade:
        return ""

    old_direction = last_trade.get(
        "direction"
    )

    result = last_trade.get(
        "result"
    )

    if (
        old_direction
        and
        old_direction != new_direction
    ):
        if result == "SL":
            return (
                "🔄 REVERSAL AFTER STOP\n"
                f"Previous {old_direction} "
                "trade hit Stop Loss.\n"
                "New opposite setup detected.\n\n"
            )

        return (
            "🔄 OPPOSITE / REVERSAL SETUP\n"
            f"Previous direction: "
            f"{old_direction}\n"
            f"New direction: "
            f"{new_direction}\n\n"
        )

    return ""


# =========================================================
# FORMAT SIGNAL
# =========================================================

def format_signal(
    symbol,
    signal,
    candle_time,
    current_price,
    reversal_context="",
):
    if signal["direction"] == "LONG":
        icon = "🟢"
    else:
        icon = "🔴"

    score = get_score(
        signal
    )

    volume_text = (
        "Strong"
        if signal["volume"]
        else "Normal"
    )

    zone_low, zone_high = (
        calculate_entry_zone(
            signal
        )
    )

    status = get_entry_status(
        current_price,
        zone_low,
        zone_high,
    )

    baghdad_time = (
        candle_time
        .tz_convert(BAGHDAD_TZ)
        .strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    if current_price is None:
        current_text = "Unavailable"
    else:
        current_text = (
            f"{current_price:.2f}"
        )

    return (
        f"{reversal_context}"
        f"{icon} {symbol} — "
        f"{signal['direction']}\n\n"

        f"{status}\n\n"

        f"Setup: "
        f"{signal['setup']}\n"

        f"Timeframe: 15m\n"

        f"Confirmation candle: "
        f"{baghdad_time} Baghdad\n\n"

        f"📍 ENTRY ZONE\n"
        f"{zone_low:.2f} — "
        f"{zone_high:.2f}\n\n"

        f"Current price: "
        f"{current_text}\n"

        f"Confirmation close: "
        f"{signal['entry']:.2f}\n\n"

        f"🛑 Stop Loss: "
        f"{signal['stop']:.2f}\n\n"

        f"🎯 TP1: "
        f"{signal['tp1']:.2f}\n"

        f"🎯 TP2: "
        f"{signal['tp2']:.2f}\n\n"

        f"R:R TP1: 1:1.5\n"
        f"R:R TP2: 1:2.0\n\n"

        f"Volume: "
        f"{volume_text}\n"

        f"Score: "
        f"{score}\n\n"

        f"Reason:\n"
        f"{signal['reason']}"
    )


# =========================================================
# CLEAN SIGNAL HISTORY
# =========================================================

def clean_signal_history(state):
    signals = state.get(
        "signals",
        {}
    )

    if len(signals) <= 500:
        return

    items = list(
        signals.items()
    )

    state["signals"] = dict(
        items[-500:]
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print(
        "Session Market Scanner + Trade Tracker"
    )

    state = load_state()

    for symbol, config in MARKETS.items():

        print(
            f"\nChecking {symbol}..."
        )

        if not market_allowed_today(
            config["weekdays_only"]
        ):
            print(
                f"{symbol}: "
                "disabled on weekend."
            )

            continue

        try:
            # =================================================
            # GET CURRENT PRICE FIRST
            # Used for trade tracking and entry status
            # =================================================

            current_price = (
                get_current_price(
                    config["ticker"]
                )
            )

            # =================================================
            # MONITOR EXISTING OPEN TRADE
            # =================================================

            monitor_open_trade(
                state,
                symbol,
                current_price,
            )

            # =================================================
            # GET 15M DATA
            # =================================================

            df = get_data(
                config["ticker"]
            )

            if (
                df is None
                or
                len(df) < 30
            ):
                print(
                    f"{symbol}: "
                    "insufficient data."
                )

                continue

            df = remove_active_candle(
                df
            )

            if (
                df is None
                or
                df.empty
            ):
                print(
                    f"{symbol}: "
                    "no closed candles."
                )

                continue

            candle_time = (
                df.index[-1]
            )

            if not candle_is_fresh(
                candle_time
            ):
                print(
                    f"{symbol}: "
                    "stale candle."
                )

                continue

            sessions = [
                london_range(df),
                new_york_range(df),
            ]

            for session in sessions:

                if session is None:
                    continue

                signal = detect_signal(
                    df,
                    session,
                )

                if signal is None:
                    continue

                if (
                    signal["rr"]
                    <
                    MIN_RR
                ):
                    continue

                sid = signal_id(
                    symbol,
                    signal["setup"],
                    signal["direction"],
                    candle_time,
                )

                # =================================================
                # DUPLICATE PROTECTION
                # =================================================

                if sid in state["signals"]:
                    print(
                        f"{symbol}: "
                        "duplicate ignored."
                    )

                    continue

                # =================================================
                # REFRESH PRICE RIGHT BEFORE ALERT
                # =================================================

                current_price = (
                    get_current_price(
                        config["ticker"]
                    )
                )

                zone_low, zone_high = (
                    calculate_entry_zone(
                        signal
                    )
                )

                inside_zone = (
                    price_inside_entry_zone(
                        current_price,
                        zone_low,
                        zone_high,
                    )
                )

                # =================================================
                # REVERSAL CONTEXT
                # =================================================

                reversal_context = (
                    get_reversal_context(
                        state,
                        symbol,
                        signal["direction"],
                    )
                )

                # =================================================
                # SEND SIGNAL
                # =================================================

                message = format_signal(
                    symbol,
                    signal,
                    candle_time,
                    current_price,
                    reversal_context,
                )

                send_telegram(
                    message
                )

                # =================================================
                # SAVE SIGNAL HISTORY
                # =================================================

                state["signals"][sid] = {
                    "sent_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "symbol":
                    symbol,

                    "setup":
                    signal["setup"],

                    "direction":
                    signal["direction"],

                    "entry":
                    signal["entry"],

                    "current_price":
                    current_price,

                    "entry_valid":
                    inside_zone,
                }

                # =================================================
                # ONLY TRACK AS OPEN IF BOT SAID ENTER NOW
                # =================================================

                if inside_zone:

                    existing_trade = (
                        state["trades"]
                        .get(symbol)
                    )

                    # Do not silently overwrite
                    # an already open trade.
                    if existing_trade:

                        if (
                            existing_trade["direction"]
                            != signal["direction"]
                        ):
                            send_telegram(
                                f"⚠️ {symbol} — "
                                "OPPOSITE SIGNAL\n\n"
                                f"Existing tracked trade: "
                                f"{existing_trade['direction']}\n"
                                f"New signal: "
                                f"{signal['direction']}\n\n"
                                "The existing trade is still "
                                "marked OPEN.\n"
                                "No automatic replacement was made."
                            )

                    else:
                        open_trade(
                            state,
                            symbol,
                            signal,
                            current_price,
                            candle_time,
                        )

                        print(
                            f"{symbol}: "
                            "new trade tracked."
                        )

                else:
                    print(
                        f"{symbol}: "
                        "signal sent but entry "
                        "was outside zone; "
                        "not tracked as open."
                    )

        except Exception as e:
            print(
                f"{symbol}: ERROR -> {e}"
            )

    clean_signal_history(
        state
    )

    save_state(
        state
    )

    print(
        "\nScan completed."
    )


if __name__ == "__main__":
    main()
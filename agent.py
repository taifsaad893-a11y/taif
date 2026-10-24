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

BREAK_BUFFER_ATR = 0.08
SL_BUFFER_ATR = 0.15

# Latest closed candle should not be too old.
# 45 minutes gives some tolerance for delayed Yahoo data.
MAX_CANDLE_AGE_MINUTES = 45

STATE_FILE = "state.json"

LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")


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

def load_state():

    try:

        with open(STATE_FILE, "r") as f:
            return json.load(f)

    except Exception:
        return {}


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
# DATA
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

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = (
            df.columns.get_level_values(0)
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"Missing column: {column}"
            )

            return None

    df = df[
        required_columns
    ].copy()

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

        df.index = (
            df.index.tz_localize("UTC")
        )

    else:

        df.index = (
            df.index.tz_convert("UTC")
        )

    return df


# =========================================================
# CLOSED CANDLE
# =========================================================

def remove_active_candle(df):

    """
    yfinance may include the currently-forming 15m candle.

    We only want candles that should already be closed.
    """

    if df is None or df.empty:
        return df

    now = pd.Timestamp.now(tz="UTC")

    candle_duration = pd.Timedelta(
        minutes=15
    )

    closed_mask = (
        df.index + candle_duration
        <= now
    )

    closed_df = df[
        closed_mask
    ].copy()

    return closed_df


# =========================================================
# FRESHNESS
# =========================================================

def candle_is_fresh(candle_time):

    now = datetime.now(
        timezone.utc
    )

    if candle_time.tzinfo is None:

        candle_time = (
            candle_time.tz_localize("UTC")
        )

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
# MARKET DAY FILTER
# =========================================================

def market_allowed_today(
    weekdays_only
):

    if not weekdays_only:
        return True

    now_utc = datetime.now(
        timezone.utc
    )

    weekday = now_utc.weekday()

    # Monday = 0
    # Friday = 4

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

    atr = (
        true_range
        .rolling(period)
        .mean()
    )

    return atr


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

    """
    London opening range:

    08:00 -> 12:00
    Europe/London local time.

    Signals are only allowed AFTER 12:00 London.
    """

    local = df.copy()

    local.index = (
        local.index.tz_convert(
            LONDON_TZ
        )
    )

    now = datetime.now(
        timezone.utc
    ).astimezone(
        LONDON_TZ
    )

    today = now.date()

    # Do not trade this range
    # until it is complete.

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

    high = float(
        session["High"].max()
    )

    low = float(
        session["Low"].min()
    )

    return {
        "name": "London",

        "high": high,

        "low": low,

        "timezone": LONDON_TZ,

        "end_hour": 12,

        "end_minute": 0,
    }


# =========================================================
# NEW YORK RANGE
# =========================================================

def new_york_range(df):

    """
    New York opening range:

    09:30 -> 11:00
    America/New_York local time.

    Signals are only allowed AFTER 11:00 New York.
    """

    local = df.copy()

    local.index = (
        local.index.tz_convert(
            NY_TZ
        )
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

    high = float(
        session["High"].max()
    )

    low = float(
        session["Low"].min()
    )

    return {
        "name": "New York",

        "high": high,

        "low": low,

        "timezone": NY_TZ,

        "end_hour": 11,

        "end_minute": 0,
    }


# =========================================================
# MAKE SURE CANDLE IS AFTER SESSION
# =========================================================

def candle_after_session(
    candle_time,
    session,
):

    local_time = (
        candle_time.tz_convert(
            session["timezone"]
        )
    )

    end_minutes = (
        session["end_hour"] * 60
        +
        session["end_minute"]
    )

    candle_minutes = (
        local_time.hour * 60
        +
        local_time.minute
    )

    return (
        candle_minutes
        >= end_minutes
    )


# =========================================================
# SIGNAL DETECTION
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

    strong_volume = (
        volume_score(df)
    )


    # =====================================================
    # LONG BREAKOUT
    # =====================================================

    bullish_breakout = (

        close
        >
        session_high
        + break_buffer

        and

        prev_close
        <=
        session_high
        + break_buffer
    )

    if bullish_breakout:

        entry = close

        stop = min(

            session_high
            - sl_buffer,

            low
            - sl_buffer,
        )

        risk = (
            entry
            - stop
        )

        if risk > 0:

            tp1 = (
                entry
                + risk * 1.5
            )

            tp2 = (
                entry
                + risk * 2.0
            )

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
                tp1,

                "tp2":
                tp2,

                "rr":
                2.0,

                "volume":
                strong_volume,

                "reason":
                (
                    f"15m candle closed above "
                    f"{session['name']} High."
                ),
            }


    # =====================================================
    # SHORT BREAKOUT
    # =====================================================

    bearish_breakout = (

        close
        <
        session_low
        - break_buffer

        and

        prev_close
        >=
        session_low
        - break_buffer
    )

    if bearish_breakout:

        entry = close

        stop = max(

            session_low
            + sl_buffer,

            high
            + sl_buffer,
        )

        risk = (
            stop
            - entry
        )

        if risk > 0:

            tp1 = (
                entry
                - risk * 1.5
            )

            tp2 = (
                entry
                - risk * 2.0
            )

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
                tp1,

                "tp2":
                tp2,

                "rr":
                2.0,

                "volume":
                strong_volume,

                "reason":
                (
                    f"15m candle closed below "
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

        close
        <
        open_price

        or

        close
        <
        prev_close
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

        risk = (
            stop
            - entry
        )

        if risk > 0:

            tp1 = (
                entry
                - risk * 1.5
            )

            tp2 = (
                entry
                - risk * 2.0
            )

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
                tp1,

                "tp2":
                tp2,

                "rr":
                2.0,

                "volume":
                strong_volume,

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

        close
        >
        open_price

        or

        close
        >
        prev_close
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

        risk = (
            entry
            - stop
        )

        if risk > 0:

            tp1 = (
                entry
                + risk * 1.5
            )

            tp2 = (
                entry
                + risk * 2.0
            )

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
                tp1,

                "tp2":
                tp2,

                "rr":
                2.0,

                "volume":
                strong_volume,

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

    # Volume improves score.
    # It does NOT block a trade.

    if signal["volume"]:
        score += 10

    # Sweeps get small bonus.

    if "Sweep" in signal["setup"]:
        score += 5

    if score >= 80:
        return "A"

    if score >= 70:
        return "B"

    return "C"


# =========================================================
# TELEGRAM SIGNAL
# =========================================================

def format_signal(
    symbol,
    signal,
    candle_time,
):

    if signal["direction"] == "LONG":

        icon = "🟢"

    else:

        icon = "🔴"


    score = get_score(
        signal
    )


    if signal["volume"]:

        volume_text = "Strong"

    else:

        volume_text = "Normal"


    baghdad_time = (
        candle_time
        .tz_convert(
            ZoneInfo("Asia/Baghdad")
        )
        .strftime(
            "%Y-%m-%d %H:%M"
        )
    )


    return (

        f"{icon} {symbol} — "
        f"{signal['direction']}\n\n"

        f"Setup: "
        f"{signal['setup']}\n"

        f"Timeframe: 15m\n"

        f"Candle: "
        f"{baghdad_time} Baghdad\n\n"

        f"Entry: "
        f"{signal['entry']:.2f}\n"

        f"Stop Loss: "
        f"{signal['stop']:.2f}\n\n"

        f"TP1: "
        f"{signal['tp1']:.2f}\n"

        f"TP2: "
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
# CLEAN OLD STATE
# =========================================================

def clean_state(state):

    """
    Prevent state.json from growing forever.
    Keep roughly the newest 500 signals.
    """

    if len(state) <= 500:
        return state

    items = list(
        state.items()
    )

    items = items[-500:]

    return dict(items)


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "================================"
    )

    print(
        "Session Market Scanner"
    )

    print(
        "================================"
    )


    state = load_state()


    for symbol, config in MARKETS.items():

        print()

        print(
            f"Checking {symbol}..."
        )


        # =============================================
        # WEEKDAY FILTER
        # =============================================

        if not market_allowed_today(
            config["weekdays_only"]
        ):

            print(
                f"{symbol}: "
                "market disabled on weekend."
            )

            continue


        try:

            # =========================================
            # GET DATA
            # =========================================

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
                    "insufficient market data."
                )

                continue


            # =========================================
            # CLOSED CANDLES ONLY
            # =========================================

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


            print(
                f"{symbol}: "
                f"latest closed candle = "
                f"{candle_time}"
            )


            # =========================================
            # FRESHNESS FILTER
            # =========================================

            if not candle_is_fresh(
                candle_time
            ):

                print(
                    f"{symbol}: "
                    "latest candle is stale. "
                    "Skipping."
                )

                continue


            # =========================================
            # BUILD SESSION RANGES
            # =========================================

            london = (
                london_range(df)
            )

            new_york = (
                new_york_range(df)
            )


            sessions = [
                london,
                new_york,
            ]


            available_sessions = [
                session
                for session in sessions
                if session is not None
            ]


            if not available_sessions:

                print(
                    f"{symbol}: "
                    "no completed session "
                    "range available yet."
                )

                continue


            # =========================================
            # ANALYZE
            # =========================================

            for session in available_sessions:

                print(
                    f"{symbol}: analyzing "
                    f"{session['name']} "
                    f"High={session['high']:.2f} "
                    f"Low={session['low']:.2f}"
                )


                signal = detect_signal(
                    df,
                    session,
                )


                if signal is None:

                    print(
                        f"{symbol}: "
                        f"no {session['name']} "
                        "setup."
                    )

                    continue


                # =====================================
                # RR FILTER
                # =====================================

                if (
                    signal["rr"]
                    <
                    MIN_RR
                ):

                    print(
                        f"{symbol}: "
                        "RR below minimum."
                    )

                    continue


                # =====================================
                # DUPLICATE FILTER
                # =====================================

                sid = signal_id(

                    symbol,

                    signal["setup"],

                    signal["direction"],

                    candle_time,
                )


                if sid in state:

                    print(
                        f"{symbol}: "
                        "duplicate signal ignored."
                    )

                    continue


                # =====================================
                # SEND TELEGRAM
                # =====================================

                message = format_signal(

                    symbol,

                    signal,

                    candle_time,
                )


                send_telegram(
                    message
                )


                state[sid] = {

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

                    "stop":
                    signal["stop"],

                    "tp1":
                    signal["tp1"],

                    "tp2":
                    signal["tp2"],
                }


                print(
                    f"{symbol}: "
                    f"{signal['direction']} "
                    "signal sent to Telegram."
                )


        except Exception as e:

            print(
                f"{symbol}: ERROR -> {e}"
            )


    # =============================================
    # SAVE STATE
    # =============================================

    state = clean_state(
        state
    )

    save_state(
        state
    )


    print()

    print(
        "Scan completed."
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
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
    "XAUUSD": "GC=F",
    "NASDAQ": "NQ=F",
    "BTCUSD": "BTC-USD",
}

TIMEFRAME = "15m"
PERIOD = "5d"

MIN_RR = 1.5

ATR_PERIOD = 14

# Minimum amount price must break a level
BREAK_BUFFER_ATR = 0.08

# Extra space behind stop
SL_BUFFER_ATR = 0.15

STATE_FILE = "state.json"

NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")


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
        json.dump(state, f, indent=2)


def signal_id(symbol, setup, direction, candle_time):

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

    df = df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].copy()

    df.dropna(inplace=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

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

    true_range = ranges.max(axis=1)

    atr = true_range.rolling(
        period
    ).mean()

    return atr


# =========================================================
# LONDON SESSION
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
        "high": float(
            session["High"].max()
        ),

        "low": float(
            session["Low"].min()
        ),

        "name": "London",
    }


# =========================================================
# NEW YORK SESSION
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

    minutes = (
        local.index.hour * 60
        +
        local.index.minute
    )

    start = (
        9 * 60
        + 30
    )

    end = (
        11 * 60
    )

    session = local[
        (local.index.date == today)
        &
        (minutes >= start)
        &
        (minutes < end)
    ]

    if session.empty:
        return None

    return {
        "high": float(
            session["High"].max()
        ),

        "low": float(
            session["Low"].min()
        ),

        "name": "New York",
    }


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

    if average_volume <= 0:
        return False

    return (
        current_volume
        >
        average_volume * 1.15
    )


# =========================================================
# SIGNAL DETECTION
# =========================================================

def detect_signal(df, session):

    if session is None:
        return None

    if len(df) < ATR_PERIOD + 5:
        return None

    atr_series = calculate_atr(
        df,
        ATR_PERIOD,
    )

    atr = float(
        atr_series.iloc[-1]
    )

    if pd.isna(atr) or atr <= 0:
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

    session_high = session["high"]
    session_low = session["low"]

    break_buffer = (
        atr
        * BREAK_BUFFER_ATR
    )

    sl_buffer = (
        atr
        * SL_BUFFER_ATR
    )

    strong_volume = volume_score(df)


    # =====================================================
    # BULLISH BREAKOUT
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
                    "15m candle closed above "
                    f"{session['name']} High."
                ),
            }


    # =====================================================
    # BEARISH BREAKOUT
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
# FORMAT REAL SIGNAL
# =========================================================

def format_signal(
    symbol,
    signal,
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

    return (

        f"{icon} {symbol} — "
        f"{signal['direction']}\n\n"

        f"Setup: "
        f"{signal['setup']}\n"

        f"Timeframe: 15m\n\n"

        f"Entry: "
        f"{signal['entry']:.2f}\n"

        f"Stop Loss: "
        f"{signal['stop']:.2f}\n\n"

        f"TP1: "
        f"{signal['tp1']:.2f}\n"

        f"TP2: "
        f"{signal['tp2']:.2f}\n\n"

        f"R:R: "
        f"1:{signal['rr']:.1f}\n"

        f"Volume: "
        f"{volume_text}\n"

        f"Score: "
        f"{score}\n\n"

        f"Reason:\n"
        f"{signal['reason']}"
    )


# =========================================================
# TEST REPORT
# =========================================================

def send_test_report(
    symbol,
    df,
    candle_time,
):

    last_price = float(
        df["Close"].iloc[-1]
    )

    atr_series = calculate_atr(
        df,
        ATR_PERIOD,
    )

    current_atr = float(
        atr_series.iloc[-1]
    )

    london = london_range(df)
    new_york = new_york_range(df)

    report = (

        f"🔎 TEST SCAN — {symbol}\n\n"

        f"Last closed 15m price: "
        f"{last_price:.2f}\n"

        f"Candle UTC: "
        f"{candle_time}\n"

        f"ATR(14): "
        f"{current_atr:.2f}\n\n"
    )


    # LONDON

    if london:

        report += (

            "🇬🇧 LONDON RANGE\n"

            f"High: "
            f"{london['high']:.2f}\n"

            f"Low: "
            f"{london['low']:.2f}\n\n"
        )

    else:

        report += (
            "🇬🇧 LONDON RANGE: "
            "Not available yet\n\n"
        )


    # NEW YORK

    if new_york:

        report += (

            "🇺🇸 NEW YORK RANGE\n"

            f"High: "
            f"{new_york['high']:.2f}\n"

            f"Low: "
            f"{new_york['low']:.2f}\n\n"
        )

    else:

        report += (
            "🇺🇸 NEW YORK RANGE: "
            "Not available yet\n\n"
        )


    # CHECK SIGNALS

    signals_found = []

    if london:

        london_signal = detect_signal(
            df,
            london,
        )

        if london_signal:

            signals_found.append(

                "London: "
                f"{london_signal['setup']} "
                f"{london_signal['direction']}"
            )


    if new_york:

        ny_signal = detect_signal(
            df,
            new_york,
        )

        if ny_signal:

            signals_found.append(

                "New York: "
                f"{ny_signal['setup']} "
                f"{ny_signal['direction']}"
            )


    if signals_found:

        report += (
            "🚨 SETUP DETECTED\n"
        )

        report += "\n".join(
            signals_found
        )

    else:

        report += (
            "⚪ NO SETUP on latest "
            "closed 15m candle"
        )


    send_telegram(
        report
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "Starting Session Scanner..."
    )

    state = load_state()


    for symbol, ticker in MARKETS.items():

        print(
            f"Checking {symbol}..."
        )

        try:

            df = get_data(
                ticker
            )

            if (
                df is None
                or
                len(df) < 30
            ):

                print(
                    f"No sufficient data "
                    f"for {symbol}"
                )

                continue


            # Remove potentially active candle.
            # We only analyze closed candles.

            df = df.iloc[:-1].copy()

            if df.empty:
                continue


            candle_time = (
                df.index[-1]
            )


            # =============================================
            # TEST MESSAGE
            # =============================================

            send_test_report(
                symbol,
                df,
                candle_time,
            )


            # =============================================
            # REAL SIGNAL SCAN
            # =============================================

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


                if signal["rr"] < MIN_RR:
                    continue


                sid = signal_id(

                    symbol,

                    signal["setup"],

                    signal["direction"],

                    candle_time,
                )


                if state.get(sid):

                    print(
                        "Duplicate ignored: "
                        f"{sid}"
                    )

                    continue


                message = format_signal(
                    symbol,
                    signal,
                )


                send_telegram(
                    message
                )


                state[sid] = {

                    "sent_at":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                    "entry":
                    signal["entry"],

                    "setup":
                    signal["setup"],
                }


                print(
                    "Signal sent: "
                    f"{symbol} "
                    f"{signal['direction']}"
                )


        except Exception as e:

            print(
                f"ERROR {symbol}: {e}"
            )


    save_state(
        state
    )

    print(
        "Scan completed."
    )


if __name__ == "__main__":
    main()
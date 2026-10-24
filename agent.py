import os
import json
import math
import time
import requests
import yfinance as yf
import pandas as pd

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

ATR_PERIOD = 14
MIN_RR = 1.5

# =========================================================
# SCORE
# =========================================================

A_SCORE = 7.5
B_SCORE = 6.0


# =========================================================
# 15M SETUP SETTINGS
# =========================================================

BREAK_BUFFER_ATR = 0.05
SWEEP_BUFFER_ATR = 0.02
SL_BUFFER_ATR = 0.15

ENTRY_BACK_ATR = 0.12
ENTRY_FRONT_ATR = 0.04


# =========================================================
# TIMING
# =========================================================

# ENTER NOW only during first 2 minutes after 15m candle closes.
MAX_ENTER_AGE_SECONDS = 120

# If direct entry is missed, keep watching setup for a retest.
WATCH_EXPIRY_MINUTES = 60

# Small confirmation after retest.
RETEST_CONFIRM_ATR = 0.05

# Don't chase after retest confirmation.
MAX_RETEST_CHASE_R = 0.25


# =========================================================
# HIGHER TIMEFRAME
# =========================================================

# 1H / 4H are only light context.
HTF_ZONE_DISTANCE_ATR_1H = 0.45


# =========================================================
# SCANNER LOOP
# =========================================================

SCAN_INTERVAL_SECONDS = int(
    os.environ.get("SCAN_INTERVAL_SECONDS", "60")
)

LOOP_SCANS = int(
    os.environ.get("LOOP_SCANS", "5")
)

STATE_FILE = os.environ.get(
    "STATE_FILE",
    "state.json",
)

BAGHDAD = ZoneInfo("Asia/Baghdad")


# =========================================================
# HELPERS
# =========================================================

def now_utc():
    return pd.Timestamp.now(tz="UTC")


def stamp(value):
    t = pd.Timestamp(value)

    if t.tzinfo is None:
        return t.tz_localize("UTC")

    return t.tz_convert("UTC")


def fmt(x):
    if x is None:
        return "-"

    x = float(x)

    if abs(x) >= 1000:
        return f"{x:,.2f}"

    if abs(x) >= 100:
        return f"{x:.2f}"

    return f"{x:.4f}"


def send_telegram(text):
    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=25,
    )

    response.raise_for_status()


# =========================================================
# STATE
# =========================================================

def default_state():
    return {
        "last_signal": {},
        "watch_setup": {},
        "active_trade": {},
        "closed": [],
    }


def load_state():
    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            st = json.load(f)

        if not isinstance(st, dict):
            return default_state()

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ):
        return default_state()

    st.setdefault("last_signal", {})
    st.setdefault("watch_setup", {})
    st.setdefault("active_trade", {})
    st.setdefault("closed", [])

    return st


def save_state(st):
    temp_file = STATE_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            st,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_file,
        STATE_FILE,
    )


# =========================================================
# MARKET HOURS
# =========================================================

def market_allowed(cfg, now):
    if not cfg["weekdays_only"]:
        return True

    return now.weekday() < 5


# =========================================================
# YAHOO DATA
# =========================================================

def normalize_yf(df):
    if df is None or df.empty:
        raise RuntimeError(
            "Empty market data"
        )

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = [
            column[0]
            for column in df.columns
        ]

    df = df.rename(
        columns={
            "Open": "o",
            "High": "h",
            "Low": "l",
            "Close": "c",
            "Volume": "v",
        }
    )

    required = [
        "o",
        "h",
        "l",
        "c",
        "v",
    ]

    df = (
        df[required]
        .dropna(
            subset=[
                "o",
                "h",
                "l",
                "c",
            ]
        )
        .copy()
    )

    if df.index.tz is None:
        df.index = pd.to_datetime(
            df.index,
            utc=True,
        )
    else:
        df.index = (
            pd.to_datetime(df.index)
            .tz_convert("UTC")
        )

    df["v"] = (
        pd.to_numeric(
            df["v"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return df.sort_index()


def closed_only(
    df,
    minutes,
):
    now = now_utc()

    candle_end = (
        df.index
        + pd.Timedelta(
            minutes=minutes
        )
    )

    return df[
        candle_end <= now
    ].copy()


# =========================================================
# GET FRAMES
# =========================================================

def get_frames(ticker):

    # -------------------------
    # 15M
    # -------------------------

    d15 = yf.download(
        ticker,
        period="5d",
        interval="15m",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    d15 = normalize_yf(d15)

    d15 = closed_only(
        d15,
        15,
    )

    # -------------------------
    # 1H
    # -------------------------

    d1 = yf.download(
        ticker,
        period="60d",
        interval="1h",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    d1 = normalize_yf(d1)

    d1 = closed_only(
        d1,
        60,
    )

    # -------------------------
    # 4H
    # -------------------------

    d4 = (
        d1.resample(
            "4h",
            label="left",
            closed="left",
        )
        .agg(
            {
                "o": "first",
                "h": "max",
                "l": "min",
                "c": "last",
                "v": "sum",
            }
        )
        .dropna()
    )

    d4 = closed_only(
        d4,
        240,
    )

    if (
        len(d15) < 80
        or len(d1) < 60
        or len(d4) < 20
    ):
        raise RuntimeError(
            "Not enough closed candles"
        )

    return {
        "15m": d15,
        "1h": d1,
        "4h": d4,
    }


# =========================================================
# LIVE PRICE
# =========================================================

def get_live_price(
    key,
    ticker,
):

    # BTC:
    # Binance public price is much faster than Yahoo.
    if key == "BTCUSD":

        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={
                    "symbol": "BTCUSDT",
                },
                timeout=10,
            )

            response.raise_for_status()

            return float(
                response.json()["price"]
            )

        except Exception:
            pass

    # Gold / Nasdaq fallback:
    # Yahoo 1-minute data.

    df = yf.download(
        ticker,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    df = normalize_yf(df)

    if df.empty:
        raise RuntimeError(
            "Live price unavailable"
        )

    return float(
        df["c"].iloc[-1]
    )


# =========================================================
# INDICATORS
# =========================================================

def ema(
    series,
    period,
):
    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def atr(
    df,
    period=ATR_PERIOD,
):
    previous_close = (
        df["c"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            df["h"] - df["l"],

            (
                df["h"]
                - previous_close
            ).abs(),

            (
                df["l"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    return true_range.rolling(
        period
    ).mean()


# =========================================================
# TREND
# =========================================================

def trend(df):
    close = df["c"]

    ema20 = float(
        ema(
            close,
            20,
        ).iloc[-1]
    )

    ema50 = float(
        ema(
            close,
            50,
        ).iloc[-1]
    )

    price = float(
        close.iloc[-1]
    )

    if (
        price > ema20
        and ema20 > ema50
    ):
        return 1

    if (
        price < ema20
        and ema20 < ema50
    ):
        return -1

    return 0


def trend_name(value):
    names = {
        1: "BULLISH",
        -1: "BEARISH",
        0: "RANGE",
    }

    return names[value]


# =========================================================
# VOLUME
# =========================================================

def volume_info(d15):
    volume = (
        d15["v"]
        .astype(float)
    )

    average = float(
        volume
        .iloc[-21:-1]
        .mean()
    )

    if average <= 0:
        return {
            "ratio": None,
            "label": "N/A",
        }

    ratio = (
        float(volume.iloc[-1])
        / average
    )

    if ratio >= 1.50:
        label = "HIGH"

    elif ratio >= 1.10:
        label = "GOOD"

    elif ratio >= 0.75:
        label = "NORMAL"

    else:
        label = "LOW"

    return {
        "ratio": ratio,
        "label": label,
    }


# =========================================================
# CISD
# =========================================================

def cisd_confirm(
    d15,
    side,
):
    """
    CISD is only a bonus.
    It is NOT mandatory.
    """

    x = d15.tail(12)

    if len(x) < 6:
        return False

    opens = x["o"].values
    closes = x["c"].values

    start = len(x) - 2
    stop = max(
        0,
        len(x) - 7,
    )

    for i in range(
        start,
        stop,
        -1,
    ):

        if side == 1:
            opposite = (
                closes[i]
                < opens[i]
            )

        else:
            opposite = (
                closes[i]
                > opens[i]
            )

        if opposite:

            level = float(
                opens[i]
            )

            if side == 1:
                return bool(
                    closes[-1]
                    > level
                )

            return bool(
                closes[-1]
                < level
            )

    return False


# =========================================================
# HIGHER TIMEFRAME LEVELS
# =========================================================

def recent_levels(
    df,
    lookback,
):
    x = df.tail(
        lookback
    )

    return {
        "support": float(
            x["l"].min()
        ),

        "resistance": float(
            x["h"].max()
        ),
    }


# =========================================================
# HTF CONTEXT
# =========================================================

def htf_context(
    frames,
    side,
    price,
):

    d1 = frames["1h"]
    d4 = frames["4h"]

    trend_1h = trend(d1)
    trend_4h = trend(d4)

    atr_1h = float(
        atr(d1).iloc[-1]
    )

    if (
        not math.isfinite(atr_1h)
        or atr_1h <= 0
    ):
        atr_1h = max(
            price * 0.002,
            1e-9,
        )

    levels_1h = recent_levels(
        d1,
        24,
    )

    levels_4h = recent_levels(
        d4,
        20,
    )

    # -------------------------
    # LONG
    # -------------------------

    if side == 1:

        levels = [
            levels_1h["support"],
            levels_4h["support"],
        ]

        distances = [
            price - level
            for level in levels
            if level <= price
        ]

        nearest = (
            min(distances)
            if distances
            else float("inf")
        )

        near_zone = (
            nearest
            <= (
                HTF_ZONE_DISTANCE_ATR_1H
                * atr_1h
            )
        )

        zone_text = (
            f"Support 1H "
            f"{fmt(levels_1h['support'])}"
            f" | "
            f"Support 4H "
            f"{fmt(levels_4h['support'])}"
        )

    # -------------------------
    # SHORT
    # -------------------------

    else:

        levels = [
            levels_1h["resistance"],
            levels_4h["resistance"],
        ]

        distances = [
            level - price
            for level in levels
            if level >= price
        ]

        nearest = (
            min(distances)
            if distances
            else float("inf")
        )

        near_zone = (
            nearest
            <= (
                HTF_ZONE_DISTANCE_ATR_1H
                * atr_1h
            )
        )

        zone_text = (
            f"Resistance 1H "
            f"{fmt(levels_1h['resistance'])}"
            f" | "
            f"Resistance 4H "
            f"{fmt(levels_4h['resistance'])}"
        )

    aligned_1h = (
        trend_1h == side
    )

    aligned_4h = (
        trend_4h == side
    )

    strong_against = (
        trend_1h == -side
        and trend_4h == -side
    )

    return {
        "t1": trend_1h,
        "t4": trend_4h,

        "aligned_1h": aligned_1h,
        "aligned_4h": aligned_4h,

        "near_zone": near_zone,

        "strong_against": (
            strong_against
        ),

        "zone_text": zone_text,
    }


# =========================================================
# ENTRY ZONE
# =========================================================

def calculate_entry_zone(
    side,
    close,
    atr15,
):

    # LONG
    if side == 1:

        low = (
            close
            - ENTRY_BACK_ATR
            * atr15
        )

        high = (
            close
            + ENTRY_FRONT_ATR
            * atr15
        )

    # SHORT
    else:

        low = (
            close
            - ENTRY_FRONT_ATR
            * atr15
        )

        high = (
            close
            + ENTRY_BACK_ATR
            * atr15
        )

    return (
        float(low),
        float(high),
    )


# =========================================================
# BUILD TRADE PLAN
# =========================================================

def build_trade_plan(
    side,
    setup_type,
    d15,
    reference_level,
    atr15,
):

    last = d15.iloc[-1]

    entry = float(
        last["c"]
    )

    # -------------------------
    # LONG
    # -------------------------

    if side == 1:

        if setup_type == "SWEEP":

            stop = (
                float(last["l"])
                - SL_BUFFER_ATR
                * atr15
            )

        else:

            stop = (
                min(
                    float(last["l"]),
                    reference_level,
                )
                - SL_BUFFER_ATR
                * atr15
            )

        risk = (
            entry - stop
        )

    # -------------------------
    # SHORT
    # -------------------------

    else:

        if setup_type == "SWEEP":

            stop = (
                float(last["h"])
                + SL_BUFFER_ATR
                * atr15
            )

        else:

            stop = (
                max(
                    float(last["h"]),
                    reference_level,
                )
                + SL_BUFFER_ATR
                * atr15
            )

        risk = (
            stop - entry
        )

    if (
        not math.isfinite(risk)
        or risk <= 0
    ):
        return None

    target_1 = (
        entry
        + side
        * risk
        * MIN_RR
    )

    target_2 = (
        entry
        + side
        * risk
        * 2.0
    )

    target_3 = (
        entry
        + side
        * risk
        * 3.0
    )

    zone_low, zone_high = (
        calculate_entry_zone(
            side,
            entry,
            atr15,
        )
    )

    return {
        "entry": entry,

        "zone_low": zone_low,
        "zone_high": zone_high,

        "stop": float(stop),

        "tps": [
            float(target_1),
            float(target_2),
            float(target_3),
        ],

        "risk": float(risk),

        "atr15": float(atr15),

        "retest_confirm_buffer": float(
            RETEST_CONFIRM_ATR
            * atr15
        ),
    }


# =========================================================
# DETECT SETUP
# =========================================================

def detect_signal(
    key,
    frames,
):

    d15 = frames["15m"]

    last = d15.iloc[-1]

    previous = (
        d15
        .iloc[-21:-1]
    )

    atr15 = float(
        atr(d15).iloc[-1]
    )

    if (
        not math.isfinite(atr15)
        or atr15 <= 0
    ):
        return None

    previous_high = float(
        previous["h"].max()
    )

    previous_low = float(
        previous["l"].min()
    )

    close = float(
        last["c"]
    )

    high = float(
        last["h"]
    )

    low = float(
        last["l"]
    )

    # =====================================================
    # SWEEP
    # =====================================================

    sweep_low = (
        low
        < (
            previous_low
            - SWEEP_BUFFER_ATR
            * atr15
        )
        and close > previous_low
    )

    sweep_high = (
        high
        > (
            previous_high
            + SWEEP_BUFFER_ATR
            * atr15
        )
        and close < previous_high
    )

    # =====================================================
    # BREAKOUT
    # =====================================================

    break_up = (
        close
        > (
            previous_high
            + BREAK_BUFFER_ATR
            * atr15
        )
    )

    break_down = (
        close
        < (
            previous_low
            - BREAK_BUFFER_ATR
            * atr15
        )
    )

    # =====================================================
    # PICK SETUP
    # =====================================================

    if sweep_low:

        side = 1

        setup_type = "SWEEP"

        setup_name = (
            "Liquidity Low Sweep"
        )

        reference_level = (
            previous_low
        )

        score = 6.0

    elif sweep_high:

        side = -1

        setup_type = "SWEEP"

        setup_name = (
            "Liquidity High Sweep"
        )

        reference_level = (
            previous_high
        )

        score = 6.0

    elif break_up:

        side = 1

        setup_type = "BREAKOUT"

        setup_name = "Breakout"

        reference_level = (
            previous_high
        )

        score = 6.3

    elif break_down:

        side = -1

        setup_type = "BREAKOUT"

        setup_name = "Breakout"

        reference_level = (
            previous_low
        )

        score = 6.3

    else:
        return None

    # =====================================================
    # EXTRA CONTEXT
    # =====================================================

    volume = volume_info(
        d15
    )

    htf = htf_context(
        frames,
        side,
        close,
    )

    cisd = cisd_confirm(
        d15,
        side,
    )

    # =====================================================
    # LIGHT SCORE
    # =====================================================

    # 1H alignment
    if htf["aligned_1h"]:
        score += 0.8

    # 4H alignment
    if htf["aligned_4h"]:
        score += 0.5

    # Near 1H/4H important zone
    if htf["near_zone"]:
        score += 0.8

    # Both higher frames strongly against trade
    if htf["strong_against"]:
        score -= 1.2

    # Volume
    if volume["ratio"] is not None:

        if volume["ratio"] >= 1.50:
            score += 0.8

        elif volume["ratio"] >= 1.10:
            score += 0.4

        elif volume["ratio"] < 0.65:
            score -= 0.4

    # CISD = bonus only
    if cisd:
        score += 0.5

    score = round(
        max(
            0.0,
            min(
                10.0,
                score,
            ),
        ),
        1,
    )

    # =====================================================
    # GRADE
    # =====================================================

    if (
        score >= A_SCORE
        and not htf["strong_against"]
    ):
        grade = "A"

    elif score >= B_SCORE:
        grade = "B"

    else:
        return None

    # =====================================================
    # PLAN
    # =====================================================

    plan = build_trade_plan(
        side,
        setup_type,
        d15,
        reference_level,
        atr15,
    )

    if not plan:
        return None

    candle_start = stamp(
        d15.index[-1]
    )

    candle_end = (
        candle_start
        + pd.Timedelta(
            minutes=15
        )
    )

    signal_id = (
        f"{key}:"
        f"{setup_type}:"
        f"{'L' if side == 1 else 'S'}:"
        f"{candle_end.isoformat()}"
    )

    return {
        "id": signal_id,

        "key": key,

        "side": side,

        "setup_type": (
            setup_type
        ),

        "setup_name": (
            setup_name
        ),

        "score": score,

        "grade": grade,

        "volume": volume,

        "htf": htf,

        "cisd": cisd,

        "plan": plan,

        "confirmation_end": (
            candle_end
        ),
    }


# =========================================================
# INITIAL ENTRY DECISION
# =========================================================

def initial_entry_status(
    signal,
    live_price,
    now,
):

    plan = signal["plan"]

    age_seconds = max(
        0.0,
        (
            now
            - signal["confirmation_end"]
        ).total_seconds(),
    )

    inside = (
        plan["zone_low"]
        <= live_price
        <= plan["zone_high"]
    )

    # =====================================================
    # B SETUP
    # =====================================================

    # B never gets immediate ENTER NOW.
    if signal["grade"] == "B":

        return {
            "status": (
                "WAIT FOR CONFIRMATION"
            ),

            "tradable": False,

            "watch": True,

            "age_seconds": (
                age_seconds
            ),
        }

    # =====================================================
    # A SETUP
    # =====================================================

    # Fresh + still inside zone.
    if (
        age_seconds
        <= MAX_ENTER_AGE_SECONDS
        and inside
    ):

        return {
            "status": "ENTER NOW",

            "tradable": True,

            "watch": False,

            "age_seconds": (
                age_seconds
            ),
        }

    # Setup is still valid,
    # but direct entry is missed.
    if inside:

        status = (
            "WAIT FOR RETEST CONFIRMATION"
        )

    else:

        status = (
            "WAIT FOR RETEST — DO NOT CHASE"
        )

    return {
        "status": status,

        "tradable": False,

        "watch": True,

        "age_seconds": (
            age_seconds
        ),
    }


# =========================================================
# STORE WATCH SETUP
# =========================================================

def signal_to_watch(
    signal,
    live_price,
    now,
):

    plan = signal["plan"]

    inside = (
        plan["zone_low"]
        <= live_price
        <= plan["zone_high"]
    )

    return {
        "id": signal["id"],

        "key": signal["key"],

        "side": signal["side"],

        "setup_name": (
            signal["setup_name"]
        ),

        "grade": signal["grade"],

        "score": signal["score"],

        "plan": plan,

        "confirmation_end": (
            signal["confirmation_end"]
            .isoformat()
        ),

        "created_at": (
            now.isoformat()
        ),

        "expires_at": (
            signal["confirmation_end"]
            + pd.Timedelta(
                minutes=(
                    WATCH_EXPIRY_MINUTES
                )
            )
        ).isoformat(),

        # If signal arrived while already
        # back inside zone, consider zone touched.
        "touched": bool(inside),

        "touch_time": (
            now.isoformat()
            if inside
            else None
        ),
    }


# =========================================================
# RETEST CONFIRM LEVEL
# =========================================================

def retest_confirmation_level(
    watch,
):

    plan = watch["plan"]

    buffer_value = (
        plan[
            "retest_confirm_buffer"
        ]
    )

    # LONG
    if watch["side"] == 1:

        return (
            plan["zone_high"]
            + buffer_value
        )

    # SHORT
    return (
        plan["zone_low"]
        - buffer_value
    )


# =========================================================
# OPEN PAPER TRADE
# =========================================================

def open_trade(
    st,
    signal,
    live_price,
    now,
):

    plan = signal["plan"]

    key = signal["key"]

    st["active_trade"][key] = {
        "id": signal["id"],

        "side": signal["side"],

        "setup": (
            signal["setup_name"]
        ),

        "entry": float(
            live_price
        ),

        "stop": plan["stop"],

        "tps": plan["tps"],

        "hits": [
            False,
            False,
            False,
        ],

        "opened_at": (
            now.isoformat()
        ),

        "state": "OPEN",
    }


def open_trade_from_watch(
    st,
    watch,
    live_price,
    now,
):

    plan = watch["plan"]

    key = watch["key"]

    st["active_trade"][key] = {
        "id": watch["id"],

        "side": watch["side"],

        "setup": (
            watch["setup_name"]
            + " RETEST"
        ),

        "entry": float(
            live_price
        ),

        "stop": plan["stop"],

        "tps": plan["tps"],

        "hits": [
            False,
            False,
            False,
        ],

        "opened_at": (
            now.isoformat()
        ),

        "state": "OPEN",
    }


# =========================================================
# PROCESS WAITING RETEST
# =========================================================

def process_watch(
    st,
    key,
    live_price,
    now,
):

    watch = (
        st["watch_setup"]
        .get(key)
    )

    if not watch:
        return None, False

    plan = watch["plan"]

    side = int(
        watch["side"]
    )

    # =====================================================
    # EXPIRY
    # =====================================================

    if now > stamp(
        watch["expires_at"]
    ):

        st["watch_setup"].pop(
            key,
            None,
        )

        return None, False

    # =====================================================
    # INVALIDATION
    # =====================================================

    if side == 1:

        stop_breached = (
            live_price
            <= plan["stop"]
        )

    else:

        stop_breached = (
            live_price
            >= plan["stop"]
        )

    if stop_breached:

        st["watch_setup"].pop(
            key,
            None,
        )

        message = (
            f"❌ {key} | SETUP INVALIDATED\n"
            f"SETUP: {watch['setup_name']}\n"
            f"GRADE: {watch['grade']} | "
            f"SCORE: {watch['score']}/10\n"
            f"LIVE: {fmt(live_price)}\n"
            f"Reason: price breached "
            f"invalidation / SL area\n"
            f"SETUP ID: {watch['id']}"
        )

        return message, False

    # =====================================================
    # CHECK ZONE
    # =====================================================

    inside = (
        plan["zone_low"]
        <= live_price
        <= plan["zone_high"]
    )

    # Price returned to zone.
    if (
        inside
        and not watch.get(
            "touched",
            False,
        )
    ):

        watch["touched"] = True

        watch["touch_time"] = (
            now.isoformat()
        )

        st["watch_setup"][key] = (
            watch
        )

        message = (
            f"🟡 {key} | RETEST IN ENTRY ZONE\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"SETUP: {watch['setup_name']}\n"
            f"GRADE: {watch['grade']} | "
            f"SCORE: {watch['score']}/10\n"
            f"ENTRY ZONE: "
            f"{fmt(plan['zone_low'])}"
            f" — "
            f"{fmt(plan['zone_high'])}\n"
            f"LIVE: {fmt(live_price)}\n"
            f"STATUS: "
            f"WAIT FOR RETEST CONFIRMATION\n"
            f"SETUP ID: {watch['id']}"
        )

        return message, False

    # Still hasn't returned to zone.
    if not watch.get(
        "touched",
        False,
    ):
        return None, False

    # =====================================================
    # RETEST CONFIRMATION
    # =====================================================

    confirmation_level = (
        retest_confirmation_level(
            watch
        )
    )

    if side == 1:

        confirmed = (
            live_price
            >= confirmation_level
        )

    else:

        confirmed = (
            live_price
            <= confirmation_level
        )

    if not confirmed:
        return None, False

    # =====================================================
    # DO NOT CHASE AFTER RETEST
    # =====================================================

    if side == 1:

        zone_edge = (
            plan["zone_high"]
        )

        move_from_zone = (
            live_price
            - zone_edge
        )

    else:

        zone_edge = (
            plan["zone_low"]
        )

        move_from_zone = (
            zone_edge
            - live_price
        )

    max_chase = (
        MAX_RETEST_CHASE_R
        * plan["risk"]
    )

    # Confirmation happened,
    # but price already ran too far.
    if move_from_zone > max_chase:

        # Require another fresh return.
        watch["touched"] = False

        watch["touch_time"] = None

        st["watch_setup"][key] = (
            watch
        )

        message = (
            f"⚠️ {key} | RETEST MOVED TOO FAR\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Confirmation happened but "
            f"price already moved too far.\n"
            f"DO NOT CHASE.\n"
            f"Waiting for another retest.\n"
            f"ENTRY ZONE: "
            f"{fmt(plan['zone_low'])}"
            f" — "
            f"{fmt(plan['zone_high'])}\n"
            f"LIVE: {fmt(live_price)}\n"
            f"SETUP ID: {watch['id']}"
        )

        return message, False

    # =====================================================
    # VALID RETEST ENTRY
    # =====================================================

    st["watch_setup"].pop(
        key,
        None,
    )

    open_trade_from_watch(
        st,
        watch,
        live_price,
        now,
    )

    current_risk = abs(
        live_price
        - plan["stop"]
    )

    if current_risk <= 0:
        current_risk = 1e-9

    rr_tp1 = (
        abs(
            plan["tps"][0]
            - live_price
        )
        / current_risk
    )

    message = (
        f"✅ {key} | ENTER ON RETEST\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"SETUP: {watch['setup_name']}\n"
        f"GRADE: {watch['grade']} | "
        f"SCORE: {watch['score']}/10\n"
        f"\n"
        f"ENTRY: {fmt(live_price)}\n"
        f"SL: {fmt(plan['stop'])}\n"
        f"TP1: {fmt(plan['tps'][0])}\n"
        f"TP2: {fmt(plan['tps'][1])}\n"
        f"TP3: {fmt(plan['tps'][2])}\n"
        f"R:R TO TP1: {rr_tp1:.2f}R\n"
        f"\n"
        f"SETUP ID: {watch['id']}"
    )

    return message, True


# =========================================================
# CLOSE PAPER TRADE
# =========================================================

def close_trade(
    st,
    key,
    reason,
    exit_price,
    now,
):

    trade = (
        st["active_trade"]
        .get(key)
    )

    if not trade:
        return

    closed_trade = dict(
        trade
    )

    closed_trade["state"] = (
        "CLOSED"
    )

    closed_trade["exit_reason"] = (
        reason
    )

    closed_trade["exit_price"] = (
        float(exit_price)
    )

    closed_trade["closed_at"] = (
        now.isoformat()
    )

    st["closed"].append(
        closed_trade
    )

    st["active_trade"].pop(
        key,
        None,
    )


# =========================================================
# MONITOR PAPER TRADE
# =========================================================

def monitor_open_trade(
    st,
    key,
    live_price,
    now,
):

    trade = (
        st["active_trade"]
        .get(key)
    )

    if not trade:
        return None

    side = trade["side"]

    # =====================================================
    # STOP
    # =====================================================

    if side == 1:

        stop_hit = (
            live_price
            <= trade["stop"]
        )

    else:

        stop_hit = (
            live_price
            >= trade["stop"]
        )

    if stop_hit:

        close_trade(
            st,
            key,
            "STOP",
            live_price,
            now,
        )

        return (
            f"❌ {key} | "
            f"{trade['id']}\n"
            f"STOP HIT @ "
            f"{fmt(live_price)}"
        )

    # =====================================================
    # TARGETS
    # =====================================================

    for i, target in enumerate(
        trade["tps"]
    ):

        if trade["hits"][i]:
            continue

        if side == 1:

            target_hit = (
                live_price
                >= target
            )

        else:

            target_hit = (
                live_price
                <= target
            )

        if target_hit:

            trade["hits"][i] = (
                True
            )

            # TP3 = fully closed paper trade.
            if i == 2:

                close_trade(
                    st,
                    key,
                    "TP3",
                    live_price,
                    now,
                )

            return (
                f"🎯 {key} | "
                f"{trade['id']}\n"
                f"TP{i + 1} HIT @ "
                f"{fmt(live_price)}"
            )

    return None


# =========================================================
# FORMAT INITIAL SIGNAL
# =========================================================

def format_signal(
    signal,
    live_price,
    entry_status,
):

    side = (
        "LONG"
        if signal["side"] == 1
        else "SHORT"
    )

    plan = signal["plan"]

    htf = signal["htf"]

    volume = signal["volume"]

    confirmation_time = (
        signal["confirmation_end"]
        .tz_convert(BAGHDAD)
    )

    age_seconds = int(
        entry_status[
            "age_seconds"
        ]
    )

    volume_text = (
        volume["label"]
    )

    if volume["ratio"] is not None:

        volume_text += (
            f" "
            f"({volume['ratio']:.2f}x)"
        )

    near_htf = (
        "YES"
        if htf["near_zone"]
        else "NO"
    )

    cisd_text = (
        "YES"
        if signal["cisd"]
        else "NO"
    )

    return (
        f"🚨 {signal['key']} {side}\n"
        f"━━━━━━━━━━━━━━━━━━\n"

        f"STATUS: "
        f"{entry_status['status']}\n"

        f"SETUP: "
        f"{signal['setup_name']}\n"

        f"GRADE: "
        f"{signal['grade']} | "
        f"SCORE: "
        f"{signal['score']}/10\n"

        f"\n"

        f"ENTRY ZONE: "
        f"{fmt(plan['zone_low'])}"
        f" — "
        f"{fmt(plan['zone_high'])}\n"

        f"LIVE: "
        f"{fmt(live_price)}\n"

        f"SL: "
        f"{fmt(plan['stop'])}\n"

        f"TP1: "
        f"{fmt(plan['tps'][0])} "
        f"(1.5R reference)\n"

        f"TP2: "
        f"{fmt(plan['tps'][1])} "
        f"(2.0R reference)\n"

        f"TP3: "
        f"{fmt(plan['tps'][2])} "
        f"(3.0R reference)\n"

        f"\n"

        f"15M CONFIRMATION CLOSE: "
        f"{confirmation_time:%Y-%m-%d %H:%M} "
        f"Baghdad\n"

        f"ALERT AGE: "
        f"{age_seconds}s\n"

        f"VOLUME: "
        f"{volume_text}\n"

        f"1H: "
        f"{trend_name(htf['t1'])}"
        f" | "
        f"4H: "
        f"{trend_name(htf['t4'])}\n"

        f"NEAR HTF ZONE: "
        f"{near_htf}\n"

        f"{htf['zone_text']}\n"

        f"CISD: "
        f"{cisd_text}\n"

        f"\n"

        f"SETUP ID: "
        f"{signal['id']}"
    )


# =========================================================
# SCAN ONE MARKET
# =========================================================

def scan_market(
    st,
    key,
    cfg,
):

    now = now_utc()

    if not market_allowed(
        cfg,
        now,
    ):
        return

    # =====================================================
    # DATA
    # =====================================================

    try:

        frames = get_frames(
            cfg["ticker"]
        )

        live_price = get_live_price(
            key,
            cfg["ticker"],
        )

    except Exception as error:

        print(
            key,
            "data error:",
            error,
        )

        return

    # =====================================================
    # ACTIVE TRADE
    # =====================================================

    trade_message = (
        monitor_open_trade(
            st,
            key,
            live_price,
            now,
        )
    )

    if trade_message:

        send_telegram(
            trade_message
        )

    if key in st["active_trade"]:
        return

    # =====================================================
    # OLD SETUP / RETEST
    # =====================================================

    watch_message, entered = (
        process_watch(
            st,
            key,
            live_price,
            now,
        )
    )

    if watch_message:

        send_telegram(
            watch_message
        )

    if (
        entered
        or key in st["active_trade"]
    ):
        return

    # =====================================================
    # NEW 15M SETUP
    # =====================================================

    signal = detect_signal(
        key,
        frames,
    )

    if not signal:
        return

    # Don't resend same initial candle signal.
    if (
        st["last_signal"].get(key)
        == signal["id"]
    ):
        return

    st["last_signal"][key] = (
        signal["id"]
    )

    status = (
        initial_entry_status(
            signal,
            live_price,
            now,
        )
    )

    # =====================================================
    # SEND INITIAL ALERT
    # =====================================================

    send_telegram(
        format_signal(
            signal,
            live_price,
            status,
        )
    )

    # =====================================================
    # DIRECT ENTER NOW
    # =====================================================

    if status["tradable"]:

        st["watch_setup"].pop(
            key,
            None,
        )

        open_trade(
            st,
            signal,
            live_price,
            now,
        )

        return

    # =====================================================
    # WAIT FOR RETEST / CONFIRMATION
    # =====================================================

    if status["watch"]:

        st["watch_setup"][key] = (
            signal_to_watch(
                signal,
                live_price,
                now,
            )
        )


# =========================================================
# ONE FULL SCAN
# =========================================================

def run_once():

    st = load_state()

    for key, cfg in MARKETS.items():

        try:

            scan_market(
                st,
                key,
                cfg,
            )

        except Exception as error:

            print(
                key,
                "scan error:",
                error,
            )

    save_state(st)


# =========================================================
# MAIN
# =========================================================

def main():

    scans = max(
        1,
        LOOP_SCANS,
    )

    for i in range(scans):

        started = time.time()

        print(
            f"scan "
            f"{i + 1}/"
            f"{scans}"
        )

        run_once()

        if i < scans - 1:

            spent = (
                time.time()
                - started
            )

            sleep_seconds = max(
                1,
                SCAN_INTERVAL_SECONDS
                - spent,
            )

            time.sleep(
                sleep_seconds
            )


if __name__ == "__main__":
    main()
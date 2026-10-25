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
TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]

MARKETS = {
    "XAUUSD": {
        "source": "twelve",
        "symbol": "XAU/USD",
        "weekdays_only": True,
    },
    "NASDAQ": {
        "source": "yahoo",
        "ticker": "NQ=F",
        "weekdays_only": True,
    },
    "BTCUSD": {
        "source": "binance",
        "symbol": "BTCUSDT",
        "weekdays_only": False,
    },
    "ETHUSD": {
        "source": "binance",
        "symbol": "ETHUSDT",
        "weekdays_only": False,
    },
    "SOLUSD": {
        "source": "binance",
        "symbol": "SOLUSDT",
        "weekdays_only": False,
    },
}

ATR_PERIOD = 14
MIN_RR = 1.5

A_SCORE = 7.5
B_SCORE = 6.0

BREAK_BUFFER_ATR = 0.05
SWEEP_BUFFER_ATR = 0.02
SL_BUFFER_ATR = 0.15

ENTRY_BACK_ATR = 0.12
ENTRY_FRONT_ATR = 0.04

MAX_ENTER_AGE_SECONDS = 120
WATCH_EXPIRY_MINUTES = 60

RETEST_CONFIRM_ATR = 0.05
MAX_RETEST_CHASE_R = 0.25

HTF_ZONE_DISTANCE_ATR_1H = 0.45

SCAN_INTERVAL_SECONDS = int(
    os.environ.get(
        "SCAN_INTERVAL_SECONDS",
        "60",
    )
)

LOOP_SCANS = int(
    os.environ.get(
        "LOOP_SCANS",
        "5",
    )
)

STATE_FILE = os.environ.get(
    "STATE_FILE",
    "state.json",
)

BAGHDAD = ZoneInfo(
    "Asia/Baghdad"
)

SESSION = requests.Session()


# =========================================================
# BASIC HELPERS
# =========================================================

def now_utc():
    return pd.Timestamp.now(
        tz="UTC"
    )


def stamp(value):
    t = pd.Timestamp(value)

    if t.tzinfo is None:
        return t.tz_localize(
            "UTC"
        )

    return t.tz_convert(
        "UTC"
    )


def fmt(x):
    if x is None:
        return "-"

    x = float(x)

    if abs(x) >= 1000:
        return f"{x:,.2f}"

    if abs(x) >= 100:
        return f"{x:.2f}"

    return f"{x:.4f}"


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(text):

    response = SESSION.post(
        (
            "https://api.telegram.org/"
            f"bot{TELEGRAM_BOT_TOKEN}/"
            "sendMessage"
        ),
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

        if not isinstance(
            st,
            dict,
        ):
            return default_state()

    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ):

        return default_state()

    defaults = default_state()

    for key, value in defaults.items():

        st.setdefault(
            key,
            value,
        )

    return st


def save_state(st):

    temp_file = (
        STATE_FILE + ".tmp"
    )

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

def market_allowed(
    cfg,
    now,
):

    if not cfg["weekdays_only"]:
        return True

    return (
        now.weekday() < 5
    )


# =========================================================
# YAHOO DATA
# =========================================================

def normalize_yf(df):

    if (
        df is None
        or df.empty
    ):

        raise RuntimeError(
            "Empty Yahoo data"
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

    df = (
        df[
            [
                "o",
                "h",
                "l",
                "c",
                "v",
            ]
        ]
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

    df.index = pd.to_datetime(
        df.index,
        utc=True,
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
# VALIDATE FRAMES
# =========================================================

def validate_frames(
    d15,
    d1,
    d4,
):

    if (
        len(d15) < 80
        or len(d1) < 60
        or len(d4) < 20
    ):

        raise RuntimeError(
            "Not enough closed candles: "
            f"15m={len(d15)} "
            f"1h={len(d1)} "
            f"4h={len(d4)}"
        )

    return {
        "15m": d15,
        "1h": d1,
        "4h": d4,
    }


# =========================================================
# YAHOO FRAMES - NASDAQ
# =========================================================

def yahoo_frames(
    ticker,
):

    d15 = yf.download(
        ticker,
        period="5d",
        interval="15m",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    d15 = normalize_yf(
        d15
    )

    d15 = closed_only(
        d15,
        15,
    )

    d1 = yf.download(
        ticker,
        period="60d",
        interval="1h",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    d1 = normalize_yf(
        d1
    )

    d1 = closed_only(
        d1,
        60,
    )

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

    return validate_frames(
        d15,
        d1,
        d4,
    )


# =========================================================
# BINANCE DATA
# =========================================================

def binance_klines(
    symbol,
    interval,
    limit=300,
    start_ms=None,
):

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    if start_ms is not None:

        params["startTime"] = (
            start_ms
        )

    response = SESSION.get(
        "https://api.binance.com/api/v3/klines",
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    raw = response.json()

    rows = []

    for candle in raw:

        rows.append(
            {
                "time": pd.to_datetime(
                    int(candle[0]),
                    unit="ms",
                    utc=True,
                ),

                "end": pd.to_datetime(
                    int(candle[6]),
                    unit="ms",
                    utc=True,
                ),

                "o": float(
                    candle[1]
                ),

                "h": float(
                    candle[2]
                ),

                "l": float(
                    candle[3]
                ),

                "c": float(
                    candle[4]
                ),

                "v": float(
                    candle[5]
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    return (
        df.set_index(
            "time"
        )
        .sort_index()
    )


def binance_frames(
    symbol,
):

    now = now_utc()

    d15 = binance_klines(
        symbol,
        "15m",
        300,
    )

    d1 = binance_klines(
        symbol,
        "1h",
        300,
    )

    if (
        d15.empty
        or d1.empty
    ):

        raise RuntimeError(
            "Empty Binance data"
        )

    d15 = (
        d15[
            d15["end"] <= now
        ][
            [
                "o",
                "h",
                "l",
                "c",
                "v",
            ]
        ]
        .copy()
    )

    d1 = (
        d1[
            d1["end"] <= now
        ][
            [
                "o",
                "h",
                "l",
                "c",
                "v",
            ]
        ]
        .copy()
    )

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

    return validate_frames(
        d15,
        d1,
        d4,
    )


# =========================================================
# TWELVE DATA
# =========================================================

def twelve_request(
    endpoint,
    params,
):

    params = dict(
        params
    )

    params[
        "apikey"
    ] = TWELVE_DATA_API_KEY

    response = SESSION.get(
        (
            "https://api.twelvedata.com/"
            f"{endpoint}"
        ),
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if (
        isinstance(
            data,
            dict,
        )
        and data.get(
            "status"
        ) == "error"
    ):

        raise RuntimeError(
            data.get(
                "message",
                str(data),
            )
        )

    return data


def twelve_series(
    symbol,
    interval,
    outputsize,
):

    data = twelve_request(
        "time_series",
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "timezone": "UTC",
            "format": "JSON",
        },
    )

    values = (
        data.get(
            "values"
        )
        or []
    )

    if not values:

        raise RuntimeError(
            "Empty Twelve Data: "
            f"{symbol} {interval}"
        )

    rows = []

    for candle in values:

        rows.append(
            {
                "time": pd.to_datetime(
                    candle["datetime"],
                    utc=True,
                ),

                "o": float(
                    candle["open"]
                ),

                "h": float(
                    candle["high"]
                ),

                "l": float(
                    candle["low"]
                ),

                "c": float(
                    candle["close"]
                ),

                "v": float(
                    candle.get(
                        "volume"
                    )
                    or 0.0
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    return (
        df.set_index(
            "time"
        )
        .sort_index()
    )


# =========================================================
# XAUUSD SPOT FRAMES
# =========================================================

def twelve_frames(
    symbol,
):

    d15 = twelve_series(
        symbol,
        "15min",
        300,
    )

    d15 = closed_only(
        d15,
        15,
    )

    d1 = twelve_series(
        symbol,
        "1h",
        300,
    )

    d1 = closed_only(
        d1,
        60,
    )

    # Build 4H locally from 1H.
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

    return validate_frames(
        d15,
        d1,
        d4,
    )


# =========================================================
# UNIFIED FRAMES
# =========================================================

def get_frames(
    key,
    cfg,
):

    source = cfg[
        "source"
    ]

    if source == "binance":

        return binance_frames(
            cfg["symbol"]
        )

    if source == "twelve":

        return twelve_frames(
            cfg["symbol"]
        )

    return yahoo_frames(
        cfg["ticker"]
    )


# =========================================================
# LIVE PRICE
# =========================================================

def get_live_price(
    key,
    cfg,
):

    source = cfg[
        "source"
    ]

    if source == "binance":

        response = SESSION.get(
            (
                "https://api.binance.com/"
                "api/v3/ticker/price"
            ),
            params={
                "symbol": cfg[
                    "symbol"
                ],
            },
            timeout=10,
        )

        response.raise_for_status()

        return float(
            response.json()[
                "price"
            ]
        )

    if source == "twelve":

        data = twelve_request(
            "price",
            {
                "symbol": cfg[
                    "symbol"
                ],
                "format": "JSON",
            },
        )

        if "price" not in data:

            raise RuntimeError(
                "Twelve Data live "
                "price unavailable"
            )

        return float(
            data["price"]
        )

    df = yf.download(
        cfg["ticker"],
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    df = normalize_yf(
        df
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
        df["c"].shift(1)
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

    ema20 = float(
        ema(
            df["c"],
            20,
        ).iloc[-1]
    )

    ema50 = float(
        ema(
            df["c"],
            50,
        ).iloc[-1]
    )

    price = float(
        df["c"].iloc[-1]
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

    return {
        1: "BULLISH",
        -1: "BEARISH",
        0: "RANGE",
    }[value]


# =========================================================
# VOLUME
# =========================================================

def volume_info(
    d15,
):

    volume = (
        d15["v"]
        .astype(float)
    )

    average = float(
        volume
        .iloc[-21:-1]
        .mean()
    )

    # Spot gold may not have useful centralized volume.
    if average <= 0:

        return {
            "ratio": None,
            "label": "N/A",
        }

    ratio = (
        float(
            volume.iloc[-1]
        )
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

    x = d15.tail(
        12
    )

    if len(x) < 6:
        return False

    opens = (
        x["o"].values
    )

    closes = (
        x["c"].values
    )

    start = (
        len(x) - 2
    )

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
# HTF LEVELS
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

    d1 = frames[
        "1h"
    ]

    d4 = frames[
        "4h"
    ]

    trend_1h = trend(
        d1
    )

    trend_4h = trend(
        d4
    )

    atr_1h = float(
        atr(
            d1
        ).iloc[-1]
    )

    if (
        not math.isfinite(
            atr_1h
        )
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

    if side == 1:

        levels = [
            levels_1h[
                "support"
            ],
            levels_4h[
                "support"
            ],
        ]

        distances = [
            price - level
            for level in levels
            if level <= price
        ]

        zone_text = (
            f"Support 1H "
            f"{fmt(levels_1h['support'])}"
            f" | Support 4H "
            f"{fmt(levels_4h['support'])}"
        )

    else:

        levels = [
            levels_1h[
                "resistance"
            ],
            levels_4h[
                "resistance"
            ],
        ]

        distances = [
            level - price
            for level in levels
            if level >= price
        ]

        zone_text = (
            f"Resistance 1H "
            f"{fmt(levels_1h['resistance'])}"
            f" | Resistance 4H "
            f"{fmt(levels_4h['resistance'])}"
        )

    nearest = (
        min(
            distances
        )
        if distances
        else float(
            "inf"
        )
    )

    return {
        "t1": trend_1h,
        "t4": trend_4h,

        "aligned_1h": (
            trend_1h == side
        ),

        "aligned_4h": (
            trend_4h == side
        ),

        "near_zone": (
            nearest
            <= (
                HTF_ZONE_DISTANCE_ATR_1H
                * atr_1h
            )
        ),

        "strong_against": (
            trend_1h == -side
            and trend_4h == -side
        ),

        "zone_text": (
            zone_text
        ),
    }


# =========================================================
# ENTRY ZONE
# =========================================================

def calculate_entry_zone(
    side,
    close,
    atr15,
):

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

    if side == 1:

        if setup_type == "SWEEP":

            stop = (
                float(
                    last["l"]
                )
                - SL_BUFFER_ATR
                * atr15
            )

        else:

            stop = (
                min(
                    float(
                        last["l"]
                    ),
                    reference_level,
                )
                - SL_BUFFER_ATR
                * atr15
            )

        risk = (
            entry - stop
        )

    else:

        if setup_type == "SWEEP":

            stop = (
                float(
                    last["h"]
                )
                + SL_BUFFER_ATR
                * atr15
            )

        else:

            stop = (
                max(
                    float(
                        last["h"]
                    ),
                    reference_level,
                )
                + SL_BUFFER_ATR
                * atr15
            )

        risk = (
            stop - entry
        )

    if (
        not math.isfinite(
            risk
        )
        or risk <= 0
    ):
        return None

    targets = [
        (
            entry
            + side
            * risk
            * MIN_RR
        ),
        (
            entry
            + side
            * risk
            * 2.0
        ),
        (
            entry
            + side
            * risk
            * 3.0
        ),
    ]

    zone_low, zone_high = (
        calculate_entry_zone(
            side,
            entry,
            atr15,
        )
    )

    return {
        "entry": entry,

        "zone_low": (
            zone_low
        ),

        "zone_high": (
            zone_high
        ),

        "stop": float(
            stop
        ),

        "tps": [
            float(x)
            for x in targets
        ],

        "risk": float(
            risk
        ),

        "atr15": float(
            atr15
        ),

        "retest_confirm_buffer": float(
            RETEST_CONFIRM_ATR
            * atr15
        ),
    }


# =========================================================
# DETECT SIGNAL
# =========================================================

def detect_signal(
    key,
    frames,
):

    d15 = frames[
        "15m"
    ]

    last = (
        d15.iloc[-1]
    )

    previous = (
        d15.iloc[-21:-1]
    )

    atr15 = float(
        atr(
            d15
        ).iloc[-1]
    )

    if (
        not math.isfinite(
            atr15
        )
        or atr15 <= 0
    ):
        return None

    previous_high = float(
        previous[
            "h"
        ].max()
    )

    previous_low = float(
        previous[
            "l"
        ].min()
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

    sweep_low = (
        low
        < (
            previous_low
            - SWEEP_BUFFER_ATR
            * atr15
        )
        and close
        > previous_low
    )

    sweep_high = (
        high
        > (
            previous_high
            + SWEEP_BUFFER_ATR
            * atr15
        )
        and close
        < previous_high
    )

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
        setup_type = (
            "BREAKOUT"
        )
        setup_name = (
            "Breakout"
        )
        reference_level = (
            previous_high
        )
        score = 6.3

    elif break_down:

        side = -1
        setup_type = (
            "BREAKOUT"
        )
        setup_name = (
            "Breakout"
        )
        reference_level = (
            previous_low
        )
        score = 6.3

    else:

        return None

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

    if htf[
        "aligned_1h"
    ]:
        score += 0.8

    if htf[
        "aligned_4h"
    ]:
        score += 0.5

    if htf[
        "near_zone"
    ]:
        score += 0.8

    if htf[
        "strong_against"
    ]:
        score -= 1.2

    if volume[
        "ratio"
    ] is not None:

        if volume[
            "ratio"
        ] >= 1.50:

            score += 0.8

        elif volume[
            "ratio"
        ] >= 1.10:

            score += 0.4

        elif volume[
            "ratio"
        ] < 0.65:

            score -= 0.4

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

    if (
        score >= A_SCORE
        and not htf[
            "strong_against"
        ]
    ):

        grade = "A"

    elif score >= B_SCORE:

        grade = "B"

    else:

        return None

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
# INITIAL ENTRY STATUS
# =========================================================

def initial_entry_status(
    signal,
    live_price,
    now,
):

    plan = signal[
        "plan"
    ]

    age_seconds = max(
        0.0,
        (
            now
            - signal[
                "confirmation_end"
            ]
        ).total_seconds(),
    )

    inside = (
        plan[
            "zone_low"
        ]
        <= live_price
        <= plan[
            "zone_high"
        ]
    )

    if signal[
        "grade"
    ] == "B":

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

    if (
        age_seconds
        <= MAX_ENTER_AGE_SECONDS
        and inside
    ):

        return {
            "status": (
                "ENTER NOW"
            ),
            "tradable": True,
            "watch": False,
            "age_seconds": (
                age_seconds
            ),
        }

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
# WATCH SETUP
# =========================================================

def signal_to_watch(
    signal,
    live_price,
    now,
):

    plan = signal[
        "plan"
    ]

    inside = (
        plan[
            "zone_low"
        ]
        <= live_price
        <= plan[
            "zone_high"
        ]
    )

    return {
        "id": signal[
            "id"
        ],

        "key": signal[
            "key"
        ],

        "side": signal[
            "side"
        ],

        "setup_name": signal[
            "setup_name"
        ],

        "grade": signal[
            "grade"
        ],

        "score": signal[
            "score"
        ],

        "plan": plan,

        "confirmation_end": (
            signal[
                "confirmation_end"
            ].isoformat()
        ),

        "created_at": (
            now.isoformat()
        ),

        "expires_at": (
            signal[
                "confirmation_end"
            ]
            + pd.Timedelta(
                minutes=(
                    WATCH_EXPIRY_MINUTES
                )
            )
        ).isoformat(),

        "touched": bool(
            inside
        ),

        "touch_time": (
            now.isoformat()
            if inside
            else None
        ),
    }


# =========================================================
# RETEST CONFIRMATION
# =========================================================

def retest_confirmation_level(
    watch,
):

    plan = watch[
        "plan"
    ]

    buffer_value = (
        plan[
            "retest_confirm_buffer"
        ]
    )

    if watch[
        "side"
    ] == 1:

        return (
            plan[
                "zone_high"
            ]
            + buffer_value
        )

    return (
        plan[
            "zone_low"
        ]
        - buffer_value
    )


# =========================================================
# TRADE CREATION
# =========================================================

def make_trade(
    signal_id,
    side,
    setup,
    entry_type,
    live_price,
    plan,
    now,
):

    return {
        "id": signal_id,
        "side": side,
        "setup": setup,

        "entry_type": (
            entry_type
        ),

        "entry": float(
            live_price
        ),

        "stop": float(
            plan["stop"]
        ),

        "initial_stop": float(
            plan["stop"]
        ),

        "tps": [
            float(x)
            for x in plan[
                "tps"
            ]
        ],

        "hits": [
            False,
            False,
            False,
        ],

        "opened_at": (
            now.isoformat()
        ),

        # Start tracking after entry.
        # Do not replay pre-entry minute data.
        "last_tracking_bar": (
            now.floor(
                "min"
            ).isoformat()
        ),

        "state": "OPEN",
    }


def open_trade(
    st,
    signal,
    live_price,
    now,
):

    st[
        "active_trade"
    ][
        signal["key"]
    ] = make_trade(
        signal["id"],
        signal["side"],
        signal["setup_name"],
        "DIRECT",
        live_price,
        signal["plan"],
        now,
    )


def open_trade_from_watch(
    st,
    watch,
    live_price,
    now,
):

    st[
        "active_trade"
    ][
        watch["key"]
    ] = make_trade(
        watch["id"],
        watch["side"],
        (
            watch[
                "setup_name"
            ]
            + " RETEST"
        ),
        "RETEST",
        live_price,
        watch["plan"],
        now,
    )


# =========================================================
# PROCESS WATCH
# =========================================================

def process_watch(
    st,
    key,
    live_price,
    now,
):

    watch = (
        st[
            "watch_setup"
        ].get(
            key
        )
    )

    if not watch:
        return None, False

    plan = watch[
        "plan"
    ]

    side = int(
        watch[
            "side"
        ]
    )

    if now > stamp(
        watch[
            "expires_at"
        ]
    ):

        st[
            "watch_setup"
        ].pop(
            key,
            None,
        )

        return None, False

    if side == 1:

        stop_breached = (
            live_price
            <= plan[
                "stop"
            ]
        )

    else:

        stop_breached = (
            live_price
            >= plan[
                "stop"
            ]
        )

    if stop_breached:

        st[
            "watch_setup"
        ].pop(
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

    inside = (
        plan[
            "zone_low"
        ]
        <= live_price
        <= plan[
            "zone_high"
        ]
    )

    if (
        inside
        and not watch.get(
            "touched",
            False,
        )
    ):

        watch[
            "touched"
        ] = True

        watch[
            "touch_time"
        ] = now.isoformat()

        st[
            "watch_setup"
        ][key] = watch

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

    if not watch.get(
        "touched",
        False,
    ):
        return None, False

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

    if side == 1:

        zone_edge = (
            plan[
                "zone_high"
            ]
        )

        move_from_zone = (
            live_price
            - zone_edge
        )

    else:

        zone_edge = (
            plan[
                "zone_low"
            ]
        )

        move_from_zone = (
            zone_edge
            - live_price
        )

    max_chase = (
        MAX_RETEST_CHASE_R
        * plan[
            "risk"
        ]
    )

    if (
        move_from_zone
        > max_chase
    ):

        watch[
            "touched"
        ] = False

        watch[
            "touch_time"
        ] = None

        st[
            "watch_setup"
        ][key] = watch

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

    st[
        "watch_setup"
    ].pop(
        key,
        None,
    )

    open_trade_from_watch(
        st,
        watch,
        live_price,
        now,
    )

    current_risk = max(
        abs(
            live_price
            - plan[
                "stop"
            ]
        ),
        1e-9,
    )

    rr_tp1 = (
        abs(
            plan[
                "tps"
            ][0]
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
        f"R:R TO TP1: "
        f"{rr_tp1:.2f}R\n"
        f"\n"
        f"SETUP ID: {watch['id']}"
    )

    return message, True


# =========================================================
# CLOSE TRADE
# =========================================================

def close_trade(
    st,
    key,
    reason,
    exit_price,
    now,
):

    trade = (
        st[
            "active_trade"
        ].get(
            key
        )
    )

    if not trade:
        return

    closed_trade = dict(
        trade
    )

    closed_trade.update(
        {
            "state": "CLOSED",

            "exit_reason": (
                reason
            ),

            "exit_price": float(
                exit_price
            ),

            "closed_at": (
                stamp(
                    now
                ).isoformat()
            ),
        }
    )

    st[
        "closed"
    ].append(
        closed_trade
    )

    st[
        "active_trade"
    ].pop(
        key,
        None,
    )


# =========================================================
# TRACKING DATA
# =========================================================

def get_tracking_bars(
    key,
    cfg,
    trade,
):

    cursor = stamp(
        trade.get(
            "last_tracking_bar"
        )
        or trade[
            "opened_at"
        ]
    )

    source = cfg[
        "source"
    ]

    # =====================================================
    # BINANCE CRYPTO
    # =====================================================

    if source == "binance":

        start_ms = int(
            cursor.timestamp()
            * 1000
        )

        return binance_klines(
            cfg[
                "symbol"
            ],
            "1m",
            1000,
            start_ms,
        )

    # =====================================================
    # TWELVE DATA XAUUSD
    # =====================================================

    if source == "twelve":

        df = twelve_series(
            cfg[
                "symbol"
            ],
            "1min",
            30,
        )

        df[
            "end"
        ] = (
            df.index
            + pd.Timedelta(
                minutes=1
            )
        )

        return df

    # =====================================================
    # YAHOO NASDAQ
    # =====================================================

    df = yf.download(
        cfg[
            "ticker"
        ],
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    df = normalize_yf(
        df
    )

    result = (
        df[
            [
                "o",
                "h",
                "l",
                "c",
            ]
        ]
        .copy()
    )

    result[
        "end"
    ] = (
        result.index
        + pd.Timedelta(
            minutes=1
        )
    )

    return result


# =========================================================
# TRADE HEADER
# =========================================================

def trade_header(
    key,
    trade,
):

    direction = (
        "LONG"
        if int(
            trade[
                "side"
            ]
        ) == 1
        else "SHORT"
    )

    return (
        f"{key} {direction}\n"
        f"SETUP: "
        f"{trade.get('setup', '-')}\n"
        f"ENTRY TYPE: "
        f"{trade.get('entry_type', 'LEGACY')}\n"
        f"SETUP ID: "
        f"{trade['id']}"
    )


# =========================================================
# STOP TYPE
# =========================================================

def stop_reason(
    trade,
):

    hits = trade.get(
        "hits",
        [
            False,
            False,
            False,
        ],
    )

    if hits[1]:
        return "TP1_LOCKED"

    if hits[0]:
        return "BREAKEVEN"

    return "STOP"


def stop_message(
    key,
    trade,
    stop,
    live_price=None,
):

    reason = stop_reason(
        trade
    )

    if reason == "BREAKEVEN":

        icon = "🟰"

        title = (
            "BREAKEVEN HIT"
        )

    elif reason == "TP1_LOCKED":

        icon = "🔒"

        title = (
            "PROTECTED STOP HIT"
        )

    else:

        icon = "🛑"

        title = (
            "STOP LOSS HIT"
        )

    live_text = ""

    if live_price is not None:

        live_text = (
            f"\nLIVE: "
            f"{fmt(live_price)}"
        )

    return (
        f"{icon} {key} | {title}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{trade_header(key, trade)}\n"
        f"\n"
        f"ENTRY: "
        f"{fmt(trade['entry'])}\n"
        f"EXIT: "
        f"{fmt(stop)}"
        f"{live_text}\n"
        f"STATUS: CLOSED"
    )


# =========================================================
# PROCESS CLOSED 1M BAR
# =========================================================

def process_trade_bar(
    st,
    key,
    trade,
    bar,
    bar_time,
):

    side = int(
        trade[
            "side"
        ]
    )

    high = float(
        bar[
            "h"
        ]
    )

    low = float(
        bar[
            "l"
        ]
    )

    stop = float(
        trade[
            "stop"
        ]
    )

    targets = list(
        map(
            float,
            trade[
                "tps"
            ],
        )
    )

    hits = trade.setdefault(
        "hits",
        [
            False,
            False,
            False,
        ],
    )

    messages = []

    # =====================================================
    # STOP FIRST
    #
    # If stop + target both appear inside the same 1m bar,
    # exact tick order is unknown.
    # Conservative rule = STOP first.
    # =====================================================

    if side == 1:

        stop_hit = (
            low <= stop
        )

    else:

        stop_hit = (
            high >= stop
        )

    if stop_hit:

        reason = stop_reason(
            trade
        )

        close_trade(
            st,
            key,
            reason,
            stop,
            bar_time,
        )

        messages.append(
            stop_message(
                key,
                trade,
                stop,
            )
        )

        return (
            messages,
            True,
        )

    # =====================================================
    # TARGETS
    # =====================================================

    for i, target in enumerate(
        targets
    ):

        if hits[i]:
            continue

        if side == 1:

            target_hit = (
                high >= target
            )

        else:

            target_hit = (
                low <= target
            )

        if not target_hit:
            break

        hits[i] = True

        # =================================================
        # TP1
        # =================================================

        if i == 0:

            old_stop = float(
                trade[
                    "stop"
                ]
            )

            trade[
                "stop"
            ] = float(
                trade[
                    "entry"
                ]
            )

            messages.append(
                (
                    f"🎯 {key} | TP1 HIT\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP1: "
                    f"{fmt(target)}\n"
                    f"🔒 SL MOVED: "
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: BREAKEVEN"
                )
            )

        # =================================================
        # TP2
        # =================================================

        elif i == 1:

            old_stop = float(
                trade[
                    "stop"
                ]
            )

            trade[
                "stop"
            ] = float(
                targets[0]
            )

            messages.append(
                (
                    f"🎯 {key} | TP2 HIT\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP2: "
                    f"{fmt(target)}\n"
                    f"🔒 SL MOVED: "
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: TP1 LOCKED"
                )
            )

        # =================================================
        # TP3
        # =================================================

        else:

            close_trade(
                st,
                key,
                "TP3",
                target,
                bar_time,
            )

            messages.append(
                (
                    f"🏆 {key} | TP3 HIT — CLOSED\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP3: "
                    f"{fmt(target)}\n"
                    f"ENTRY: "
                    f"{fmt(trade['entry'])}\n"
                    f"STATUS: CLOSED"
                )
            )

            return (
                messages,
                True,
            )

    trade[
        "hits"
    ] = hits

    return (
        messages,
        False,
    )


# =========================================================
# MONITOR OPEN TRADE
# =========================================================

def monitor_open_trade(
    st,
    key,
    cfg,
    live_price,
    now,
):

    trade = (
        st[
            "active_trade"
        ].get(
            key
        )
    )

    if not trade:
        return []

    trade.setdefault(
        "hits",
        [
            False,
            False,
            False,
        ],
    )

    trade.setdefault(
        "initial_stop",
        trade[
            "stop"
        ],
    )

    trade.setdefault(
        "entry_type",
        "LEGACY",
    )

    # =====================================================
    # MIGRATION PROTECTION
    #
    # Old trade without last_tracking_bar:
    # start tracking NOW.
    #
    # Do NOT replay old historical candles because that can
    # create a fake retrospective TP1.
    # =====================================================

    if not trade.get(
        "last_tracking_bar"
    ):

        trade[
            "last_tracking_bar"
        ] = (
            now.floor(
                "min"
            ).isoformat()
        )

        st[
            "active_trade"
        ][key] = trade

    messages = []

    # =====================================================
    # CLOSED 1M CANDLES
    # =====================================================

    try:

        bars = get_tracking_bars(
            key,
            cfg,
            trade,
        )

    except Exception as error:

        print(
            key,
            "tracking error:",
            error,
        )

        bars = pd.DataFrame()

    if not bars.empty:

        last_seen = stamp(
            trade[
                "last_tracking_bar"
            ]
        )

        for ts, bar in (
            bars.iterrows()
        ):

            ts = stamp(
                ts
            )

            bar_end = stamp(
                bar.get(
                    "end",
                    (
                        ts
                        + pd.Timedelta(
                            minutes=1
                        )
                    ),
                )
            )

            # =============================================
            # NEVER process unfinished 1m candle.
            # =============================================

            if bar_end > now:
                continue

            # =============================================
            # NEVER process same bar twice.
            # =============================================

            if ts <= last_seen:
                continue

            bar_messages, closed = (
                process_trade_bar(
                    st,
                    key,
                    trade,
                    bar,
                    bar_end,
                )
            )

            messages.extend(
                bar_messages
            )

            if closed:
                return messages

            trade[
                "last_tracking_bar"
            ] = ts.isoformat()

            st[
                "active_trade"
            ][key] = trade

    if key not in st[
        "active_trade"
    ]:

        return messages

    # =====================================================
    # LIVE PRICE CHECK
    #
    # Current live price can trigger immediately.
    # We DO NOT use unfinished 1m high/low.
    # =====================================================

    trade = st[
        "active_trade"
    ][key]

    side = int(
        trade[
            "side"
        ]
    )

    stop = float(
        trade[
            "stop"
        ]
    )

    if side == 1:

        live_stop_hit = (
            live_price <= stop
        )

    else:

        live_stop_hit = (
            live_price >= stop
        )

    if live_stop_hit:

        reason = stop_reason(
            trade
        )

        close_trade(
            st,
            key,
            reason,
            stop,
            now,
        )

        messages.append(
            stop_message(
                key,
                trade,
                stop,
                live_price,
            )
        )

        return messages

    # =====================================================
    # LIVE TARGET CHECK
    # =====================================================

    targets = list(
        map(
            float,
            trade[
                "tps"
            ],
        )
    )

    hits = trade[
        "hits"
    ]

    for i, target in enumerate(
        targets
    ):

        if hits[i]:
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

        if not target_hit:
            break

        hits[i] = True

        # =================================================
        # TP1
        # =================================================

        if i == 0:

            old_stop = float(
                trade[
                    "stop"
                ]
            )

            trade[
                "stop"
            ] = float(
                trade[
                    "entry"
                ]
            )

            messages.append(
                (
                    f"🎯 {key} | TP1 HIT\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP1: "
                    f"{fmt(target)}\n"
                    f"LIVE: "
                    f"{fmt(live_price)}\n"
                    f"🔒 SL MOVED: "
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: BREAKEVEN"
                )
            )

        # =================================================
        # TP2
        # =================================================

        elif i == 1:

            old_stop = float(
                trade[
                    "stop"
                ]
            )

            trade[
                "stop"
            ] = float(
                targets[0]
            )

            messages.append(
                (
                    f"🎯 {key} | TP2 HIT\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP2: "
                    f"{fmt(target)}\n"
                    f"LIVE: "
                    f"{fmt(live_price)}\n"
                    f"🔒 SL MOVED: "
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: TP1 LOCKED"
                )
            )

        # =================================================
        # TP3
        # =================================================

        else:

            close_trade(
                st,
                key,
                "TP3",
                target,
                now,
            )

            messages.append(
                (
                    f"🏆 {key} | TP3 HIT — CLOSED\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP3: "
                    f"{fmt(target)}\n"
                    f"LIVE: "
                    f"{fmt(live_price)}\n"
                    f"STATUS: CLOSED"
                )
            )

            return messages

    trade[
        "hits"
    ] = hits

    if key in st[
        "active_trade"
    ]:

        st[
            "active_trade"
        ][key] = trade

    return messages


# =========================================================
# FORMAT SIGNAL
# =========================================================

def format_signal(
    signal,
    live_price,
    entry_status,
):

    side = (
        "LONG"
        if signal[
            "side"
        ] == 1
        else "SHORT"
    )

    plan = signal[
        "plan"
    ]

    htf = signal[
        "htf"
    ]

    volume = signal[
        "volume"
    ]

    confirmation_time = (
        signal[
            "confirmation_end"
        ].tz_convert(
            BAGHDAD
        )
    )

    age_seconds = int(
        entry_status[
            "age_seconds"
        ]
    )

    volume_text = (
        volume[
            "label"
        ]
    )

    if volume[
        "ratio"
    ] is not None:

        volume_text += (
            f" "
            f"({volume['ratio']:.2f}x)"
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
        f"(1.5R)\n"
        f"TP2: "
        f"{fmt(plan['tps'][1])} "
        f"(2.0R)\n"
        f"TP3: "
        f"{fmt(plan['tps'][2])} "
        f"(3.0R)\n"
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
        f"{'YES' if htf['near_zone'] else 'NO'}\n"
        f"{htf['zone_text']}\n"
        f"CISD: "
        f"{'YES' if signal['cisd'] else 'NO'}\n"
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
    # LIVE PRICE FIRST
    #
    # Same source is used for each market:
    #
    # XAUUSD = Twelve Data
    # Crypto = Binance
    # NASDAQ = Yahoo
    # =====================================================

    try:

        live_price = (
            get_live_price(
                key,
                cfg,
            )
        )

    except Exception as error:

        print(
            key,
            "live data error:",
            error,
        )

        return

    # =====================================================
    # ACTIVE TRADE
    # =====================================================

    trade_messages = (
        monitor_open_trade(
            st,
            key,
            cfg,
            live_price,
            now,
        )
    )

    for message in (
        trade_messages
    ):

        send_telegram(
            message
        )

    if key in st[
        "active_trade"
    ]:

        return

    # =====================================================
    # WAITING RETEST
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
        or key in st[
            "active_trade"
        ]
    ):

        return

    # =====================================================
    # ANALYSIS DATA
    # =====================================================

    try:

        frames = get_frames(
            key,
            cfg,
        )

    except Exception as error:

        print(
            key,
            "frame data error:",
            error,
        )

        return

    # =====================================================
    # NEW SIGNAL
    # =====================================================

    signal = detect_signal(
        key,
        frames,
    )

    if not signal:
        return

    if (
        st[
            "last_signal"
        ].get(
            key
        )
        == signal[
            "id"
        ]
    ):

        return

    st[
        "last_signal"
    ][key] = signal[
        "id"
    ]

    status = (
        initial_entry_status(
            signal,
            live_price,
            now,
        )
    )

    # =====================================================
    # INITIAL ALERT
    # =====================================================

    send_telegram(
        format_signal(
            signal,
            live_price,
            status,
        )
    )

    # =====================================================
    # DIRECT ENTRY
    # =====================================================

    if status[
        "tradable"
    ]:

        st[
            "watch_setup"
        ].pop(
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
    # WAIT FOR RETEST
    # =====================================================

    if status[
        "watch"
    ]:

        st[
            "watch_setup"
        ][key] = (
            signal_to_watch(
                signal,
                live_price,
                now,
            )
        )


# =========================================================
# RUN ONCE
# =========================================================

def run_once():

    st = load_state()

    for key, cfg in (
        MARKETS.items()
    ):

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

    save_state(
        st
    )


# =========================================================
# MAIN
# =========================================================

def main():

    scans = max(
        1,
        LOOP_SCANS,
    )

    for i in range(
        scans
    ):

        started = (
            time.time()
        )

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
                (
                    SCAN_INTERVAL_SECONDS
                    - spent
                ),
            )

            time.sleep(
                sleep_seconds
            )


if __name__ == "__main__":
    main()
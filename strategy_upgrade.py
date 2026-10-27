import math
import pandas as pd

import resilient_runner as runner

agent = runner.agent
_ORIG_DETECT_SIGNAL = agent.detect_signal
_ORIG_FORMAT_SIGNAL = agent.format_signal

# Structure-aware risk settings. ATR is now a buffer around structure,
# not the structure itself.
STRUCTURE_SL_BUFFER_ATR = 0.20
MAX_INITIAL_RISK_ATR = 2.20
MIN_TP1_RR = 1.50


def _pivot_levels(df, side, lookback=80):
    """Return recent confirmed swing levels, newest first."""
    x = df.tail(lookback)
    if len(x) < 5:
        return []
    values = []
    for i in range(2, len(x) - 2):
        row = x.iloc[i]
        if side == 1:
            value = float(row["h"])
            if value >= float(x["h"].iloc[i-2:i].max()) and value >= float(x["h"].iloc[i+1:i+3].max()):
                values.append(value)
        else:
            value = float(row["l"])
            if value <= float(x["l"].iloc[i-2:i].min()) and value <= float(x["l"].iloc[i+1:i+3].min()):
                values.append(value)
    return list(reversed(values))


def _reference_level(signal, d15):
    previous = d15.iloc[-21:-1]
    if signal["side"] == 1:
        return float(previous["l"].min()) if signal["setup_type"] == "SWEEP" else float(previous["h"].max())
    return float(previous["h"].max()) if signal["setup_type"] == "SWEEP" else float(previous["l"].min())


def _structure_stop(signal, frames, reference, atr15):
    side = int(signal["side"])
    d15 = frames["15m"]
    last = d15.iloc[-1]

    # Use the signal invalidation plus the most recent confirmed 15m swing.
    # This avoids placing SL inside ordinary retest noise.
    if side == 1:
        candidates = [float(last["l"]), float(reference)]
        swing_lows = _pivot_levels(d15, -1, 40)
        candidates += [x for x in swing_lows[:3] if x < float(last["c"])]
        structure = min(candidates)
        stop = structure - STRUCTURE_SL_BUFFER_ATR * atr15
        risk = float(last["c"]) - stop
    else:
        candidates = [float(last["h"]), float(reference)]
        swing_highs = _pivot_levels(d15, 1, 40)
        candidates += [x for x in swing_highs[:3] if x > float(last["c"])]
        structure = max(candidates)
        stop = structure + STRUCTURE_SL_BUFFER_ATR * atr15
        risk = stop - float(last["c"])

    if not math.isfinite(risk) or risk <= 0 or risk > MAX_INITIAL_RISK_ATR * atr15:
        return None
    return float(stop), float(risk)


def _target_candidates(frames, side, entry):
    levels = []
    for tf, lookback in (("15m", 100), ("1h", 100), ("4h", 60)):
        df = frames[tf]
        levels.extend(_pivot_levels(df, side, lookback))
        tail = df.tail(lookback)
        if side == 1:
            levels.append(float(tail["h"].max()))
        else:
            levels.append(float(tail["l"].min()))

    if side == 1:
        levels = sorted({float(x) for x in levels if float(x) > entry})
    else:
        levels = sorted({float(x) for x in levels if float(x) < entry}, reverse=True)
    return levels


def _choose_targets(frames, side, entry, risk):
    levels = _target_candidates(frames, side, entry)
    if not levels:
        return None

    targets = []
    used = set()
    for minimum_rr in (MIN_TP1_RR, 2.0, 3.0):
        chosen = None
        for level in levels:
            rr = ((level - entry) * side) / risk
            if rr >= minimum_rr and level not in used:
                chosen = level
                break
        if chosen is None:
            # TP1 must have real structural room. Later targets may use
            # the mathematical extension only after TP1 passed the filter.
            if not targets:
                return None
            chosen = entry + side * risk * minimum_rr
        targets.append(float(chosen))
        used.add(chosen)
    return targets


def upgraded_detect_signal(key, frames):
    signal = _ORIG_DETECT_SIGNAL(key, frames)
    if not signal:
        return None

    d15 = frames["15m"]
    atr15 = float(agent.atr(d15).iloc[-1])
    entry = float(d15.iloc[-1]["c"])
    reference = _reference_level(signal, d15)

    stop_info = _structure_stop(signal, frames, reference, atr15)
    if not stop_info:
        print(key, "setup rejected: structure stop too wide/invalid")
        return None
    stop, risk = stop_info

    targets = _choose_targets(frames, int(signal["side"]), entry, risk)
    if not targets:
        print(key, "setup rejected: no structural TP1 with >= 1.5R room")
        return None

    zone_low, zone_high = agent.calculate_entry_zone(int(signal["side"]), entry, atr15)
    signal["plan"] = {
        "entry": entry,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "stop": stop,
        "tps": targets,
        "risk": risk,
        "atr15": atr15,
        "retest_confirm_buffer": float(agent.RETEST_CONFIRM_ATR * atr15),
        "structure_based": True,
    }
    return signal


def upgraded_format_signal(signal, live_price, entry_status):
    text = _ORIG_FORMAT_SIGNAL(signal, live_price, entry_status)
    if signal.get("plan", {}).get("structure_based"):
        text = text.replace("TP1: " + agent.fmt(signal["plan"]["tps"][0]) + " (1.5R)", "TP1: " + agent.fmt(signal["plan"]["tps"][0]) + " (STRUCTURE, >=1.5R)")
        text = text.replace("TP2: " + agent.fmt(signal["plan"]["tps"][1]) + " (2.0R)", "TP2: " + agent.fmt(signal["plan"]["tps"][1]) + " (STRUCTURE/2R+)")
        text = text.replace("TP3: " + agent.fmt(signal["plan"]["tps"][2]) + " (3.0R)", "TP3: " + agent.fmt(signal["plan"]["tps"][2]) + " (STRUCTURE/3R+)")
    return text


agent.detect_signal = upgraded_detect_signal
agent.format_signal = upgraded_format_signal

if __name__ == "__main__":
    agent.main()

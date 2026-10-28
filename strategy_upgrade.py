import math
import pandas as pd

import resilient_runner as runner

agent = runner.agent
_ORIG_DETECT_SIGNAL = agent.detect_signal
_ORIG_FORMAT_SIGNAL = agent.format_signal
_ORIG_PROCESS_WATCH = agent.process_watch

# Structure-aware risk settings. ATR is a volatility buffer around real
# structure, not the structure itself. Avoid tiny stops that sit inside
# normal 15m market noise.
STRUCTURE_SL_BUFFER_ATR = 0.30
MIN_INITIAL_RISK_ATR = 0.75
MAX_INITIAL_RISK_ATR = 2.50
MIN_TP1_RR = 1.50

# Retest confirmation must not be rejected just because a fast market moved
# a small fixed fraction of R. Allow the larger of 0.60R or 0.50 ATR, capped
# at 1R so the bot still refuses genuinely late/chased entries.
RETEST_MIN_CHASE_R = 0.60
RETEST_MIN_CHASE_ATR = 0.50
RETEST_MAX_CHASE_R = 1.00


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


def _structure_stop_candidates(signal, frames, reference, atr15):
    """Return structure stops ordered from nearest acceptable risk to widest."""
    side = int(signal["side"])
    d15 = frames["15m"]
    last = d15.iloc[-1]
    entry = float(last["c"])

    raw_structures = [float(reference)]
    if side == 1:
        raw_structures.append(float(last["l"]))
        raw_structures += [x for x in _pivot_levels(d15, -1, 50)[:6] if x < entry]
    else:
        raw_structures.append(float(last["h"]))
        raw_structures += [x for x in _pivot_levels(d15, 1, 50)[:6] if x > entry]

    candidates = []
    seen = set()
    for structure in raw_structures:
        if not math.isfinite(structure):
            continue
        stop = structure - side * STRUCTURE_SL_BUFFER_ATR * atr15
        risk = (entry - stop) * side
        if not math.isfinite(risk) or risk <= 0:
            continue
        # Reject stops that are still inside ordinary 15m noise. This is the
        # main protection against the frequent small stop-outs we observed.
        if risk < MIN_INITIAL_RISK_ATR * atr15:
            continue
        key = round(float(stop), 10)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((float(stop), float(risk), float(structure)))

    candidates.sort(key=lambda x: x[1])
    return candidates


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
            if not targets:
                return None
            chosen = entry + side * risk * minimum_rr
        targets.append(float(chosen))
        used.add(chosen)
    return targets


def _choose_structure_plan(signal, frames, reference, atr15):
    """Use the nearest stop outside normal noise, then widen only if needed."""
    side = int(signal["side"])
    entry = float(frames["15m"].iloc[-1]["c"])

    for stop, risk, structure in _structure_stop_candidates(signal, frames, reference, atr15):
        if risk > MAX_INITIAL_RISK_ATR * atr15:
            continue
        targets = _choose_targets(frames, side, entry, risk)
        if targets:
            return stop, risk, targets, structure
    return None


def upgraded_detect_signal(key, frames):
    signal = _ORIG_DETECT_SIGNAL(key, frames)
    if not signal:
        return None

    d15 = frames["15m"]
    atr15 = float(agent.atr(d15).iloc[-1])
    entry = float(d15.iloc[-1]["c"])
    reference = _reference_level(signal, d15)

    plan_info = _choose_structure_plan(signal, frames, reference, atr15)
    if not plan_info:
        print(key, "setup rejected: no structure stop outside noise with >= 1.5R TP1 room")
        return None
    stop, risk, targets, structure = plan_info

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
        "stop_structure": structure,
    }
    return signal


def upgraded_format_signal(signal, live_price, entry_status):
    text = _ORIG_FORMAT_SIGNAL(signal, live_price, entry_status)
    if signal.get("plan", {}).get("structure_based"):
        text = text.replace("TP1: " + agent.fmt(signal["plan"]["tps"][0]) + " (1.5R)", "TP1: " + agent.fmt(signal["plan"]["tps"][0]) + " (STRUCTURE, >=1.5R)")
        text = text.replace("TP2: " + agent.fmt(signal["plan"]["tps"][1]) + " (2.0R)", "TP2: " + agent.fmt(signal["plan"]["tps"][1]) + " (STRUCTURE/2R+)")
        text = text.replace("TP3: " + agent.fmt(signal["plan"]["tps"][2]) + " (3.0R)", "TP3: " + agent.fmt(signal["plan"]["tps"][2]) + " (STRUCTURE/3R+)")
    return text


def upgraded_process_watch(st, key, live_price, now):
    """Apply an ATR + risk adaptive chase allowance to retest confirmation."""
    watch = st.get("watch_setup", {}).get(key)
    if not watch:
        return _ORIG_PROCESS_WATCH(st, key, live_price, now)

    plan = watch.get("plan") or {}
    risk = float(plan.get("risk") or 0.0)
    atr15 = float(plan.get("atr15") or 0.0)
    if risk <= 0 or atr15 <= 0:
        return _ORIG_PROCESS_WATCH(st, key, live_price, now)

    # agent.process_watch computes max_chase as MAX_RETEST_CHASE_R * risk.
    # Convert our desired absolute allowance back to an R multiplier so the
    # existing, well-tested watch state machine can remain unchanged.
    desired_abs = max(RETEST_MIN_CHASE_R * risk, RETEST_MIN_CHASE_ATR * atr15)
    desired_abs = min(desired_abs, RETEST_MAX_CHASE_R * risk)
    adaptive_r = desired_abs / risk

    old_value = agent.MAX_RETEST_CHASE_R
    agent.MAX_RETEST_CHASE_R = adaptive_r
    try:
        return _ORIG_PROCESS_WATCH(st, key, live_price, now)
    finally:
        agent.MAX_RETEST_CHASE_R = old_value


agent.detect_signal = upgraded_detect_signal
agent.format_signal = upgraded_format_signal
agent.process_watch = upgraded_process_watch

if __name__ == "__main__":
    agent.main()

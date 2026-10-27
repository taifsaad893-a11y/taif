import yfinance as yf

import strategy_upgrade

agent = strategy_upgrade.agent

_ORIG_GET_FRAMES = agent.get_frames
_ORIG_GET_LIVE_PRICE = agent.get_live_price
_ORIG_PROCESS_WATCH = agent.process_watch

GOLD_YAHOO_SYMBOL = "XAUUSD=X"


def yahoo_gold_live():
    df = agent.normalize_yf(yf.download(
        GOLD_YAHOO_SYMBOL,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
        threads=False,
    ))
    if df.empty:
        raise RuntimeError("Yahoo gold live price unavailable")
    return float(df["c"].iloc[-1])


def resilient_gold_frames(key, cfg):
    if key != "XAUUSD" or cfg.get("source") != "twelve":
        return _ORIG_GET_FRAMES(key, cfg)

    try:
        return _ORIG_GET_FRAMES(key, cfg)
    except Exception as exc:
        print(
            f"XAUUSD: Twelve Data frames unavailable -> Yahoo {GOLD_YAHOO_SYMBOL} ({exc})",
            flush=True,
        )
        return agent.yahoo_frames(GOLD_YAHOO_SYMBOL)


def resilient_gold_live(key, cfg):
    if key != "XAUUSD" or cfg.get("source") != "twelve":
        return _ORIG_GET_LIVE_PRICE(key, cfg)

    try:
        return _ORIG_GET_LIVE_PRICE(key, cfg)
    except Exception as exc:
        print(
            f"XAUUSD: Twelve Data live unavailable -> Yahoo {GOLD_YAHOO_SYMBOL} ({exc})",
            flush=True,
        )
        return yahoo_gold_live()


def deduplicated_process_watch(st, key, live_price, now):
    watch_before = st.get("watch_setup", {}).get(key)
    already_alerted = bool(watch_before and watch_before.get("moved_too_far_alerted"))

    message, entered = _ORIG_PROCESS_WATCH(st, key, live_price, now)

    if message and "RETEST MOVED TOO FAR" in message:
        watch_after = st.get("watch_setup", {}).get(key)
        if watch_after is not None:
            watch_after["moved_too_far_alerted"] = True
            st["watch_setup"][key] = watch_after

        if already_alerted:
            print(f"{key}: duplicate RETEST MOVED TOO FAR alert suppressed", flush=True)
            return None, entered

    return message, entered


agent.get_frames = resilient_gold_frames
agent.get_live_price = resilient_gold_live
agent.process_watch = deduplicated_process_watch


if __name__ == "__main__":
    agent.main()

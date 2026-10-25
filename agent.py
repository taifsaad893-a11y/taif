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
        return None

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

    return closed_trade


# =========================================================
# 1M TRACKING DATA
# =========================================================

def get_tracking_bars(
    key,
    ticker,
    trade,
):
    """
    Get 1-minute candles for trade monitoring.

    IMPORTANT:
    We use candle HIGH / LOW so a stop or target cannot
    disappear simply because price touched it between scans.

    BTC uses Binance first.
    Gold / Nasdaq use Yahoo 1m.
    """

    opened_at = stamp(
        trade["opened_at"]
    )

    # =====================================================
    # BTC - BINANCE 1M
    # =====================================================

    if key == "BTCUSD":

        try:

            start_ms = int(
                opened_at.timestamp()
                * 1000
            )

            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": start_ms,
                    "limit": 1000,
                },
                timeout=15,
            )

            response.raise_for_status()

            raw = response.json()

            if raw:

                rows = []

                for candle in raw:

                    candle_start = (
                        pd.to_datetime(
                            int(candle[0]),
                            unit="ms",
                            utc=True,
                        )
                    )

                    candle_end = (
                        pd.to_datetime(
                            int(candle[6]),
                            unit="ms",
                            utc=True,
                        )
                    )

                    rows.append(
                        {
                            "time": candle_start,
                            "end": candle_end,
                            "o": float(candle[1]),
                            "h": float(candle[2]),
                            "l": float(candle[3]),
                            "c": float(candle[4]),
                        }
                    )

                df = pd.DataFrame(
                    rows
                )

                if not df.empty:

                    df = (
                        df.set_index(
                            "time"
                        )
                        .sort_index()
                    )

                    return df

        except Exception as error:

            print(
                key,
                "Binance tracking error:",
                error,
            )

    # =====================================================
    # YAHOO 1M FALLBACK
    # =====================================================

    try:

        df = yf.download(
            ticker,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        df = normalize_yf(
            df
        )

        if df.empty:
            return pd.DataFrame()

        result = df[
            [
                "o",
                "h",
                "l",
                "c",
            ]
        ].copy()

        result["end"] = (
            result.index
            + pd.Timedelta(
                minutes=1
            )
        )

        # Only candles after trade opened.
        result = result[
            result["end"]
            >= opened_at
        ]

        return result

    except Exception as error:

        print(
            key,
            "1m tracking error:",
            error,
        )

        return pd.DataFrame()


# =========================================================
# TRADE RESULT MESSAGE
# =========================================================

def trade_header(
    key,
    trade,
):

    direction = (
        "LONG"
        if int(trade["side"]) == 1
        else "SHORT"
    )

    return (
        f"{key} {direction}\n"
        f"SETUP: "
        f"{trade.get('setup', '-')}\n"
        f"SETUP ID: "
        f"{trade['id']}"
    )


# =========================================================
# PROCESS ONE TRACKING BAR
# =========================================================

def process_trade_bar(
    st,
    key,
    trade,
    bar,
    bar_time,
):

    side = int(
        trade["side"]
    )

    high = float(
        bar["h"]
    )

    low = float(
        bar["l"]
    )

    close = float(
        bar["c"]
    )

    stop = float(
        trade["stop"]
    )

    targets = list(
        map(
            float,
            trade["tps"],
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
    # STOP CHECK
    # =====================================================

    if side == 1:

        stop_hit = (
            low <= stop
        )

    else:

        stop_hit = (
            high >= stop
        )

    # Conservative OHLC rule:
    # if stop and a new target are both inside same 1m
    # candle, STOP wins because intrabar order is unknown.

    new_target_inside = False

    for i, target in enumerate(
        targets
    ):

        if hits[i]:
            continue

        if side == 1:

            hit = (
                high >= target
            )

        else:

            hit = (
                low <= target
            )

        if hit:

            new_target_inside = True
            break

    if stop_hit:

        close_trade(
            st,
            key,
            "STOP",
            stop,
            bar_time,
        )

        message = (
            f"🛑 {key} | STOP LOSS HIT\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{trade_header(key, trade)}\n"
            f"\n"
            f"ENTRY: "
            f"{fmt(trade['entry'])}\n"
            f"STOP: "
            f"{fmt(stop)}\n"
            f"1M HIGH: "
            f"{fmt(high)}\n"
            f"1M LOW: "
            f"{fmt(low)}\n"
        )

        if new_target_inside:

            message += (
                "\n⚠️ نفس شمعة 1m لامست "
                "الستوب وهدف جديد؛ "
                "تم احتساب STOP بشكل محافظ.\n"
            )

        message += (
            f"\nTIME: "
            f"{stamp(bar_time).tz_convert(BAGHDAD):"
            f"%Y-%m-%d %H:%M} Baghdad"
        )

        messages.append(
            message
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

            # Targets are sequential.
            break

        hits[i] = True

        # =================================================
        # TP1
        # =================================================

        if i == 0:

            old_stop = float(
                trade["stop"]
            )

            # Move SL to breakeven.
            trade["stop"] = float(
                trade["entry"]
            )

            messages.append(
                (
                    f"🎯 {key} | TP1 HIT\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP1: {fmt(target)}\n"
                    f"1M HIGH: {fmt(high)}\n"
                    f"1M LOW: {fmt(low)}\n"
                    f"\n"
                    f"🔒 SL MOVED\n"
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: BREAKEVEN"
                )
            )

            # IMPORTANT:
            # We do not use the newly moved stop against
            # the same candle because we don't know whether
            # TP1 or the later pullback happened first.
            #
            # The new BE stop becomes active from the
            # next 1-minute observation.

        # =================================================
        # TP2
        # =================================================

        elif i == 1:

            old_stop = float(
                trade["stop"]
            )

            # Lock TP1 after TP2.
            trade["stop"] = float(
                targets[0]
            )

            messages.append(
                (
                    f"🎯 {key} | TP2 HIT\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"{trade_header(key, trade)}\n"
                    f"\n"
                    f"TP2: {fmt(target)}\n"
                    f"1M HIGH: {fmt(high)}\n"
                    f"1M LOW: {fmt(low)}\n"
                    f"\n"
                    f"🔒 SL MOVED\n"
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: TP1 LOCKED"
                )
            )

        # =================================================
        # TP3
        # =================================================

        elif i == 2:

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
                    f"TP3: {fmt(target)}\n"
                    f"ENTRY: "
                    f"{fmt(trade['entry'])}\n"
                    f"\n"
                    f"✅ PAPER TRADE CLOSED"
                )
            )

            return (
                messages,
                True,
            )

    trade["hits"] = hits

    trade["last_price"] = (
        close
    )

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
    """
    Monitor BOTH:
      - direct breakout entries
      - retest entries

    using 1m HIGH/LOW + current live price.
    """

    trade = (
        st["active_trade"]
        .get(key)
    )

    if not trade:
        return []

    messages = []

    # Backward compatibility for trades
    # opened by the previous version.
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
        trade["stop"],
    )

    trade.setdefault(
        "last_tracking_bar",
        None,
    )

    # =====================================================
    # GET 1M HISTORY
    # =====================================================

    bars = get_tracking_bars(
        key,
        cfg["ticker"],
        trade,
    )

    # =====================================================
    # PROCESS UNSEEN 1M BARS
    # =====================================================

    if not bars.empty:

        opened_at = stamp(
            trade["opened_at"]
        )

        last_seen = (
            stamp(
                trade[
                    "last_tracking_bar"
                ]
            )
            if trade.get(
                "last_tracking_bar"
            )
            else None
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
                    ts
                    + pd.Timedelta(
                        minutes=1
                    ),
                )
            )

            # Ignore data from before entry.
            if (
                bar_end
                < opened_at
            ):
                continue

            # Closed candles already processed.
            if (
                last_seen is not None
                and ts <= last_seen
            ):
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

            # Trade closed by SL or TP3.
            if closed:

                return messages

            trade[
                "last_tracking_bar"
            ] = ts.isoformat()

            # Make sure state object contains updates.
            st["active_trade"][
                key
            ] = trade

    # Trade may have been closed above.
    if (
        key
        not in st["active_trade"]
    ):
        return messages

    # =====================================================
    # LIVE PRICE BACKUP CHECK
    # =====================================================

    trade = (
        st["active_trade"][
            key
        ]
    )

    side = int(
        trade["side"]
    )

    stop = float(
        trade["stop"]
    )

    if side == 1:

        live_stop_hit = (
            live_price
            <= stop
        )

    else:

        live_stop_hit = (
            live_price
            >= stop
        )

    if live_stop_hit:

        close_trade(
            st,
            key,
            "STOP",
            stop,
            now,
        )

        messages.append(
            (
                f"🛑 {key} | STOP LOSS HIT\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"{trade_header(key, trade)}\n"
                f"\n"
                f"ENTRY: "
                f"{fmt(trade['entry'])}\n"
                f"STOP: "
                f"{fmt(stop)}\n"
                f"LIVE: "
                f"{fmt(live_price)}\n"
                f"\n"
                f"STATUS: CLOSED"
            )
        )

        return messages

    # =====================================================
    # LIVE TARGET BACKUP
    # =====================================================

    targets = list(
        map(
            float,
            trade["tps"],
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

            live_target_hit = (
                live_price
                >= target
            )

        else:

            live_target_hit = (
                live_price
                <= target
            )

        if not live_target_hit:
            break

        hits[i] = True

        # TP1
        if i == 0:

            old_stop = float(
                trade["stop"]
            )

            trade["stop"] = float(
                trade["entry"]
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
                    f"\n"
                    f"🔒 SL MOVED\n"
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: BREAKEVEN"
                )
            )

        # TP2
        elif i == 1:

            old_stop = float(
                trade["stop"]
            )

            trade["stop"] = float(
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
                    f"\n"
                    f"🔒 SL MOVED\n"
                    f"{fmt(old_stop)}"
                    f" → "
                    f"{fmt(trade['stop'])}\n"
                    f"STATUS: TP1 LOCKED"
                )
            )

        # TP3
        elif i == 2:

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
                    f"\n"
                    f"✅ PAPER TRADE CLOSED"
                )
            )

            return messages

    trade["hits"] = hits

    if (
        key
        in st["active_trade"]
    ):

        st["active_trade"][
            key
        ] = trade

    return messages


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
    # Breakout and Retest BOTH arrive here.
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

    for trade_message in (
        trade_messages
    ):

        send_telegram(
            trade_message
        )

    # If still open after monitoring,
    # do NOT search for another setup.
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
    # DIRECT BREAKOUT / DIRECT SETUP ENTRY
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

        # Add tracking fields immediately.
        trade = st[
            "active_trade"
        ][key]

        trade[
            "initial_stop"
        ] = trade["stop"]

        trade[
            "last_tracking_bar"
        ] = None

        trade[
            "entry_type"
        ] = "DIRECT"

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
                SCAN_INTERVAL_SECONDS
                - spent,
            )

            time.sleep(
                sleep_seconds
            )


if __name__ == "__main__":
    main()
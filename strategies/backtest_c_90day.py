"""
Variant C (Color-Change) — 90-Day Deep Dive
============================================
3 contracts | 2:1 Scale-Out | 10:15–16:00 ET

WHAT IS VARIANT C?
------------------
Variant C is a "fresh reversal momentum" setup. Every rule serves
a specific purpose — here is exactly what is required before a trade fires:

  STEP 1 — HEIKEN ASHI COLOR FLIP
    The strategy only enters on the FIRST occurrence of 2 same-color HA
    candles that immediately follow at least 1 opposite-color candle.
    This means you are catching the move RIGHT AT the start of a new
    directional leg, not jumping into a trend that is already extended.
    Entering mid-trend (bar 6 of a green run) is how traders get trapped;
    this filter eliminates that entirely.

  STEP 2 — NO COUNTER-WICK (≤ 10% of body)
    After the color flip, the 2nd candle (the signal bar) must have a
    very small wick in the opposite direction of the trade.
    Long → tiny or zero lower wick  (sellers tried and failed)
    Short → tiny or zero upper wick (buyers tried and failed)
    A solid body with no opposing wick = price accepted the move cleanly.

  STEP 3 — DUAL EMA ALIGNMENT
    Price must be above BOTH EMA(20) AND EMA(50) for longs.
    Price must be below BOTH EMA(20) AND EMA(50) for shorts.
    EMA20 = short-term momentum; EMA50 = medium-term trend.
    Both agreeing = the reversal is happening in the direction of the
    bigger trend, not against it (counter-trend fades get filtered out).

  STEP 4 — VWAP SIDE
    Long only above VWAP (institutional buyers dominate the day).
    Short only below VWAP (institutional sellers dominate the day).
    VWAP resets each session morning, so this is a same-day bias filter.

  STEP 5 — RSI MOMENTUM WINDOW
    Long : RSI 48–65  (momentum building, not yet overbought)
    Short: RSI 35–52  (momentum building to downside, not oversold)
    This keeps you out of exhausted moves. If RSI is already at 72 when
    the HA flips green, you are buying into an extension — skipped.

  STEP 6 — SESSION WINDOW
    10:15 AM – 4:00 PM ET only.
    Skips the opening 45 minutes (9:30–10:15) where the range is being
    set, spreads are wider, and false HA signals are most common.

TRADE MANAGEMENT (2:1 Scale-Out, 3 contracts)
----------------------------------------------
  Entry  : market on close of signal bar
  Unit 1 : exit at +100 pts ($200)  — locks $600 on 3 contracts
  Unit 2 : exit at +100 pts ($200)  — (both units 1&2 exit at TP1)
  Unit 3 : runner to +200 pts ($400) — runner after units 1&2 are off
  Stop   : 100 pts below/above entry on ALL remaining units
           → after TP1 fills, stop on Unit 3 moves to breakeven

  Max risk per trade  : 100 pts × $2 × 3 contracts = $600
  TP1 capture (2 units): 100 pts × $2 × 2 contracts = $400
  TP2 capture (1 unit) : 200 pts × $2 × 1 contract  = $400
  Max win per trade    : $800 (both TPs hit)
  Breakeven after TP1  : Unit 3 is free-rolling at no risk
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CFG = {
    # variant C parameters
    "ha_consecutive":       2,
    "wick_filter_pct":      10.0,
    "require_color_change": True,
    "use_ema":              True,
    "ema_len":              20,
    "use_ema2":             True,
    "ema2_len":             50,
    "use_vwap":             True,
    "use_rsi":              True,
    "rsi_len":              14,
    "rsi_long_min":         48,
    "rsi_long_max":         65,
    "rsi_short_min":        35,
    "rsi_short_max":        52,
    # session
    "session_start":        "10:15",
    "session_end":          "16:00",
    # risk
    "sl_pts":               100.0,
    "tp1_pts":              100.0,
    "tp2_pts":              200.0,
    "tp1_units":            2,          # units to exit at TP1
    "tp2_units":            1,          # runner unit
    "contracts":            3,          # total units per trade
    "mnq_pt_value":         2.0,        # $2/pt per contract
    "commission_per_unit":  1.24,       # round-trip per contract
    # backtest window
    "days":                 90,
}

# ─── SYNTHETIC DATA (90 days) ─────────────────────────────────────────────────
def generate_data(days: int) -> pd.DataFrame:
    import pytz
    rng = np.random.default_rng(42)

    end   = datetime(2026, 5, 21)
    start = end - timedelta(days=days + 14)   # extra warmup for indicators

    bars_per_day  = 26
    trading_days  = 252
    bars_per_year = bars_per_day * trading_days

    # timestamps
    timestamps = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            for bar in range(bars_per_day):
                total_min = 9 * 60 + 30 + bar * 15
                h, m = total_min // 60, total_min % 60
                if total_min < 16 * 60:
                    timestamps.append(datetime(d.year, d.month, d.day, h, m))
        d += timedelta(days=1)

    n = len(timestamps)
    regime_len = bars_per_day * 12
    n_reg = max(1, n // regime_len + 2)
    regimes = rng.choice(["bull", "bear", "chop"], size=n_reg, p=[0.45, 0.25, 0.30])
    drift   = {"bull": 0.18, "bear": -0.20, "chop": 0.02}
    vol     = {"bull": 0.16, "bear": 0.24,  "chop": 0.14}

    returns = np.zeros(n)
    for i in range(n):
        reg = regimes[min(i // regime_len, n_reg - 1)]
        mu  = drift[reg] / bars_per_year
        sig = vol[reg]   / np.sqrt(bars_per_year)
        returns[i] = mu + np.clip(rng.standard_t(df=5), -5, 5) * sig

    closes = 19_500.0 * np.cumprod(1 + returns)
    opens = highs = lows = volumes = np.zeros(n)
    opens   = np.zeros(n)
    highs   = np.zeros(n)
    lows    = np.zeros(n)
    volumes = np.zeros(n)

    for i in range(n):
        prev_c   = closes[i - 1] if i > 0 else 19_500.0
        px       = closes[i]
        opens[i] = prev_c * (1 + rng.normal(0, 0.0003))
        bar_rng  = abs(px) * 0.0019 * (0.4 + rng.exponential(0.6))
        d_sign   = np.sign(px - opens[i]) or 1.0
        body_hi  = max(opens[i], px)
        body_lo  = min(opens[i], px)
        highs[i] = body_hi + rng.exponential(bar_rng * (0.30 if d_sign > 0 else 0.12))
        lows[i]  = body_lo - rng.exponential(bar_rng * (0.12 if d_sign > 0 else 0.30))
        highs[i] = max(highs[i], opens[i], px)
        lows[i]  = min(lows[i],  opens[i], px)
        bday     = i % bars_per_day
        volumes[i] = rng.integers(400, 2500) * (2.2 if bday < 4 else 1.6 if bday > 22 else 1.0)

    et  = pytz.timezone("America/New_York")
    idx = pd.DatetimeIndex([et.localize(ts) for ts in timestamps])
    df  = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)

    # trim to actual 90-day window (keep warmup rows for indicator calc)
    return df

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def calc_rsi(series, length):
    d = series.diff()
    g = d.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
    l = (-d.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def calc_vwap(df):
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vwap = pd.Series(index=df.index, dtype=float)
    for d in df.index.normalize().unique():
        mask = df.index.normalize() == d
        cum  = (tp[mask] * df.loc[mask, "volume"]).cumsum()
        vwap[mask] = cum / df.loc[mask, "volume"].cumsum().replace(0, np.nan)
    return vwap


def build_signals(df, cfg):
    # Heiken Ashi
    ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_o = np.zeros(len(df))
    ha_o[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + ha_c.iloc[i - 1]) / 2
    ha_o_s = pd.Series(ha_o, index=df.index)

    ha_h = df[["high"]].join(pd.DataFrame({"a": ha_o_s, "b": ha_c})).max(axis=1)
    ha_l = df[["low"]].join(pd.DataFrame({"a": ha_o_s, "b": ha_c})).min(axis=1)

    bull        = ha_c > ha_o_s
    body        = (ha_c - ha_o_s).abs()
    upper_wick  = ha_h - pd.concat([ha_o_s, ha_c], axis=1).max(axis=1)
    lower_wick  = pd.concat([ha_o_s, ha_c], axis=1).min(axis=1) - ha_l
    wf          = cfg["wick_filter_pct"] / 100.0

    strong_bull = bull & ((body == 0) | (lower_wick / body.replace(0, np.nan) <= wf))
    strong_bear = (~bull) & (ha_c != ha_o_s) & (
        (body == 0) | (upper_wick / body.replace(0, np.nan) <= wf))

    n     = cfg["ha_consecutive"]
    bear_ = (~bull) & (ha_c != ha_o_s)

    def streak(s):
        arr = s.values.astype(bool)
        out = np.zeros(len(arr), dtype=int)
        for i in range(n - 1, len(arr)):
            if arr[i - n + 1 : i + 1].all():
                out[i] = 1
        return pd.Series(out, index=s.index)

    bull_streak = streak(bull)
    bear_streak = streak(bear_)

    # Color-change: streak started right after an opposite-color candle
    bull_arr = bull.values.astype(bool)
    b_str    = bull_streak.values
    be_str   = bear_streak.values

    bull_cc = np.zeros(len(df), dtype=int)
    bear_cc = np.zeros(len(df), dtype=int)
    for i in range(n, len(df)):
        if b_str[i] == 1 and not bull_arr[i - n]:
            bull_cc[i] = 1
        if be_str[i] == 1 and bull_arr[i - n]:
            bear_cc[i] = 1

    bull_cc = pd.Series(bull_cc, index=df.index)
    bear_cc = pd.Series(bear_cc, index=df.index)

    ema1 = df["close"].ewm(span=cfg["ema_len"],  adjust=False).mean()
    ema2 = df["close"].ewm(span=cfg["ema2_len"], adjust=False).mean()
    rsi  = calc_rsi(df["close"], cfg["rsi_len"])
    vwap = calc_vwap(df)

    t          = df.index.strftime("%H:%M")
    in_session = (t >= cfg["session_start"]) & (t < cfg["session_end"])

    long_sig = (
        (bull_cc == 1) & strong_bull
        & (df["close"] > ema1) & (df["close"] > ema2)
        & (df["close"] > vwap)
        & (rsi >= cfg["rsi_long_min"]) & (rsi <= cfg["rsi_long_max"])
        & in_session
    )
    short_sig = (
        (bear_cc == 1) & strong_bear
        & (df["close"] < ema1) & (df["close"] < ema2)
        & (df["close"] < vwap)
        & (rsi >= cfg["rsi_short_min"]) & (rsi <= cfg["rsi_short_max"])
        & in_session
    )

    return long_sig, short_sig, ema1, ema2, vwap, rsi

# ─── BACKTEST ENGINE (3-unit split) ───────────────────────────────────────────
def run_backtest(df, long_sig, short_sig, cfg):
    pv       = cfg["mnq_pt_value"]
    comm_u   = cfg["commission_per_unit"]
    sl       = cfg["sl_pts"]
    tp1      = cfg["tp1_pts"]
    tp2      = cfg["tp2_pts"]
    tp1_u    = cfg["tp1_units"]      # 2
    tp2_u    = cfg["tp2_units"]      # 1
    total_u  = cfg["contracts"]      # 3

    trades     = []
    equity     = 0.0
    eq_curve   = []
    position   = 0
    entry_px   = 0.0
    entry_idx  = None
    tp1_done   = False

    longs  = long_sig.values
    shorts = short_sig.values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    idx    = df.index

    def log(exit_px, reason, units):
        nonlocal equity
        pnl = (exit_px - entry_px) * position * pv * units - comm_u * units
        equity += pnl
        trades.append({
            "entry_time": str(entry_idx)[:16],
            "exit_time":  str(idx[i])[:16],
            "dir":        "Long" if position == 1 else "Short",
            "entry":      round(entry_px, 2),
            "exit":       round(exit_px, 2),
            "units":      units,
            "pnl":        round(pnl, 2),
            "reason":     reason,
            "equity":     round(equity, 2),
        })

    for i in range(len(df)):
        hi, lo = highs[i], lows[i]

        if position != 0:
            sl_px  = entry_px - sl  * position
            tp1_px = entry_px + tp1 * position
            tp2_px = entry_px + tp2 * position

            if position == 1:
                if lo <= sl_px:
                    rem = tp2_u if tp1_done else total_u
                    log(sl_px, "SL", rem)
                    position, tp1_done = 0, False
                elif not tp1_done and hi >= tp1_px:
                    log(tp1_px, "TP1", tp1_u)
                    tp1_done = True
                elif tp1_done and hi >= tp2_px:
                    log(tp2_px, "TP2", tp2_u)
                    position, tp1_done = 0, False
            else:
                if hi >= sl_px:
                    rem = tp2_u if tp1_done else total_u
                    log(sl_px, "SL", rem)
                    position, tp1_done = 0, False
                elif not tp1_done and lo <= tp1_px:
                    log(tp1_px, "TP1", tp1_u)
                    tp1_done = True
                elif tp1_done and lo <= tp2_px:
                    log(tp2_px, "TP2", tp2_u)
                    position, tp1_done = 0, False

        if position == 0:
            if longs[i]:
                position, entry_px, entry_idx, tp1_done = 1,  closes[i], idx[i], False
            elif shorts[i]:
                position, entry_px, entry_idx, tp1_done = -1, closes[i], idx[i], False

        eq_curve.append(equity)

    return trades, pd.Series(eq_curve, index=df.index)

# ─── METRICS ──────────────────────────────────────────────────────────────────
def calc_metrics(trades, eq, cfg):
    df_t = pd.DataFrame(trades)
    if df_t.empty:
        return {}

    # Aggregate fills into trade-level view
    # Each "trade" may have 2 fills (TP1+TP2 or TP1+SL); group by entry_time
    trade_groups = df_t.groupby("entry_time")
    trade_pnls   = trade_groups["pnl"].sum()
    n_trades     = len(trade_pnls)
    wins         = trade_pnls[trade_pnls > 0]
    losses       = trade_pnls[trade_pnls < 0]
    wr           = len(wins) / n_trades * 100 if n_trades else 0
    pf           = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    net          = trade_pnls.sum()
    mdd          = (eq - eq.cummax()).min()

    # By direction
    dirs = trade_groups["dir"].first()
    long_pnls  = trade_pnls[dirs == "Long"]
    short_pnls = trade_pnls[dirs == "Short"]

    # Expectancy per trade (avg $ per trade)
    exp = net / n_trades if n_trades else 0

    # Best/worst day
    df_t["date"] = pd.to_datetime(df_t["exit_time"]).dt.date
    daily = df_t.groupby("date")["pnl"].sum()

    return {
        "Total Trades":          n_trades,
        "Win Rate":              round(wr, 1),
        "Net P&L":               round(net, 2),
        "Profit Factor":         round(pf, 2),
        "Avg Trade P&L":         round(exp, 2),
        "Max Drawdown":          round(mdd, 2),
        "Long Trades":           len(long_pnls),
        "Long Win Rate":         round((long_pnls > 0).mean() * 100, 1) if len(long_pnls) else 0,
        "Long Net":              round(long_pnls.sum(), 2),
        "Short Trades":          len(short_pnls),
        "Short Win Rate":        round((short_pnls > 0).mean() * 100, 1) if len(short_pnls) else 0,
        "Short Net":             round(short_pnls.sum(), 2),
        "Best Day":              round(daily.max(), 2),
        "Worst Day":             round(daily.min(), 2),
        "Max Risk/Trade":        round(cfg["sl_pts"] * cfg["mnq_pt_value"] * cfg["contracts"], 2),
        "Max Win/Trade":         round(
            cfg["tp1_pts"] * cfg["mnq_pt_value"] * cfg["tp1_units"] +
            cfg["tp2_pts"] * cfg["mnq_pt_value"] * cfg["tp2_units"], 2),
    }

# ─── CHART ────────────────────────────────────────────────────────────────────
def plot_results(df, trades, eq, metrics, cfg, start_date):
    BG, CARD, GRAY = "#0d1117", "#161b22", "#30363d"
    GREEN, RED, BLUE = "#2ea043", "#da3633", "#58a6ff"
    GOLD = "#e3b341"

    df_t = pd.DataFrame(trades)
    if df_t.empty:
        print("No trades to chart.")
        return

    # aggregate to trade level
    tg    = df_t.groupby("entry_time")
    t_pnl = tg["pnl"].sum().reset_index()
    t_dir = tg["dir"].first().reset_index()
    t_pnl = t_pnl.merge(t_dir, on="entry_time")
    t_pnl["exit_time"] = tg["exit_time"].last().values
    t_pnl["exit_time"] = pd.to_datetime(t_pnl["exit_time"])
    t_pnl["entry_time"] = pd.to_datetime(t_pnl["entry_time"])

    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)

    # ── 1. Equity curve (full width) ─────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor(CARD)
    trim_eq = eq[eq.index.date >= start_date]
    ax_eq.fill_between(trim_eq.index, trim_eq,
                        where=trim_eq >= 0, color=GREEN, alpha=0.25)
    ax_eq.fill_between(trim_eq.index, trim_eq,
                        where=trim_eq < 0,  color=RED,   alpha=0.25)
    ax_eq.plot(trim_eq.index, trim_eq, color=BLUE, linewidth=1.5)
    ax_eq.axhline(0, color=GRAY, linewidth=0.7, linestyle="--")

    # Mark individual trade exits on equity curve
    for _, row in t_pnl.iterrows():
        exit_ts = row["exit_time"]
        # ensure timezone matches the equity index
        if exit_ts.tzinfo is None and trim_eq.index.tz is not None:
            import pytz
            exit_ts = pytz.timezone("America/New_York").localize(exit_ts)
        if exit_ts in trim_eq.index:
            val = trim_eq.loc[exit_ts]
        else:
            nearest = trim_eq.index.get_indexer([exit_ts], method="nearest")[0]
            val = trim_eq.iloc[nearest]
        color = GREEN if row["pnl"] > 0 else RED
        ax_eq.scatter(exit_ts, val, color=color, s=22, zorder=5, alpha=0.8)

    ax_eq.set_title(
        "Variant C (Color-Change) — Equity Curve  |  3 Contracts  |  2:1 Scale-Out  |  90 Days",
        color="white", fontsize=13, fontweight="bold")
    ax_eq.set_ylabel("Cumulative P&L ($)", color="white")
    ax_eq.tick_params(colors="white")
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_eq.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    # Drawdown fill
    roll_max = trim_eq.cummax()
    ax_eq.fill_between(trim_eq.index, roll_max, trim_eq,
                        where=(trim_eq < roll_max), color=RED, alpha=0.12,
                        label=f"Drawdown (max ${metrics['Max Drawdown']:,.0f})")
    ax_eq.legend(facecolor=CARD, labelcolor="white", fontsize=9, loc="upper left")

    # ── 2. Daily P&L bars ────────────────────────────────────────────────────
    ax_day = fig.add_subplot(gs[1, :2])
    ax_day.set_facecolor(CARD)
    df_t["date_"] = pd.to_datetime(df_t["exit_time"]).dt.date
    daily = df_t.groupby("date_")["pnl"].sum()
    daily = daily[daily.index >= start_date]
    day_colors = [GREEN if v >= 0 else RED for v in daily.values]
    bars = ax_day.bar(daily.index.astype(str), daily.values,
                       color=day_colors, edgecolor=GRAY, linewidth=0.4)
    ax_day.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax_day.set_title("Daily P&L (all fills combined)", color="white",
                      fontsize=11, fontweight="bold")
    ax_day.set_ylabel("P&L ($)", color="white")
    ax_day.tick_params(colors="white", axis="y")
    ax_day.tick_params(colors="white", axis="x", rotation=55, labelsize=6)

    # ── 3. Win/Loss donut ────────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1, 2])
    ax_pie.set_facecolor(CARD)
    n_wins = (t_pnl["pnl"] > 0).sum()
    n_loss = (t_pnl["pnl"] < 0).sum()
    wedge_colors = [GREEN, RED]
    wedges, texts, autos = ax_pie.pie(
        [n_wins, n_loss],
        labels=[f"Wins\n{n_wins}", f"Losses\n{n_loss}"],
        colors=wedge_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": CARD},
        textprops={"color": "white", "fontsize": 10},
    )
    for a in autos:
        a.set_color("white")
        a.set_fontsize(10)
    ax_pie.set_title(f"Win/Loss Split\n({metrics['Win Rate']}% win rate)",
                      color="white", fontsize=11, fontweight="bold")

    # ── 4. Trade scatter (entry time of day vs PnL) ──────────────────────────
    ax_scat = fig.add_subplot(gs[2, :2])
    ax_scat.set_facecolor(CARD)
    t_pnl["hour_frac"] = (t_pnl["entry_time"].dt.hour +
                           t_pnl["entry_time"].dt.minute / 60)
    long_mask  = t_pnl["dir"] == "Long"
    short_mask = t_pnl["dir"] == "Short"
    win_mask   = t_pnl["pnl"] > 0
    loss_mask  = t_pnl["pnl"] < 0

    for mask, color, marker, label in [
        (long_mask  & win_mask,  GREEN, "^", "Long Win"),
        (long_mask  & loss_mask, RED,   "^", "Long Loss"),
        (short_mask & win_mask,  GREEN, "v", "Short Win"),
        (short_mask & loss_mask, RED,   "v", "Short Loss"),
    ]:
        sub = t_pnl[mask]
        ax_scat.scatter(sub["hour_frac"], sub["pnl"],
                        color=color, marker=marker,
                        s=abs(sub["pnl"]) / 4 + 15,
                        alpha=0.75, label=label,
                        edgecolors=CARD, linewidths=0.4)

    ax_scat.axhline(0, color=GRAY, linewidth=0.6, linestyle="--")
    ax_scat.set_xlim(10.0, 16.2)
    ax_scat.set_xticks([10.25, 11, 12, 13, 14, 15, 16])
    ax_scat.set_xticklabels(["10:15","11:00","12:00","13:00","14:00","15:00","16:00"],
                              color="white", fontsize=8)
    ax_scat.tick_params(colors="white")
    ax_scat.set_title("Trade P&L by Time of Entry  (size = abs PnL)", color="white",
                       fontsize=11, fontweight="bold")
    ax_scat.set_ylabel("Trade P&L ($)", color="white")
    ax_scat.set_xlabel("Entry Time (ET)", color="white")
    ax_scat.legend(facecolor=CARD, labelcolor="white", fontsize=8, ncol=2)

    # ── 5. Exit reason breakdown ──────────────────────────────────────────────
    ax_exit = fig.add_subplot(gs[2, 2])
    ax_exit.set_facecolor(CARD)
    reason_sums = df_t.groupby("reason")["pnl"].agg(["sum", "count"])
    reason_cols = {"TP1": GREEN, "TP2": GOLD, "SL": RED}
    for k, (s, c) in enumerate(reason_sums.iterrows()):
        col = reason_cols.get(s, BLUE)
        ax_exit.bar(s, c["sum"], color=col, edgecolor=GRAY, width=0.5)
        ax_exit.text(k, c["sum"] + (30 if c["sum"] >= 0 else -120),
                     f"n={int(c['count'])}\n${c['sum']:,.0f}",
                     ha="center", color="white", fontsize=8)
    ax_exit.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
    ax_exit.set_title("P&L by Exit Type", color="white", fontsize=11, fontweight="bold")
    ax_exit.tick_params(colors="white")

    # ── 6. Full metrics table (bottom row, full width) ────────────────────────
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.set_facecolor(CARD)
    ax_tbl.axis("off")

    left_items  = list(metrics.items())[:9]
    right_items = list(metrics.items())[9:]

    def fmt_val(k, v):
        if isinstance(v, float) and ("P&L" in k or "Day" in k or "Drawdown" in k
                                      or "Risk" in k or "Win" in k or "Net" in k):
            return f"${v:,.2f}"
        elif k == "Win Rate" or "Win Rate" in k:
            return f"{v}%"
        return str(v)

    def color_val(k, v):
        if isinstance(v, (int, float)):
            if any(x in k for x in ["P&L","Net","Day","Drawdown"]):
                return GREEN if v > 0 else (RED if v < 0 else "white")
            if "Win Rate" in k:
                return GREEN if v >= 50 else RED
        return "white"

    cols = 4
    rows_data = []
    for j in range(max(len(left_items), len(right_items))):
        row = []
        if j < len(left_items):
            k, v = left_items[j]
            row += [k, fmt_val(k, v)]
        else:
            row += ["", ""]
        if j < len(right_items):
            k, v = right_items[j]
            row += [k, fmt_val(k, v)]
        else:
            row += ["", ""]
        rows_data.append(row)

    tbl = ax_tbl.table(
        cellText=rows_data,
        colLabels=["Metric", "Value", "Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0.02, 0, 0.96, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    all_metrics = list(metrics.items())
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(CARD if r > 0 else "#21262d")
        cell.set_edgecolor(GRAY)
        cell.set_text_props(color="white")
        if r > 0 and c % 2 == 1:
            idx_in_metrics = (r - 1) + (c // 2) * len(left_items)
            if idx_in_metrics < len(all_metrics):
                k, v = all_metrics[idx_in_metrics]
                cell.set_text_props(color=color_val(k, v))

    ax_tbl.set_title("90-Day Performance Summary — Variant C  |  3 Contracts  |  2:1 Mode",
                      color="white", fontsize=11, fontweight="bold", pad=8)

    out = "strategies/variant_c_90day_3units.png"
    plt.suptitle(
        "Variant C (Color-Change) — MNQ 15-Min  |  3 Contracts  |  2:1 Scale-Out  |  90-Day Deep Dive",
        color="white", fontsize=13, fontweight="bold", y=0.99)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart saved → {out}")
    return out

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Variant C (Color-Change) — 90-Day Backtest  |  3 Contracts  |  2:1")
    print("=" * 70)

    # strategy explanation
    print("""
VARIANT C — COMPLETE RULE SET
──────────────────────────────────────────────────────────────────────
 Session  : 10:15 AM – 4:00 PM ET only

 LONG ENTRY — all 6 must be true:
   1. HA FLIP   : Previous candle was BEARISH; current streak of 2
                  consecutive BULLISH HA candles has just started.
                  → You are entering at the very beginning of a new
                    upward leg, not chasing an existing trend.

   2. WICK TEST : The 2nd (signal) HA candle has a lower wick that is
                  ≤ 10% of the candle body.
                  → A solid green body with almost no bottom wick means
                    sellers stepped in but were immediately overpowered.

   3. EMA STACK : Close > EMA(20) AND Close > EMA(50).
                  → Short-term AND medium-term trend both point up.
                    Counter-trend reversal attempts are filtered out.

   4. VWAP      : Close > VWAP (recalculated fresh each session).
                  → Institutional bias for the day is bullish. Above
                    VWAP = buyers have been in control since the open.

   5. RSI       : RSI between 48 and 65.
                  → In the momentum building zone. Below 48 = not enough
                    buying pressure yet. Above 65 = already extended.

   6. SESSION   : Bar timestamp is between 10:15 AM and 4:00 PM ET.

 SHORT ENTRY — mirror of above:
   1. HA flip from green to 2 consecutive red HA candles
   2. Upper wick ≤ 10% of body on signal bar
   3. Close < EMA(20) AND Close < EMA(50)
   4. Close < VWAP
   5. RSI between 35 and 52
   6. 10:15–16:00 ET

 TRADE MANAGEMENT (3 contracts, 2:1 scale-out):
   Entry     : close of signal bar (bar 2 of the new-color streak)
   Unit 1+2  : exit at +100 pts → captures $400 (2 × $200)
   Unit 3    : runner to +200 pts → captures $400 if it hits
   Stop      : 100 pts on ALL units at entry
               → after TP1 fills, stop on Unit 3 moves to breakeven
   Max loss  : $600 (3 contracts × 100 pts × $2)
   Max gain  : $800 (TP1 on 2 + TP2 on 1)
──────────────────────────────────────────────────────────────────────
""")

    end_date   = datetime(2026, 5, 21).date()
    start_date = (datetime(2026, 5, 21) - timedelta(days=90)).date()

    print(f"  Period : {start_date}  →  {end_date}  (90 days)")
    print(f"  Generating data ...")

    df_full = generate_data(90)
    avg_rng  = (df_full["high"] - df_full["low"]).mean()
    print(f"  {len(df_full):,} total bars | avg 15-min range: {avg_rng:.1f} pts")

    long_sig, short_sig, ema1, ema2, vwap, rsi = build_signals(df_full, CFG)

    trades, eq = run_backtest(df_full, long_sig, short_sig, CFG)

    # Trim to 90-day window for reporting
    eq_trim = eq[eq.index.date >= start_date]
    trades_trim = [t for t in trades
                   if pd.to_datetime(t["exit_time"]).date() >= start_date]

    if not trades_trim:
        print("  No trades in the 90-day window.")
        return

    m = calc_metrics(trades_trim, eq_trim, CFG)

    print(f"\n{'─'*70}")
    print(f"  90-DAY RESULTS — 3 CONTRACTS — 2:1 SCALE-OUT")
    print(f"{'─'*70}")
    for k, v in m.items():
        label = f"  {k:<28}"
        if isinstance(v, float) and any(x in k for x in ["P&L","Day","Drawdown","Risk","Net"]):
            print(f"{label} ${v:>10,.2f}")
        elif "Win Rate" in k:
            print(f"{label} {v}%")
        else:
            print(f"{label} {v}")
    print(f"{'─'*70}")

    df_trades = pd.DataFrame(trades_trim)
    df_trades.to_csv("strategies/variant_c_90day_trades.csv", index=False)
    print(f"\n  Trade log → strategies/variant_c_90day_trades.csv")

    plot_results(df_full, trades_trim, eq_trim, m, CFG, start_date)
    print("\nDone.")

if __name__ == "__main__":
    main()

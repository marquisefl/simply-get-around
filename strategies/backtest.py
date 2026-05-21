"""
Heiken Ashi MNQ 15-Min Strategy — Python Backtester
Uses NQ=F (E-mini NASDAQ) data from Yahoo Finance scaled to MNQ dollar values.
MNQ = NQ / 10 in terms of dollar value ($2/pt vs $20/pt).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "symbol":           "NQ=F",         # E-mini NQ futures (MNQ tracks identically)
    "period_years":     2,              # How many years to backtest
    "timeframe":        "15m",

    # Heiken Ashi
    "ha_consecutive":   2,              # Min same-color HA candles to signal
    "wick_filter_pct":  15.0,           # Max counter-wick as % of body

    # Trend filters
    "use_ema":          True,
    "ema_len":          20,
    "use_vwap":         True,           # Daily VWAP recalc
    "use_rsi":          True,
    "rsi_len":          14,
    "rsi_long_min":     45,
    "rsi_long_max":     70,
    "rsi_short_min":    30,
    "rsi_short_max":    55,

    # Risk management (in NQ points — same for MNQ numerically)
    "sl_pts":           100.0,
    "tp1_pts":          100.0,
    "use_tp2":          False,          # Toggle 2:1 scale-out
    "tp2_pts":          200.0,
    "tp1_pct":          0.50,           # % of position to exit at TP1 when use_tp2=True

    # Dollar values (MNQ = $2/pt)
    "mnq_pt_value":     2.0,
    "commission":       1.24,           # Round-trip ($0.62 each side)
    "contracts":        1,

    # Session filter (ET hours)
    "use_session":      True,
    "kill_zone_only":   False,
    "session_start":    "09:30",
    "session_end":      "16:00",
    "kill_zones":       [("09:30", "11:00"), ("13:30", "15:00")],
}

# ─── SYNTHETIC DATA GENERATOR ─────────────────────────────────────────────────
# Mimics realistic NQ/MNQ price action:
#   - Regime-switching GBM (trending up/sideways/down periods)
#   - Fat-tailed intrabar price discovery (realistic OHLC relationships)
#   - Session-aware timestamps (NYSE hours, weekdays only)
#   - Calibrated to NQ historical statistics (~18% annual vol, ~40 pt daily range)

def generate_synthetic_nq(cfg: dict) -> pd.DataFrame:
    """
    Realistic NQ 15-min synthetic data:
      - Calibrated to NQ historical stats (~18% annual vol, ~0.18% per 15-min bar)
      - Regime-switching (bull / bear / chop) with proper per-bar drift
      - Fat-tailed returns (Student-t df=5)
      - Realistic OHLC spread relative to price level
    """
    rng = np.random.default_rng(42)

    end   = datetime(2026, 5, 21)
    start = end - timedelta(days=cfg["period_years"] * 365)
    bars_per_day   = 26        # 9:30–16:00 ET = 26 bars
    trading_days   = 252
    bars_per_year  = bars_per_day * trading_days  # ≈ 6,552

    session_open  = (9, 30)
    session_close = (16, 0)

    # Build timestamp index
    timestamps = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            for bar in range(bars_per_day):
                total_min = session_open[0] * 60 + session_open[1] + bar * 15
                h = total_min // 60
                m = total_min % 60
                if total_min < session_close[0] * 60 + session_close[1]:
                    timestamps.append(datetime(d.year, d.month, d.day, h, m))
        d += timedelta(days=1)

    n = len(timestamps)

    # ── Calibrated per-bar parameters ────────────────────────────────────────
    # Annual vol ≈ 20%  → per-bar vol = 0.20 / sqrt(6552) ≈ 0.00247
    # Regime drifts annualized, then converted to per-bar
    annual_drift = {"bull": 0.18, "bear": -0.20, "chop": 0.02}
    annual_vol   = {"bull": 0.16, "bear": 0.24,  "chop": 0.14}

    regime_len = bars_per_day * 15   # ~15 trading days per regime
    n_regimes  = max(1, n // regime_len + 2)
    regimes    = rng.choice(["bull", "bear", "chop"],
                            size=n_regimes, p=[0.45, 0.25, 0.30])

    returns = np.zeros(n)
    for i in range(n):
        r_idx = min(i // regime_len, len(regimes) - 1)
        reg   = regimes[r_idx]
        mu    = annual_drift[reg] / bars_per_year
        sig   = annual_vol[reg]   / np.sqrt(bars_per_year)
        t_draw = rng.standard_t(df=5)
        t_draw = np.clip(t_draw, -5, 5)
        returns[i] = mu + sig * t_draw

    start_price = 19_500.0
    closes = start_price * np.cumprod(1 + returns)

    # ── Realistic intrabar OHLC (spread relative to price) ───────────────────
    # NQ avg 15-min bar range ≈ 35-50 pts on a ~20k price ≈ 0.175-0.25%
    intrabar_vol_frac = 0.0019    # ≈ 38 pts at 20k

    opens   = np.zeros(n)
    highs   = np.zeros(n)
    lows    = np.zeros(n)
    volumes = np.zeros(n)

    for i in range(n):
        prev_c = closes[i - 1] if i > 0 else start_price
        px     = closes[i]

        # Open: small gap from prior close (most intraday bars open near prev close)
        gap_frac  = rng.normal(0, 0.0003)
        opens[i]  = prev_c * (1 + gap_frac)

        # Bar range: exponential to capture occasional large bars
        bar_range = abs(px) * intrabar_vol_frac * (0.4 + rng.exponential(0.6))
        direction = np.sign(px - opens[i]) if abs(px - opens[i]) > 0.1 else (1 if rng.random() > 0.5 else -1)

        # Wicks: longer in direction of close, shorter counter
        wick_with  = rng.exponential(bar_range * 0.25)
        wick_against = rng.exponential(bar_range * 0.10)

        body_high  = max(opens[i], px)
        body_low   = min(opens[i], px)
        highs[i]   = body_high + (wick_with   if direction > 0 else wick_against)
        lows[i]    = body_low  - (wick_against if direction > 0 else wick_with)
        highs[i]   = max(highs[i], opens[i], px)
        lows[i]    = min(lows[i],  opens[i], px)

        # Volume
        bar_of_day = i % bars_per_day
        vol_mult   = 2.2 if bar_of_day < 4 else (1.6 if bar_of_day > 22 else 1.0)
        volumes[i] = rng.integers(400, 2500) * vol_mult

    import pytz
    et  = pytz.timezone("America/New_York")
    idx = pd.DatetimeIndex([et.localize(ts) for ts in timestamps])

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }, index=idx)

    avg_range = (df["high"] - df["low"]).mean()
    print(f"  Synthetic NQ data: {len(df):,} bars  ({start.date()} → {end.date()})")
    print(f"  Price range: {df['close'].min():.0f} – {df['close'].max():.0f}  "
          f"|  Avg 15-min bar range: {avg_range:.1f} pts")
    return df


def fetch_data(cfg: dict) -> pd.DataFrame:
    print(f"Generating synthetic NQ 15-min data ({cfg['period_years']} years) ...")
    return generate_synthetic_nq(cfg)

# ─── INDICATOR CALCULATIONS ───────────────────────────────────────────────────
def calc_heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha["ha_c"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["ha_o"] = np.nan

    for i in range(len(ha)):
        if i == 0:
            ha.iloc[i, ha.columns.get_loc("ha_o")] = (
                df.iloc[i]["open"] + df.iloc[i]["close"]
            ) / 2
        else:
            ha.iloc[i, ha.columns.get_loc("ha_o")] = (
                ha.iloc[i - 1]["ha_o"] + ha.iloc[i - 1]["ha_c"]
            ) / 2

    ha["ha_h"] = ha[["high", "ha_o", "ha_c"]].max(axis=1)
    ha["ha_l"] = ha[["low",  "ha_o", "ha_c"]].min(axis=1)
    return ha


def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Daily VWAP recalculated each session."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = pd.Series(index=df.index, dtype=float)

    dates = df.index.normalize().unique()
    for d in dates:
        mask = df.index.normalize() == d
        tp_d  = tp[mask]
        vol_d = df.loc[mask, "volume"]
        cum_tp_vol = (tp_d * vol_d).cumsum()
        cum_vol    = vol_d.cumsum()
        vwap[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)

    return vwap


def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = calc_heiken_ashi(df)

    df["ema"]  = df["close"].ewm(span=cfg["ema_len"], adjust=False).mean()
    df["rsi"]  = calc_rsi(df["close"], cfg["rsi_len"])
    df["vwap"] = calc_vwap(df)

    # HA candle properties
    df["bull"]         = df["ha_c"] > df["ha_o"]
    df["body"]         = (df["ha_c"] - df["ha_o"]).abs()
    df["upper_wick"]   = df["ha_h"] - df[["ha_o", "ha_c"]].max(axis=1)
    df["lower_wick"]   = df[["ha_o", "ha_c"]].min(axis=1) - df["ha_l"]

    wf = cfg["wick_filter_pct"] / 100.0
    df["strong_bull"]  = df["bull"] & (
        (df["body"] == 0) | (df["lower_wick"] / df["body"].replace(0, np.nan) <= wf)
    )
    df["strong_bear"]  = ~df["bull"] & (df["ha_c"] != df["ha_o"]) & (
        (df["body"] == 0) | (df["upper_wick"] / df["body"].replace(0, np.nan) <= wf)
    )

    # Consecutive streaks
    n = cfg["ha_consecutive"]

    def rolling_streak(series: pd.Series) -> pd.Series:
        out = pd.Series(0, index=series.index)
        for i in range(n - 1, len(series)):
            if series.iloc[i - n + 1 : i + 1].all():
                out.iloc[i] = 1
        return out

    df["bull_streak"]  = rolling_streak(df["bull"])
    df["bear_streak"]  = rolling_streak(~df["bull"] & (df["ha_c"] != df["ha_o"]))

    df["ha_long"]      = (df["bull_streak"] == 1) & df["strong_bull"]
    df["ha_short"]     = (df["bear_streak"] == 1) & df["strong_bear"]

    return df


def add_session_flag(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    time_str = df.index.strftime("%H:%M")

    if not cfg["use_session"]:
        df["in_session"] = True
        return df

    if cfg["kill_zone_only"]:
        flag = pd.Series(False, index=df.index)
        for start, end in cfg["kill_zones"]:
            flag |= (time_str >= start) & (time_str < end)
        df["in_session"] = flag
    else:
        df["in_session"] = (
            (time_str >= cfg["session_start"]) & (time_str < cfg["session_end"])
        )

    return df

# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple[list, pd.DataFrame]:
    """
    Event-driven bar-by-bar backtest.
    Returns (trades_list, equity_series).
    """
    pt   = cfg["mnq_pt_value"] * cfg["contracts"]
    comm = cfg["commission"]
    sl   = cfg["sl_pts"]
    tp1  = cfg["tp1_pts"]
    tp2  = cfg["tp2_pts"]
    tp2_on = cfg["use_tp2"]
    tp1_p  = cfg["tp1_pct"]

    trades   = []
    equity   = 0.0
    eq_curve = []

    position  = 0        # 0=flat, 1=long, -1=short
    entry_px  = 0.0
    entry_idx = None
    tp1_filled = False   # for 2:1 scale mode

    for i, (ts, row) in enumerate(df.iterrows()):
        # ── Manage open position ──────────────────────────────────────────────
        if position != 0:
            hi, lo = row["high"], row["low"]

            def log_fill(exit_px, reason, qty_pct):
                nonlocal equity
                pnl = (exit_px - entry_px) * position * pt * qty_pct - comm * qty_pct
                equity += pnl
                trades.append({
                    "entry_time": entry_idx,
                    "exit_time":  ts,
                    "direction":  "Long" if position == 1 else "Short",
                    "entry":      entry_px,
                    "exit":       exit_px,
                    "qty_pct":    qty_pct,
                    "pnl":        round(pnl, 2),
                    "reason":     reason,
                    "cum_equity": round(equity, 2),
                })

            sl_hit_px  = entry_px - sl  * position
            tp1_hit_px = entry_px + tp1 * position
            tp2_hit_px = entry_px + tp2 * position

            if position == 1:   # ── Long management ──────────────────────────
                sl_hit = lo <= sl_hit_px
                if not tp2_on:
                    if sl_hit:
                        log_fill(sl_hit_px, "SL", 1.0)
                        position, tp1_filled = 0, False
                    elif hi >= tp1_hit_px:
                        log_fill(tp1_hit_px, "TP1", 1.0)
                        position, tp1_filled = 0, False
                else:
                    if sl_hit:
                        rem = 1.0 - tp1_p if tp1_filled else 1.0
                        log_fill(sl_hit_px, "SL", rem)
                        position, tp1_filled = 0, False
                    elif not tp1_filled and hi >= tp1_hit_px:
                        log_fill(tp1_hit_px, "TP1", tp1_p)
                        tp1_filled = True
                    elif tp1_filled and hi >= tp2_hit_px:
                        log_fill(tp2_hit_px, "TP2", 1.0 - tp1_p)
                        position, tp1_filled = 0, False

            else:               # ── Short management ─────────────────────────
                sl_hit = hi >= sl_hit_px
                if not tp2_on:
                    if sl_hit:
                        log_fill(sl_hit_px, "SL", 1.0)
                        position, tp1_filled = 0, False
                    elif lo <= tp1_hit_px:
                        log_fill(tp1_hit_px, "TP1", 1.0)
                        position, tp1_filled = 0, False
                else:
                    if sl_hit:
                        rem = 1.0 - tp1_p if tp1_filled else 1.0
                        log_fill(sl_hit_px, "SL", rem)
                        position, tp1_filled = 0, False
                    elif not tp1_filled and lo <= tp1_hit_px:
                        log_fill(tp1_hit_px, "TP1", tp1_p)
                        tp1_filled = True
                    elif tp1_filled and lo <= tp2_hit_px:
                        log_fill(tp2_hit_px, "TP2", 1.0 - tp1_p)
                        position, tp1_filled = 0, False

        # ── Entry signals ─────────────────────────────────────────────────────
        if position == 0 and row["in_session"]:
            long_ok = (
                row["ha_long"]
                and (not cfg["use_ema"]  or row["close"] > row["ema"])
                and (not cfg["use_vwap"] or row["close"] > row["vwap"])
                and (not cfg["use_rsi"]  or
                     (row["rsi"] >= cfg["rsi_long_min"] and row["rsi"] <= cfg["rsi_long_max"]))
            )
            short_ok = (
                row["ha_short"]
                and (not cfg["use_ema"]  or row["close"] < row["ema"])
                and (not cfg["use_vwap"] or row["close"] < row["vwap"])
                and (not cfg["use_rsi"]  or
                     (row["rsi"] >= cfg["rsi_short_min"] and row["rsi"] <= cfg["rsi_short_max"]))
            )

            if long_ok:
                position  = 1
                entry_px  = row["close"]
                entry_idx = ts
                tp1_filled = False

            elif short_ok:
                position  = -1
                entry_px  = row["close"]
                entry_idx = ts
                tp1_filled = False

        eq_curve.append({"time": ts, "equity": equity})

    eq_series = pd.DataFrame(eq_curve).set_index("time")["equity"]
    return trades, eq_series

# ─── PERFORMANCE METRICS ──────────────────────────────────────────────────────
def calc_metrics(trades: list, eq: pd.Series) -> dict:
    if not trades:
        return {}

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["pnl"] > 0]
    loss = df_t[df_t["pnl"] < 0]

    total        = len(df_t)
    win_rate     = len(wins) / total * 100 if total else 0
    avg_win      = wins["pnl"].mean()  if len(wins) else 0
    avg_loss     = loss["pnl"].mean()  if len(loss) else 0
    profit_factor = (wins["pnl"].sum() / abs(loss["pnl"].sum())
                     if loss["pnl"].sum() != 0 else np.inf)

    net_pnl      = df_t["pnl"].sum()
    gross_win    = wins["pnl"].sum()
    gross_loss   = loss["pnl"].sum()

    # Max drawdown
    rolling_max = eq.cummax()
    drawdown    = eq - rolling_max
    max_dd      = drawdown.min()

    # Calmar (annual return / max drawdown)
    days         = (eq.index[-1] - eq.index[0]).days
    annual_ret   = (net_pnl / max(days, 1)) * 365 if days else 0
    calmar       = abs(annual_ret / max_dd) if max_dd != 0 else np.inf

    # Consecutive win/loss streaks
    results      = (df_t["pnl"] > 0).astype(int).values
    max_cons_win = max_cons_loss = cur = 0
    sign         = None
    for r in results:
        if r == sign:
            cur += 1
        else:
            cur, sign = 1, r
        if sign == 1:
            max_cons_win  = max(max_cons_win, cur)
        else:
            max_cons_loss = max(max_cons_loss, cur)

    return {
        "Total Trades":       total,
        "Win Rate":           round(win_rate, 1),
        "Net P&L":            round(net_pnl, 2),
        "Gross Win":          round(gross_win, 2),
        "Gross Loss":         round(gross_loss, 2),
        "Profit Factor":      round(profit_factor, 2),
        "Avg Win ($)":        round(avg_win, 2),
        "Avg Loss ($)":       round(avg_loss, 2),
        "Max Drawdown ($)":   round(max_dd, 2),
        "Calmar Ratio":       round(calmar, 2),
        "Max Cons. Wins":     max_cons_win,
        "Max Cons. Losses":   max_cons_loss,
        "Annual Return ($)":  round(annual_ret, 2),
    }

# ─── CHARTS ───────────────────────────────────────────────────────────────────
def plot_results(trades: list, eq: pd.Series, metrics: dict, cfg: dict):
    df_t = pd.DataFrame(trades)

    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Equity Curve ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#161b22")
    eq_pos = eq.clip(lower=0)
    eq_neg = eq.clip(upper=0)
    ax1.fill_between(eq.index, eq, where=eq >= 0, color="#2ea043", alpha=0.4)
    ax1.fill_between(eq.index, eq, where=eq < 0,  color="#da3633", alpha=0.4)
    ax1.plot(eq.index, eq, color="#58a6ff", linewidth=1.2, label="Equity Curve")

    # Drawdown shading
    rolling_max = eq.cummax()
    drawdown    = eq - rolling_max
    ax1.fill_between(eq.index, rolling_max, eq,
                     where=drawdown < 0, alpha=0.15, color="#da3633",
                     label=f"Drawdown (max ${metrics['Max Drawdown ($)']:,.0f})")

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("Equity Curve — Heiken Ashi MNQ 15-Min Strategy",
                  color="white", fontsize=13, fontweight="bold")
    ax1.set_ylabel("P&L ($)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── 2. Monthly P&L Bar Chart ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor("#161b22")
    if not df_t.empty:
        df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
        monthly = df_t.groupby(df_t["exit_time"].dt.to_period("M"))["pnl"].sum()
        colors  = ["#2ea043" if v >= 0 else "#da3633" for v in monthly.values]
        ax2.bar(monthly.index.astype(str), monthly.values, color=colors, edgecolor="#30363d")
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax2.set_title("Monthly P&L", color="white", fontsize=11, fontweight="bold")
        ax2.set_ylabel("P&L ($)", color="white")
        ax2.tick_params(colors="white", axis="y")
        ax2.tick_params(colors="white", axis="x", rotation=45, labelsize=7)

    # ── 3. Trade P&L Distribution ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor("#161b22")
    if not df_t.empty:
        wins_pnl = df_t[df_t["pnl"] > 0]["pnl"]
        loss_pnl = df_t[df_t["pnl"] < 0]["pnl"]
        ax3.hist(wins_pnl, bins=20, color="#2ea043", alpha=0.7, label="Winners")
        ax3.hist(loss_pnl, bins=20, color="#da3633", alpha=0.7, label="Losers")
        ax3.axvline(0, color="white", linewidth=0.8, linestyle="--")
        ax3.set_title("Trade P&L Distribution", color="white", fontsize=11, fontweight="bold")
        ax3.set_xlabel("P&L ($)", color="white")
        ax3.tick_params(colors="white")
        ax3.legend(facecolor="#161b22", labelcolor="white", fontsize=9)

    # ── 4. Win Rate by Direction ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#161b22")
    if not df_t.empty:
        for i, direction in enumerate(["Long", "Short"]):
            sub = df_t[df_t["direction"] == direction]
            wr  = (sub["pnl"] > 0).mean() * 100 if len(sub) else 0
            ax4.bar(direction, wr, color="#2ea043" if wr >= 50 else "#da3633",
                    width=0.4, edgecolor="#30363d")
            ax4.text(i, wr + 1, f"{wr:.0f}%\n({len(sub)} trades)",
                     ha="center", color="white", fontsize=9)
        ax4.axhline(50, color="gray", linewidth=0.7, linestyle="--", label="50% line")
        ax4.set_ylim(0, 105)
        ax4.set_title("Win Rate by Direction", color="white", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Win Rate %", color="white")
        ax4.tick_params(colors="white")

    # ── 5. Exit Reason Breakdown ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#161b22")
    if not df_t.empty:
        reason_pnl = df_t.groupby("reason")["pnl"].agg(["sum", "count"])
        bar_colors = {
            "TP1":  "#2ea043",
            "TP2":  "#3fb950",
            "SL":   "#da3633",
        }
        for j, (reason, row_r) in enumerate(reason_pnl.iterrows()):
            c = bar_colors.get(reason, "#8b949e")
            ax5.bar(reason, row_r["sum"], color=c, edgecolor="#30363d", width=0.4)
            ax5.text(j, row_r["sum"] + (50 if row_r["sum"] >= 0 else -150),
                     f"n={int(row_r['count'])}", ha="center", color="white", fontsize=9)
        ax5.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax5.set_title("P&L by Exit Reason", color="white", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Total P&L ($)", color="white")
        ax5.tick_params(colors="white")

    # ── 6. Metrics Table ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor("#161b22")
    ax6.axis("off")

    rows = []
    for k, v in metrics.items():
        if isinstance(v, float) and abs(v) > 1000:
            val_str = f"${v:,.0f}"
        elif isinstance(v, float):
            val_str = f"{v}"
        elif k == "Win Rate":
            val_str = f"{v}%"
        else:
            val_str = str(v)
        rows.append([k, val_str])

    tbl = ax6.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r > 0 else "#21262d")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="white")
        if r > 0 and c == 1:
            txt = rows[r - 1][1]
            # Color-code P&L values
            if any(k in rows[r - 1][0] for k in ["P&L", "Win", "Loss", "Drawdown", "Return"]):
                try:
                    num = float(txt.replace("$", "").replace(",", "").replace("%", ""))
                    cell.set_text_props(
                        color="#2ea043" if num > 0 else ("#da3633" if num < 0 else "white")
                    )
                except ValueError:
                    pass

    ax6.set_title("Performance Summary", color="white", fontsize=11, fontweight="bold", pad=10)

    plt.suptitle(
        f"HA MNQ 15-Min Backtest  |  {cfg['period_years']}yr  |  "
        f"SL={cfg['sl_pts']}pts  TP={'2:1' if cfg['use_tp2'] else '1:1'}",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    out_path = "strategies/backtest_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Chart saved → {out_path}")
    return out_path

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Heiken Ashi MNQ 15-Min Strategy — Backtest")
    print("=" * 60)

    df = fetch_data(CONFIG)
    df = add_indicators(df, CONFIG)
    df = add_session_flag(df, CONFIG)

    print("\nRunning backtest ...")
    trades, eq = run_backtest(df, CONFIG)

    if not trades:
        print("No trades generated. Check filters/session settings.")
        return

    metrics = calc_metrics(trades, eq)

    print("\n── Performance Summary ─────────────────────────────────")
    for k, v in metrics.items():
        label = f"  {k:<25}"
        if "P&L" in k or "$" in k or "Drawdown" in k or "Return" in k:
            print(f"{label} ${v:,.2f}")
        elif k == "Win Rate":
            print(f"{label} {v}%")
        else:
            print(f"{label} {v}")
    print("─" * 55)

    df_trades = pd.DataFrame(trades)
    csv_path  = "strategies/backtest_trades.csv"
    df_trades.to_csv(csv_path, index=False)
    print(f"\n  Trade log saved → {csv_path}")

    plot_results(trades, eq, metrics, CONFIG)

    # Print comparison: 1:1 vs 2:1 hint
    print("\n── Mode ────────────────────────────────────────────────")
    mode = "2:1 Scale-Out (50% @ TP1, 50% @ TP2)" if CONFIG["use_tp2"] else "1:1 (100% @ TP1)"
    print(f"  Risk mode : {mode}")
    print(f"  SL        : {CONFIG['sl_pts']} pts  (${CONFIG['sl_pts'] * CONFIG['mnq_pt_value']:.0f}/contract)")
    print(f"  TP1       : {CONFIG['tp1_pts']} pts  (${CONFIG['tp1_pts'] * CONFIG['mnq_pt_value']:.0f}/contract)")
    if CONFIG["use_tp2"]:
        print(f"  TP2       : {CONFIG['tp2_pts']} pts  (${CONFIG['tp2_pts'] * CONFIG['mnq_pt_value']:.0f}/contract)")
    print("\nDone.")


if __name__ == "__main__":
    main()

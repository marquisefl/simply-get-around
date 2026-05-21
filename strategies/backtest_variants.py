"""
Heiken Ashi MNQ 15-Min — Multi-Variant Backtest
================================================
SESSION: 10:15am – 4:00pm ET  (avoids opening-range chop)

STRATEGY CORE LOGIC
-------------------
Heiken Ashi candles smooth OHLC price noise by averaging values across bars.
A green HA candle with NO lower wick = maximum bullish conviction.
A red HA candle with NO upper wick = maximum bearish conviction.

We enter when multiple same-color HA candles confirm a directional move,
then layer trend/momentum filters to keep only the highest-probability setups.

FOUR VARIANTS TESTED
--------------------
A  Base          — 2 HA candles, 15% wick, EMA20 + VWAP + RSI(45-70)
B  Conservative  — 3 HA candles, 0% wick (no counter-wick allowed), tighter RSI
C  Color-Change  — 2 HA candles that must follow a prior opposite-color candle
                   (catches the move early at a fresh reversal), dual EMA filter
D  Strict        — 3 HA candles, 5% wick, dual EMA, VWAP, tight RSI,
                   lunchtime exclusion 11:30–13:00 ET

RISK (all variants)
-------------------
  SL  : 100 pts  ($200/contract)
  TP1 : 100 pts  ($200/contract)  — tested at both 1:1 and 2:1 scale-out
  TP2 : 200 pts  ($400/contract)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# ─── VARIANT DEFINITIONS ──────────────────────────────────────────────────────
# Each dict is merged on top of DEFAULTS before running.

DEFAULTS = {
    "period_years":   2,
    "mnq_pt_value":   2.0,
    "commission":     1.24,        # round-trip
    "contracts":      1,
    "sl_pts":         100.0,
    "tp1_pts":        100.0,
    "use_tp2":        False,
    "tp2_pts":        200.0,
    "tp1_pct":        0.50,
    # session
    "session_start":  "10:15",
    "session_end":    "16:00",
    "lunch_exclude":  False,
    "lunch_start":    "11:30",
    "lunch_end":      "13:00",
    # HA
    "ha_consecutive":      2,
    "wick_filter_pct":     15.0,
    "require_color_change": False,
    # filters
    "use_ema":         True,
    "ema_len":         20,
    "use_ema2":        False,      # dual-EMA trend alignment
    "ema2_len":        50,
    "use_vwap":        True,
    "use_rsi":         True,
    "rsi_len":         14,
    "rsi_long_min":    45,
    "rsi_long_max":    70,
    "rsi_short_min":   30,
    "rsi_short_max":   55,
}

VARIANTS = {
    "A — Base": {
        # Stock settings, just session corrected to 10:15
    },
    "B — Conservative": {
        "ha_consecutive":  3,
        "wick_filter_pct": 0.0,    # zero counter-wick allowed
        "rsi_long_min":    50,
        "rsi_long_max":    68,
        "rsi_short_min":   32,
        "rsi_short_max":   50,
    },
    "C — Color-Change": {
        "ha_consecutive":       2,
        "wick_filter_pct":      10.0,
        "require_color_change": True,   # streak must follow an opposite candle
        "use_ema2":             True,   # EMA20 & EMA50 must both agree
        "rsi_long_min":         48,
        "rsi_long_max":         65,
        "rsi_short_min":        35,
        "rsi_short_max":        52,
    },
    "D — Strict": {
        "ha_consecutive":  3,
        "wick_filter_pct": 5.0,
        "use_ema2":        True,
        "rsi_long_min":    52,
        "rsi_long_max":    68,
        "rsi_short_min":   32,
        "rsi_short_max":   48,
        "lunch_exclude":   True,   # avoid 11:30–13:00 chop
    },
}

# ─── SYNTHETIC DATA ───────────────────────────────────────────────────────────
def generate_synthetic_nq(period_years: int) -> pd.DataFrame:
    """
    Regime-switching GBM calibrated to NQ 15-min statistics:
      Annual vol ~20%, avg 15-min bar range ~42 pts, Student-t fat tails.
    """
    import pytz
    rng = np.random.default_rng(42)

    end   = datetime(2026, 5, 21)
    start = end - timedelta(days=period_years * 365)

    bars_per_day  = 26        # 9:30–16:00 ET
    trading_days  = 252
    bars_per_year = bars_per_day * trading_days   # 6,552

    # timestamp grid
    timestamps = []
    d = start
    session_open  = (9, 30)
    session_close = (16, 0)
    while d <= end:
        if d.weekday() < 5:
            for bar in range(bars_per_day):
                total_min = session_open[0] * 60 + session_open[1] + bar * 15
                h, m = total_min // 60, total_min % 60
                if total_min < session_close[0] * 60 + session_close[1]:
                    timestamps.append(datetime(d.year, d.month, d.day, h, m))
        d += timedelta(days=1)

    n = len(timestamps)

    # regime switching
    regime_len = bars_per_day * 15
    n_reg      = max(1, n // regime_len + 2)
    regimes    = rng.choice(["bull", "bear", "chop"], size=n_reg, p=[0.45, 0.25, 0.30])
    drift_map  = {"bull": 0.18, "bear": -0.20, "chop": 0.02}
    vol_map    = {"bull": 0.16, "bear": 0.24,  "chop": 0.14}

    returns = np.zeros(n)
    for i in range(n):
        reg = regimes[min(i // regime_len, n_reg - 1)]
        mu  = drift_map[reg] / bars_per_year
        sig = vol_map[reg]   / np.sqrt(bars_per_year)
        t   = np.clip(rng.standard_t(df=5), -5, 5)
        returns[i] = mu + sig * t

    start_price = 19_500.0
    closes = start_price * np.cumprod(1 + returns)

    intrabar_vol = 0.0019
    opens = highs = lows = volumes = None
    opens   = np.zeros(n)
    highs   = np.zeros(n)
    lows    = np.zeros(n)
    volumes = np.zeros(n)

    for i in range(n):
        prev_c   = closes[i - 1] if i > 0 else start_price
        px       = closes[i]
        opens[i] = prev_c * (1 + rng.normal(0, 0.0003))
        bar_range = abs(px) * intrabar_vol * (0.4 + rng.exponential(0.6))
        direction = np.sign(px - opens[i]) or 1
        body_hi = max(opens[i], px)
        body_lo = min(opens[i], px)
        highs[i] = body_hi + rng.exponential(bar_range * (0.30 if direction > 0 else 0.12))
        lows[i]  = body_lo - rng.exponential(bar_range * (0.12 if direction > 0 else 0.30))
        highs[i] = max(highs[i], opens[i], px)
        lows[i]  = min(lows[i],  opens[i], px)
        bday = i % bars_per_day
        volumes[i] = rng.integers(400, 2500) * (2.2 if bday < 4 else 1.6 if bday > 22 else 1.0)

    et  = pytz.timezone("America/New_York")
    idx = pd.DatetimeIndex([et.localize(ts) for ts in timestamps])
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def calc_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=length - 1, min_periods=length).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=length - 1, min_periods=length).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    vwap = pd.Series(index=df.index, dtype=float)
    for d in df.index.normalize().unique():
        mask = df.index.normalize() == d
        cum  = (tp[mask] * df.loc[mask, "volume"]).cumsum()
        vwap[mask] = cum / df.loc[mask, "volume"].cumsum().replace(0, np.nan)
    return vwap


def build_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # ── Heiken Ashi ──────────────────────────────────────────────────────────
    ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_o = np.zeros(len(df))
    ha_o[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + ha_c.iloc[i - 1]) / 2

    ha_o_s = pd.Series(ha_o, index=df.index)
    ha_h   = df[["high"]].join(pd.DataFrame({"ha_o": ha_o_s, "ha_c": ha_c})).max(axis=1)
    ha_l   = df[["low"]].join(pd.DataFrame({"ha_o": ha_o_s, "ha_c": ha_c})).min(axis=1)

    bull       = ha_c > ha_o_s
    body       = (ha_c - ha_o_s).abs()
    upper_wick = ha_h - pd.concat([ha_o_s, ha_c], axis=1).max(axis=1)
    lower_wick = pd.concat([ha_o_s, ha_c], axis=1).min(axis=1) - ha_l

    wf = cfg["wick_filter_pct"] / 100.0
    strong_bull = bull & ((body == 0) | (lower_wick / body.replace(0, np.nan) <= wf))
    strong_bear = (~bull) & (ha_c != ha_o_s) & (
        (body == 0) | (upper_wick / body.replace(0, np.nan) <= wf)
    )

    # ── Streak counting ───────────────────────────────────────────────────────
    n = cfg["ha_consecutive"]
    bear_series = (~bull) & (ha_c != ha_o_s)

    def streak(series: pd.Series) -> pd.Series:
        arr = series.values.astype(bool)
        out = np.zeros(len(arr), dtype=int)
        for i in range(n - 1, len(arr)):
            if arr[i - n + 1 : i + 1].all():
                out[i] = 1
        return pd.Series(out, index=series.index)

    bull_streak = streak(bull)
    bear_streak = streak(bear_series)

    # ── Color-change requirement ───────────────────────────────────────────────
    # The streak must have started exactly n candles ago (preceded by opposite color).
    if cfg["require_color_change"]:
        # For longs: bar at position -(n) must be bearish; for shorts: bullish
        def preceded_by_opposite(bull_streak_s, is_bull):
            arr_bull = is_bull.values.astype(bool)
            arr_str  = bull_streak_s.values
            out      = np.zeros(len(arr_str), dtype=int)
            for i in range(n, len(arr_str)):
                if arr_str[i] == 1:
                    # check bar right before the streak began
                    if not arr_bull[i - n]:   # opposite color
                        out[i] = 1
            return pd.Series(out, index=bull_streak_s.index)

        def preceded_by_bull(bear_streak_s, is_bull):
            arr_bull = is_bull.values.astype(bool)
            arr_str  = bear_streak_s.values
            out      = np.zeros(len(arr_str), dtype=int)
            for i in range(n, len(arr_str)):
                if arr_str[i] == 1:
                    if arr_bull[i - n]:
                        out[i] = 1
            return pd.Series(out, index=bear_streak_s.index)

        bull_ok = preceded_by_opposite(bull_streak, bull)
        bear_ok = preceded_by_bull(bear_streak, bull)
    else:
        bull_ok = bull_streak
        bear_ok = bear_streak

    # ── Trend indicators ──────────────────────────────────────────────────────
    ema1 = df["close"].ewm(span=cfg["ema_len"], adjust=False).mean()
    ema2 = df["close"].ewm(span=cfg["ema2_len"], adjust=False).mean()
    rsi  = calc_rsi(df["close"], cfg["rsi_len"])
    vwap = calc_vwap(df)

    # ── Session mask ──────────────────────────────────────────────────────────
    t = df.index.strftime("%H:%M")
    in_session = (t >= cfg["session_start"]) & (t < cfg["session_end"])
    if cfg["lunch_exclude"]:
        in_session &= ~((t >= cfg["lunch_start"]) & (t < cfg["lunch_end"]))

    # ── Entry signals ─────────────────────────────────────────────────────────
    ema_long  = df["close"] > ema1
    ema_short = df["close"] < ema1
    if cfg["use_ema2"]:
        ema_long  &= df["close"] > ema2
        ema_short &= df["close"] < ema2

    long_sig = (
        (bull_ok == 1) & strong_bull
        & (ema_long  if cfg["use_ema"]  else True)
        & ((df["close"] > vwap)   if cfg["use_vwap"] else True)
        & (((rsi >= cfg["rsi_long_min"]) & (rsi <= cfg["rsi_long_max"])) if cfg["use_rsi"] else True)
        & in_session
    )
    short_sig = (
        (bear_ok == 1) & strong_bear
        & (ema_short  if cfg["use_ema"]  else True)
        & ((df["close"] < vwap)   if cfg["use_vwap"] else True)
        & (((rsi >= cfg["rsi_short_min"]) & (rsi <= cfg["rsi_short_max"])) if cfg["use_rsi"] else True)
        & in_session
    )

    return long_sig, short_sig

# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, long_sig: pd.Series, short_sig: pd.Series,
                 cfg: dict) -> tuple[list, pd.Series]:
    pt   = cfg["mnq_pt_value"] * cfg["contracts"]
    comm = cfg["commission"]
    sl   = cfg["sl_pts"]
    tp1  = cfg["tp1_pts"]
    tp2  = cfg["tp2_pts"]
    tp2_on = cfg["use_tp2"]
    tp1_p  = cfg["tp1_pct"]

    trades     = []
    equity     = 0.0
    eq_curve   = []
    position   = 0
    entry_px   = 0.0
    entry_idx  = None
    tp1_filled = False

    longs  = long_sig.values
    shorts = short_sig.values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    idx    = df.index

    def log(exit_px, reason, qty_pct):
        nonlocal equity
        pnl = (exit_px - entry_px) * position * pt * qty_pct - comm * qty_pct
        equity += pnl
        trades.append({
            "entry_time": entry_idx,
            "exit_time":  idx[i],
            "direction":  "Long" if position == 1 else "Short",
            "entry":      round(entry_px, 2),
            "exit":       round(exit_px, 2),
            "qty_pct":    qty_pct,
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
                sl_hit = lo <= sl_px
                if not tp2_on:
                    if sl_hit:
                        log(sl_px, "SL", 1.0);  position, tp1_filled = 0, False
                    elif hi >= tp1_px:
                        log(tp1_px, "TP1", 1.0); position, tp1_filled = 0, False
                else:
                    if sl_hit:
                        rem = 1.0 - tp1_p if tp1_filled else 1.0
                        log(sl_px, "SL", rem);   position, tp1_filled = 0, False
                    elif not tp1_filled and hi >= tp1_px:
                        log(tp1_px, "TP1", tp1_p); tp1_filled = True
                    elif tp1_filled and hi >= tp2_px:
                        log(tp2_px, "TP2", 1.0 - tp1_p); position, tp1_filled = 0, False
            else:
                sl_hit = hi >= sl_px
                if not tp2_on:
                    if sl_hit:
                        log(sl_px, "SL", 1.0);  position, tp1_filled = 0, False
                    elif lo <= tp1_px:
                        log(tp1_px, "TP1", 1.0); position, tp1_filled = 0, False
                else:
                    if sl_hit:
                        rem = 1.0 - tp1_p if tp1_filled else 1.0
                        log(sl_px, "SL", rem);   position, tp1_filled = 0, False
                    elif not tp1_filled and lo <= tp1_px:
                        log(tp1_px, "TP1", tp1_p); tp1_filled = True
                    elif tp1_filled and lo <= tp2_px:
                        log(tp2_px, "TP2", 1.0 - tp1_p); position, tp1_filled = 0, False

        if position == 0:
            if longs[i]:
                position, entry_px, entry_idx, tp1_filled = 1, closes[i], idx[i], False
            elif shorts[i]:
                position, entry_px, entry_idx, tp1_filled = -1, closes[i], idx[i], False

        eq_curve.append(equity)

    eq = pd.Series(eq_curve, index=df.index)
    return trades, eq

# ─── METRICS ──────────────────────────────────────────────────────────────────
def metrics(trades: list, eq: pd.Series) -> dict:
    if not trades:
        return {k: 0 for k in ["n","wr","net","pf","avg_w","avg_l","mdd","cons_w","cons_l"]}

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["pnl"] > 0]["pnl"]
    loss = df_t[df_t["pnl"] < 0]["pnl"]
    n    = len(df_t)
    wr   = len(wins) / n * 100 if n else 0
    pf   = wins.sum() / abs(loss.sum()) if loss.sum() != 0 else np.inf
    mdd  = (eq - eq.cummax()).min()

    # consecutive runs
    res = (df_t["pnl"] > 0).astype(int).values
    cw = cl = cur = 0; sign = None
    for r in res:
        cur = cur + 1 if r == sign else 1; sign = r
        cw = max(cw, cur) if sign == 1 else cw
        cl = max(cl, cur) if sign == 0 else cl

    return {
        "n":      n,
        "wr":     round(wr, 1),
        "net":    round(df_t["pnl"].sum(), 2),
        "pf":     round(pf, 2),
        "avg_w":  round(wins.mean() if len(wins) else 0, 2),
        "avg_l":  round(loss.mean() if len(loss) else 0, 2),
        "mdd":    round(mdd, 2),
        "cons_w": cw,
        "cons_l": cl,
    }

# ─── COMPARISON CHART ─────────────────────────────────────────────────────────
def plot_comparison(results: dict, tp_label: str):
    sns.set_theme(style="darkgrid")
    BG   = "#0d1117"
    CARD = "#161b22"
    GRAY = "#30363d"
    GREEN = "#2ea043"
    RED   = "#da3633"
    BLUE  = "#58a6ff"

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BG)

    n_variants = len(results)
    gs = gridspec.GridSpec(3, n_variants, figure=fig, hspace=0.55, wspace=0.35)

    variant_names = list(results.keys())
    colors_eq     = [BLUE, "#f78166", "#3fb950", "#d2a8ff"]

    # ── Row 0: Equity curves side-by-side ────────────────────────────────────
    ax_eq_main = fig.add_subplot(gs[0, :])
    ax_eq_main.set_facecolor(CARD)
    for j, (name, res) in enumerate(results.items()):
        eq = res["eq"]
        ax_eq_main.plot(eq.index, eq, linewidth=1.5,
                        color=colors_eq[j], label=f"{name}  (net ${res['m']['net']:,.0f})")
    ax_eq_main.axhline(0, color=GRAY, linewidth=0.7, linestyle="--")
    ax_eq_main.set_title(
        f"Equity Curves — All Variants  |  Session 10:15–16:00 ET  |  {tp_label}",
        color="white", fontsize=13, fontweight="bold")
    ax_eq_main.set_ylabel("Cumulative P&L ($)", color="white")
    ax_eq_main.tick_params(colors="white")
    ax_eq_main.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_eq_main.legend(facecolor=CARD, labelcolor="white", fontsize=9,
                      loc="upper left", framealpha=0.8)

    # ── Row 1: Monthly bars per variant ──────────────────────────────────────
    for j, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, j])
        ax.set_facecolor(CARD)
        df_t = pd.DataFrame(res["trades"])
        if df_t.empty:
            ax.set_title(name, color="white", fontsize=9)
            continue
        df_t["exit_time"] = pd.to_datetime(df_t["exit_time"])
        monthly = df_t.groupby(df_t["exit_time"].dt.to_period("M"))["pnl"].sum()
        bar_cols = [GREEN if v >= 0 else RED for v in monthly.values]
        ax.bar(monthly.index.astype(str), monthly.values, color=bar_cols, edgecolor=GRAY)
        ax.axhline(0, color="white", linewidth=0.4, linestyle="--")
        ax.set_title(f"{name}\nMonthly P&L", color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="white", axis="y", labelsize=7)
        ax.tick_params(colors="white", axis="x", rotation=55, labelsize=6)

    # ── Row 2: KPI scorecards ─────────────────────────────────────────────────
    for j, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[2, j])
        ax.set_facecolor(CARD)
        ax.axis("off")
        m = res["m"]

        rows = [
            ("Trades",     str(m["n"])),
            ("Win Rate",   f"{m['wr']}%"),
            ("Net P&L",    f"${m['net']:,.0f}"),
            ("Prof Factor",f"{m['pf']:.2f}"),
            ("Avg Win",    f"${m['avg_w']:.0f}"),
            ("Avg Loss",   f"${m['avg_l']:.0f}"),
            ("Max DD",     f"${m['mdd']:,.0f}"),
            ("Con. Wins",  str(m["cons_w"])),
            ("Con. Loss",  str(m["cons_l"])),
        ]

        tbl = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(CARD if r > 0 else "#21262d")
            cell.set_edgecolor(GRAY)
            cell.set_text_props(color="white")
            if r > 0 and c == 1:
                val_str = rows[r - 1][1]
                key     = rows[r - 1][0]
                if key in ("Net P&L", "Max DD", "Avg Win", "Avg Loss"):
                    try:
                        num = float(val_str.replace("$", "").replace(",", ""))
                        cell.set_text_props(color=GREEN if num > 0 else (RED if num < 0 else "white"))
                    except ValueError:
                        pass
                elif key == "Win Rate":
                    try:
                        num = float(val_str.replace("%", ""))
                        cell.set_text_props(color=GREEN if num >= 50 else RED)
                    except ValueError:
                        pass

        ax.set_title(f"{name}", color="white", fontsize=9, fontweight="bold", pad=8)

    out = f"strategies/variant_comparison_{tp_label.replace(':', '').replace(' ', '_').lower()}.png"
    plt.suptitle(
        f"HA MNQ 15-Min  |  SL=100pts  |  {tp_label}  |  10:15–16:00 ET  |  2yr Backtest",
        color="white", fontsize=13, fontweight="bold", y=0.99)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart → {out}")
    return out

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Heiken Ashi MNQ 15-Min — Multi-Variant Backtest")
    print("=" * 70)
    print()

    # Strategy description
    print("STRATEGY LOGIC")
    print("─" * 70)
    print("  Instrument : MNQ (Micro E-mini NASDAQ-100), $2/pt per contract")
    print("  Timeframe  : 15-minute bars")
    print("  Session    : 10:15 AM – 4:00 PM ET (avoids opening range chaos)")
    print()
    print("  LONG entry (all must be true):")
    print("    1. N consecutive bullish HA candles (HA Close > HA Open)")
    print("    2. Current HA candle has minimal/no lower wick (≤ X% of body)")
    print("    3. Price above EMA(20)  [+ EMA(50) for dual-EMA variants]")
    print("    4. Price above VWAP")
    print("    5. RSI in momentum window (not overbought)")
    print("    6. Session window active")
    print()
    print("  SHORT entry: mirror of above (red HA candles, no upper wick,")
    print("    price below EMA/VWAP, RSI in bearish momentum zone)")
    print()
    print("  RISK: 100pt SL ($200) | TP1 100pt ($200) | TP2 200pt ($400)")
    print("─" * 70)
    print()

    # Generate data once (shared across all variants)
    print("Generating synthetic NQ 15-min data (2 years) ...")
    df = generate_synthetic_nq(2)
    avg_range = (df["high"] - df["low"]).mean()
    print(f"  {len(df):,} bars | price {df['close'].min():.0f}–{df['close'].max():.0f}"
          f" | avg bar range {avg_range:.1f} pts\n")

    # Run all variants for 1:1 and 2:1
    for tp_mode, use_tp2 in [("1:1", False), ("2:1 Scale-Out", True)]:
        print(f"{'='*70}")
        print(f"  MODE: {tp_mode}  ({'50% @ TP1, 50% @ TP2' if use_tp2 else '100% @ TP1'})")
        print(f"{'='*70}")

        all_results = {}

        for name, overrides in VARIANTS.items():
            cfg = {**DEFAULTS, **overrides, "use_tp2": use_tp2}

            long_sig, short_sig = build_signals(df, cfg)
            trades, eq = run_backtest(df, long_sig, short_sig, cfg)
            m = metrics(trades, eq)

            all_results[name] = {"trades": trades, "eq": eq, "m": m}

            # Console summary
            wr_indicator = "✓" if m["wr"] >= 52 else ("~" if m["wr"] >= 48 else "✗")
            pf_indicator = "✓" if m["pf"] >= 1.1 else ("~" if m["pf"] >= 0.95 else "✗")
            print(f"\n  {name}")
            print(f"    Trades: {m['n']:>5}   Win Rate: {m['wr']:>5.1f}% {wr_indicator}"
                  f"   Net P&L: ${m['net']:>8,.0f}"
                  f"   PF: {m['pf']:.2f} {pf_indicator}")
            print(f"    Avg W: ${m['avg_w']:>6.0f}   Avg L: ${m['avg_l']:>7.0f}"
                  f"   Max DD: ${m['mdd']:>8,.0f}"
                  f"   Con W/L: {m['cons_w']}/{m['cons_l']}")

        print()
        plot_comparison(all_results, tp_mode)

    print("\nDone.")

if __name__ == "__main__":
    main()

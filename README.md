# Heiken Ashi MNQ 15-Min Strategy

A momentum-based trading strategy for **Micro E-mini NASDAQ-100 (MNQ)** futures on the **15-minute timeframe**, using Heiken Ashi candles as the primary signal with trend and momentum filters.

---

## Strategy Overview

| Parameter | Value |
|---|---|
| Instrument | MNQ (Micro E-mini NASDAQ-100) |
| Timeframe | 15-minute |
| Stop Loss | 100 points ($200/contract) |
| Take Profit 1 (1:1) | 100 points ($200/contract) |
| Take Profit 2 (2:1) | 200 points ($400/contract) |
| Risk per Trade | ~$200/contract + commissions |

---

## Why Heiken Ashi on MNQ?

Heiken Ashi candles smooth out price noise by averaging OHLC values. This makes trending moves on NQ/MNQ much cleaner to read than raw candlesticks — the candles stay green during an uptrend and red during a downtrend without the constant color flipping you see with standard candles. On the 15-min MNQ chart, this removes a lot of the chop and helps identify genuine momentum moves.

---

## Entry Rules

### Long Entry (all must be true)
1. **2+ consecutive bullish HA candles** — color confirmation of trending momentum
2. **Current HA candle has no (or minimal) lower wick** — counter-wick ≤ 15% of body; a solid green HA candle with no lower shadow is the strongest long signal
3. **Price above EMA(20)** — ensures you're trading with the short-term trend
4. **Price above VWAP** — institutional bias is to the long side
5. **RSI between 45–70** — in momentum territory but not overbought
6. **NY Session active** — 9:30–16:00 ET (or Kill Zones: 9:30–11:00, 13:30–15:00)

### Short Entry (all must be true)
1. **2+ consecutive bearish HA candles**
2. **Current HA candle has no (or minimal) upper wick** — counter-wick ≤ 15% of body
3. **Price below EMA(20)**
4. **Price below VWAP**
5. **RSI between 30–55** — bearish momentum, not oversold
6. **NY Session active**

---

## Exit Rules

### 1:1 Mode (default)
- **Stop Loss:** 100 points below/above entry
- **Take Profit:** 100 points in favor of trade

### 2:1 Scale-Out Mode (optional)
- Exit **50% of position at TP1** (100 pts) — locks in profit and reduces risk
- Exit **remaining 50% at TP2** (200 pts) — lets a portion run for the full move
- Stop Loss remains at 100 pts for both portions

---

## MNQ Dollar Values

| Points | $ per Contract | Notes |
|---|---|---|
| 25 | $50 | ~1 ATR on slow 15-min bar |
| 50 | $100 | Typical intraday scalp target |
| 100 | $200 | SL / TP1 target |
| 200 | $400 | TP2 target (2:1) |

MNQ point value: **$2.00 per point per contract**

---

## Recommended TradingView Setup

1. Open **MNQ1!** chart on the **15-minute** timeframe
2. Set chart timezone to **America/New_York**
3. Open Pine Script Editor → paste `strategies/heiken_ashi_mnq_15min.pine`
4. Click **Add to chart**
5. In Strategy Tester → Settings:
   - Initial Capital: `$10,000`
   - Order Size: `1 contract`
   - Commission: `$0.62/side` (typical MNQ rate)

---

## Key Settings to Tune

| Setting | Default | Notes |
|---|---|---|
| Consecutive HA candles | 2 | Raise to 3 for fewer but higher-quality entries |
| Max counter-wick % | 15% | Lower = stricter, higher quality signals |
| EMA length | 20 | Try 50 for longer-term trend alignment |
| RSI Long range | 45–70 | Keeps entries in momentum, avoids overbought |
| Session filter | NY only | Kill zones only cuts trades to highest-probability windows |
| Enable 2:1 scale-out | OFF | Turn ON to let winners run while securing profit |

---

## Strategy Logic: How HA Signals Work

```
Standard Candle:          Heiken Ashi Candle:
  Open = actual open        HA-Open  = (prev HA-Open + prev HA-Close) / 2
  High = actual high        HA-High  = max(High, HA-Open, HA-Close)
  Low  = actual low         HA-Low   = min(Low,  HA-Open, HA-Close)
  Close = actual close      HA-Close = (Open + High + Low + Close) / 4
```

**Strongest signals:**
- Green HA candle with **no lower wick** → maximum bullish conviction
- Red HA candle with **no upper wick** → maximum bearish conviction
- First appearance of these after a trend change = highest R:R entry

---

## Best Market Conditions

**Works well in:**
- Trending NY sessions (consistent directional move)
- Post-news continuation (FOMC, CPI — after the initial spike settles)
- Opening range breakouts (9:30–10:30 ET moves)

**Avoid:**
- Choppy/low-volatility days (VIX < 12)
- Around major economic releases (wait for the 15-min bar to close after the news)
- Overnight/Asian session (low liquidity, more false signals)

---

## Files

```
strategies/
└── heiken_ashi_mnq_15min.pine    ← TradingView Pine Script v5
README.md                          ← This file
```

---

## Risk Disclaimer

This strategy is for educational and research purposes. Past performance in backtests does not guarantee future results. Always backtest thoroughly before live trading. Futures trading involves substantial risk of loss.

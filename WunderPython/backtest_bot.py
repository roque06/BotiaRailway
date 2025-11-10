# AI_MODEL/backtest_bot.py
# Backtest "v6-like": 15m base + 5m confirm + 1h macro, IA/ML, BE, trailing y SL por ATR

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # para importar el bot

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importa funciones/constantes del bot v6
from smart_trading_bot import (
    compute_indicators, detect_regime, ml_score, ia_decision, fetch_klines,
    EMA_DIFF_MARGIN, ATR_SL_MULT, BREAKEVEN_TRIGGER, BREAKEVEN_OFFSET,
    TRAIL_BASE_PCT, TRAIL_TIGHT_AT, TRAIL_VALUES, ML_THRESHOLD,
    VOLATILITY_MULT_LIMIT, ADX_MIN, RSI_LONG_MIN, RSI_LONG_MAX,
    RSI_SHORT_MIN, RSI_SHORT_MAX
)

# ---------- utilidades de gestión (idénticas en espíritu al bot) ----------

def current_profit_pct(side: str, entry: float, price: float) -> float:
    if side == "LONG":
        return (price - entry) / entry * 100.0
    return (entry - price) / entry * 100.0

def pick_dynamic_trailing(base_pct: float, profit_pct: float) -> float:
    t = base_pct
    for trigger, tight in zip(TRAIL_TIGHT_AT, TRAIL_VALUES):
        if profit_pct >= trigger:
            t = min(t, tight)
    return t

# ---------- parámetros de backtest ----------
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
CONFIRM_INTERVAL = "5m"
MACRO_INTERVAL = "1h"
LIMIT_15M = 5000          # ~ 30-40 días
COOLDOWN_AFTER_EXIT_SEC = 0  # sin cooldown en backtest
USE_STRICT_FILTERS = False # si False, relaja para ver más operaciones

# ---------- descarga y preparación multi-TF ----------
df15 = compute_indicators(fetch_klines(SYMBOL, INTERVAL, LIMIT_15M).copy())
df5  = compute_indicators(fetch_klines(SYMBOL, CONFIRM_INTERVAL, 5 * LIMIT_15M).copy())
df1h = compute_indicators(fetch_klines(SYMBOL, MACRO_INTERVAL, LIMIT_15M // 4).copy())

# Alinea por close_time
df15 = df15.rename(columns={"close_time": "t"}).set_index("t")
df5  = df5.rename(columns={"close_time": "t"})[["t","ema_fast","ema_slow","rsi","rsi_slope"]]
df1h = df1h.rename(columns={"close_time": "t"})[["t","ema_fast","ema_slow","adx"]]

# usa merge_asof (busca la barra más reciente <= t)
df15 = df15.sort_index()
df5  = df5.sort_values("t")
df1h = df1h.sort_values("t")

df15 = pd.merge_asof(
    df15.reset_index(), df5, on="t", direction="backward", suffixes=("", "_5m")
)
df15 = pd.merge_asof(
    df15, df1h, on="t", direction="backward", suffixes=("", "_1h")
).set_index("t")

# ---------- la simulación bar-a-bar ----------
trades = []
entries = []
state = {
    "side": None,
    "entry": None,
    "sl": None,
    "trail": None,
    "be_on": False,
    "highest": None,
    "lowest": None,
}

def maybe_exit(price: float):
    """Comprueba si golpea trailing/SL o si tras BE cae demasiado."""
    if state["side"] == "LONG":
        if state["trail"] is not None and price <= state["trail"]:
            return True, f"Trailing hit {state['trail']:.2f}"
        if state["sl"] is not None and price <= state["sl"]:
            return True, f"SL hit {state['sl']:.2f}"
    else:
        if state["trail"] is not None and price >= state["trail"]:
            return True, f"Trailing hit {state['trail']:.2f}"
        if state["sl"] is not None and price >= state["sl"]:
            return True, f"SL hit {state['sl']:.2f}"
    return False, ""

def update_trailing(price: float, atr: float):
    if state["side"] is None: return
    entry = state["entry"]
    profit = current_profit_pct(state["side"], entry, price)

    # activa BE
    if (not state["be_on"]) and profit >= BREAKEVEN_TRIGGER:
        be_price = entry * (1 + BREAKEVEN_OFFSET/100.0) if state["side"] == "LONG" \
                   else entry * (1 - BREAKEVEN_OFFSET/100.0)
        state["sl"] = be_price
        state["be_on"] = True

    # trailing dinámico sobre máximos/mínimos a favor
    if state["side"] == "LONG":
        state["highest"] = price if state["highest"] is None else max(state["highest"], price)
        dyn_trail_pct = pick_dynamic_trailing(TRAIL_BASE_PCT, profit)
        new_trail = state["highest"] * (1 - dyn_trail_pct/100.0)
        if state["trail"] is None or new_trail > state["trail"]:
            state["trail"] = new_trail
        if state["be_on"] and state["sl"] is not None:
            state["sl"] = max(state["sl"], state["trail"])
    else:
        state["lowest"] = price if state["lowest"] is None else min(state["lowest"], price)
        dyn_trail_pct = pick_dynamic_trailing(TRAIL_BASE_PCT, profit)
        new_trail = state["lowest"] * (1 + dyn_trail_pct/100.0)
        if state["trail"] is None or new_trail < state["trail"]:
            state["trail"] = new_trail
        if state["be_on"] and state["sl"] is not None:
            state["sl"] = min(state["sl"], state["trail"])

# índice mínimo para tener EMAs/ATR/ADX listos
start_idx = max(220,  # warmup EMA200
                50)   # RSI/ATR

for t, row in df15.iloc[start_idx:].iterrows():
    price = float(row["close"])
    ema_f, ema_s, ema_long = float(row["ema_fast"]), float(row["ema_slow"]), float(row["ema_long"])
    rsi = float(row["rsi"])
    rsi_slope_now = float(row["rsi_slope"])
    adx_now = float(row["adx"])
    atr_fast = float(row["atr"])
    atr_stable = float(row["atr_ma"])
    atr_p90 = row.get("atr_p90", np.nan)
    if pd.isna(atr_p90): atr_p90 = None
    vol_now = float(row["volume"])
    vol_ma = float(row["vol_ma"]) if not pd.isna(row["vol_ma"]) else 0.0

    # 5m
    ema_f_5, ema_s_5 = float(row["ema_fast_5m"]), float(row["ema_slow_5m"])
    rsi_5, rsi_slope_5 = float(row["rsi_5m"]), float(row["rsi_slope_5m"])
    # 1h
    ema_f_1h, ema_s_1h = float(row["ema_fast_1h"]), float(row["ema_slow_1h"])
    adx_1h = float(row["adx_1h"]) if not pd.isna(row["adx_1h"]) else 20.0
    macro_ok = (ema_f_1h > ema_s_1h) and (adx_1h >= 20)

    # régimen
    # usamos el detect_regime con DF 15m hasta ese punto para coherencia
    regime = detect_regime(df15.loc[:t].rename_axis("t").reset_index())

    # IA
    ia_prob = ia_decision(
        ema_f, ema_s, ema_long, rsi, atr_fast,
        rsi_slope_now=rsi_slope_now,
        atr_stable=atr_stable,
        vol_now=vol_now, vol_ma=max(vol_ma,1e-9),
        ema_fast_ref=ema_f, ema_slow_ref=ema_s
    )

    # Señales base (idénticas al bot)
    ema_cross_up_now = ema_f > ema_s * (1 + EMA_DIFF_MARGIN)
    ema_cross_dn_now = ema_f < ema_s * (1 - EMA_DIFF_MARGIN)
    long_align_5m  = (ema_f_5 > ema_s_5) and (rsi_slope_5 > 0) and (rsi_5 >= 40)
    short_align_5m = (ema_f_5 < ema_s_5) and (rsi_slope_5 < 0) and (rsi_5 <= 60)

    bullish_ok = (
        ema_cross_up_now
        and (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX)
        and (rsi_slope_now > 0)
        and (price > ema_long)
        and (vol_now > max(vol_ma,1e-9))
        and long_align_5m
    )
    bearish_ok = (
        ema_cross_dn_now
        and (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX)
        and (rsi_slope_now < 0)
        and (price < ema_long)
        and (vol_now > max(vol_ma,1e-9))
        and short_align_5m
    )

    # IA/Sentimiento/Macro (estricto/relajado)
    if USE_STRICT_FILTERS:
        if ia_prob < 0.40:
            bullish_ok = bearish_ok = False
        if not macro_ok:
            bullish_ok = bearish_ok = False

    # Riesgo/volatilidad
    if USE_STRICT_FILTERS:
        if (atr_p90 is not None and atr_fast > atr_p90) or (atr_fast > atr_stable * VOLATILITY_MULT_LIMIT) or (adx_now < ADX_MIN):
            bullish_ok = bearish_ok = False

    # ML-lite
    features = {
        "rsi": rsi,
        "rsi_slope": rsi_slope_now,
        "adx": adx_now,
        "ema_trend": ema_f > ema_s,
        "vol_rel": (vol_now / max(vol_ma, 1e-9)) - 1.0,
        "regime": regime,
    }
    score = ml_score(features)
    if score < ML_THRESHOLD:
        bullish_ok = bearish_ok = False

    # ===== gestión de posición existente =====
    if state["side"] is not None:
        update_trailing(price, atr_fast)
        do_exit, reason = maybe_exit(price)
        if do_exit:
            pnl = current_profit_pct(state["side"], state["entry"], price)
            trades.append(pnl)
            entries.append((t, state["side"], state["entry"], price, pnl, reason))
            state = {"side": None, "entry": None, "sl": None, "trail": None,
                     "be_on": False, "highest": None, "lowest": None}
            continue  # pasa a la siguiente barra

    # ===== nueva entrada =====
    if state["side"] is None and (bullish_ok or bearish_ok):
        side = "LONG" if bullish_ok else "SHORT"
        entry = price
        sl = entry - ATR_SL_MULT * atr_fast if side == "LONG" else entry + ATR_SL_MULT * atr_fast
        state.update({
            "side": side, "entry": entry, "sl": sl, "trail": None,
            "be_on": False, "highest": None, "lowest": None
        })
        continue

# Cierra posición si quedó abierta al final (mark-to-market último precio)
if state["side"] is not None:
    last_price = float(df15["close"].iloc[-1])
    pnl = current_profit_pct(state["side"], state["entry"], last_price)
    trades.append(pnl)
    entries.append((df15.index[-1], state["side"], state["entry"], last_price, pnl, "EoP"))

# ---------- resultados ----------
trades_arr = np.array(trades, dtype=float)
wins = trades_arr[trades_arr > 0].sum() if trades else 0.0
loss = -trades_arr[trades_arr < 0].sum() if trades else 0.0
pf = (wins / loss) if loss > 0 else float("inf")
winrate = (trades_arr > 0).mean()*100 if len(trades_arr) else float("nan")
total = trades_arr.sum() if len(trades_arr) else 0.0

print(f"Trades simulados: {len(trades)}")
print(f"Winrate: {winrate:.2f}%")
print(f"Profit Factor: {pf:.2f}")
print(f"Ganancia total: {total:.2f}%")

# curva de equity (1 = sin apalancamiento)
equity = np.cumprod(1 + trades_arr/100.0) if len(trades_arr) else np.array([1.0])
plt.plot(equity)
plt.title("Curva de equity simulada")
plt.xlabel("Trade #")
plt.ylabel("Crecimiento (x)")
plt.grid(True)
plt.show()

# guarda CSV de resultados
res_df = pd.DataFrame(entries, columns=["time","side","entry","exit","profit_pct","reason"])
out_path = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
res_df.to_csv(out_path, index=False)
print(f"Resultados guardados en: {out_path}")

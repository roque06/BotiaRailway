import pandas as pd
import numpy as np
import requests
import ta
from datetime import datetime, timedelta

# ====================================
# CONFIGURACIÓN
# ====================================
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 1000  # velas (15m = 10 días aprox)
OUTPUT_FILE = "AI_MODEL/trades_extra.csv"

# Parámetros estrategia
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 40, 70
RSI_SHORT_MIN, RSI_SHORT_MAX = 30, 60
EMA_DIFF_MARGIN = 0.0008


# ====================================
# DESCARGA DE DATOS HISTÓRICOS
# ====================================
def fetch_binance_klines(symbol, interval, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "num_trades",
            "tbbav",
            "tbqav",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


# ====================================
# GENERACIÓN DE SEÑALES
# ====================================
def simulate_trades(df):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], EMA_SLOW)
    df["rsi"] = ta.momentum.rsi(df["close"], RSI_PERIOD)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    trades = []
    position = None
    entry_price = 0

    for i in range(2, len(df)):
        ema_f, ema_s, rsi, adx = df.loc[i, ["ema_fast", "ema_slow", "rsi", "adx"]]
        price = df.loc[i, "close"]

        # Entradas
        if position is None:
            if (
                (ema_f > ema_s * (1 + EMA_DIFF_MARGIN))
                and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX
                and adx > 20
            ):
                position = "LONG"
                entry_price = price
            elif (
                (ema_f < ema_s * (1 - EMA_DIFF_MARGIN))
                and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
                and adx > 20
            ):
                position = "SHORT"
                entry_price = price
        # Salidas
        elif position == "LONG":
            if rsi > 70 or ema_f < ema_s or adx < 15:
                profit = (price - entry_price) / entry_price
                trades.append(
                    {
                        "symbol": SYMBOL,
                        "position": "LONG",
                        "price_entry": entry_price,
                        "price_exit": price,
                        "profit": profit,
                    }
                )
                position = None
        elif position == "SHORT":
            if rsi < 30 or ema_f > ema_s or adx < 15:
                profit = (entry_price - price) / entry_price
                trades.append(
                    {
                        "symbol": SYMBOL,
                        "position": "SHORT",
                        "price_entry": entry_price,
                        "price_exit": price,
                        "profit": profit,
                    }
                )
                position = None

    return pd.DataFrame(trades)


# ====================================
# EJECUCIÓN PRINCIPAL
# ====================================
if __name__ == "__main__":
    print(f"📥 Descargando datos históricos de {SYMBOL}...")
    df = fetch_binance_klines(SYMBOL, INTERVAL, LIMIT)
    print(f"✅ {len(df)} velas descargadas.")

    print("🧮 Simulando operaciones...")
    trades_df = simulate_trades(df)

    if len(trades_df) == 0:
        print("⚠️ No se generaron operaciones con los parámetros actuales.")
    else:
        trades_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ {len(trades_df)} operaciones simuladas guardadas en {OUTPUT_FILE}.")
        print(trades_df.head(10))

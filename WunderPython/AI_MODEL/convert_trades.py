import pandas as pd
import os

# ========================================
# 1️⃣ CONVERTIR TRADES REALES DEL BOT
# ========================================
rows = []
if os.path.exists("trades_log.csv"):
    with open("trades_log.csv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")

            # Ignorar encabezado
            if line.startswith("timestamp") or line.startswith("symbol"):
                continue

            # Caso 1: formato tipo BTCUSDT,ENTER-SHORT,103344.73,31.84...
            if parts[0] == "BTCUSDT":
                if len(parts) >= 3:
                    symbol = parts[0]
                    event = parts[1]
                    price = float(parts[2])
                    rows.append({"symbol": symbol, "event": event, "price": price})

            # Caso 2: formato tipo 2025-11-05T01...,BTCUSDT,SHORT,100426.95
            elif len(parts) >= 4 and parts[1] == "BTCUSDT":
                timestamp = parts[0]
                symbol = parts[1]
                event = parts[2]
                price = float(parts[3])
                rows.append(
                    {
                        "symbol": symbol,
                        "event": event,
                        "price": price,
                        "timestamp": timestamp,
                    }
                )
else:
    print("⚠️ No se encontró trades_log.csv — no hay datos reales aún.")

# Convertir a DataFrame
df = pd.DataFrame(rows)

# Filtrar eventos válidos
if not df.empty:
    df = df[df["event"].str.contains("ENTER|EXIT", na=False)]
else:
    df = pd.DataFrame(columns=["symbol", "event", "price", "timestamp"])

trades = []
current_trade = None

for _, row in df.iterrows():
    if "ENTER" in row["event"]:
        current_trade = {
            "symbol": row["symbol"],
            "position": "LONG" if "LONG" in row["event"] else "SHORT",
            "price_entry": row["price"],
            "timestamp_entry": row.get("timestamp", ""),
        }
    elif "EXIT" in row["event"] and current_trade:
        current_trade["price_exit"] = row["price"]
        current_trade["timestamp_exit"] = row.get("timestamp", "")

        # Calcular profit relativo
        entry = current_trade["price_entry"]
        exit = current_trade["price_exit"]
        if current_trade["position"] == "LONG":
            profit = (exit - entry) / entry
        else:
            profit = (entry - exit) / entry

        current_trade["profit"] = profit
        trades.append(current_trade)
        current_trade = None

# Crear dataset base
df_clean = pd.DataFrame(trades)

# ========================================
# 2️⃣ AGREGAR DATA EXTRA (SIMULADA)
# ========================================
extra_path = "AI_MODEL/trades_extra.csv"
if os.path.exists(extra_path):
    extra_df = pd.read_csv(extra_path)
    if "profit" in extra_df.columns:
        before = len(df_clean)
        df_clean = pd.concat([df_clean, extra_df], ignore_index=True)
        print(f"📈 Agregados {len(extra_df)} trades simulados de entrenamiento extra.")
        print(f"🔹 Total final: {len(df_clean)} operaciones (antes {before}).")
    else:
        print("⚠️ Archivo trades_extra.csv no tiene columna 'profit'. Revisa formato.")
else:
    print("ℹ️ No se encontró trades_extra.csv, solo se usarán operaciones reales.")

# ========================================
# 3️⃣ GUARDAR RESULTADO FINAL
# ========================================
df_clean.to_csv("AI_MODEL/trades_clean.csv", index=False)

print(f"✅ Archivo limpio creado con {len(df_clean)} operaciones válidas totales.")
if not df_clean.empty:
    print(df_clean.head())
else:
    print("⚠️ No se generaron operaciones válidas.")

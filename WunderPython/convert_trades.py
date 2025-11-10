import pandas as pd

# === Leer el archivo original ===
df = pd.read_csv("trades_log.csv", header=0)

# Asegurar nombres correctos
df.columns = ["symbol", "event", "price", "timestamp"]

# Filtrar solo eventos válidos
df = df[df["event"].str.contains("ENTER|EXIT")]

trades = []
current_trade = None

for _, row in df.iterrows():
    if "ENTER" in row["event"]:
        current_trade = {
            "symbol": row["symbol"],
            "position": "LONG" if "LONG" in row["event"] else "SHORT",
            "price_entry": row["price"],
            "timestamp_entry": row["timestamp"],
        }
    elif "EXIT" in row["event"] and current_trade:
        current_trade["price_exit"] = row["price"]
        current_trade["timestamp_exit"] = row["timestamp"]

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

# Crear DataFrame nuevo
df_clean = pd.DataFrame(trades)

# Guardar resultado
df_clean.to_csv("AI_MODEL/trades_clean.csv", index=False)

print("✅ Archivo limpio creado: AI_MODEL/trades_clean.csv")
print(df_clean.head())

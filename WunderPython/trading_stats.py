# 📈 trading_stats.py
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import analyze_results

LOG_FILE = "trades_log.csv"
INITIAL_CAPITAL = 100.0  # igual al capital que usas en tu bot


def analyze_trades():
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("⚠️ No hay operaciones registradas aún.")
        return

    df["profit_pct"] = pd.to_numeric(df["profit_pct"], errors="coerce")
    df.dropna(subset=["profit_pct"], inplace=True)

    # === Estadísticas básicas ===
    total_trades = len(df)
    wins = len(df[df["profit_pct"] > 0])
    losses = len(df[df["profit_pct"] <= 0])
    win_rate = (wins / total_trades * 100) if total_trades else 0

    avg_win = df[df["profit_pct"] > 0]["profit_pct"].mean() if wins else 0
    avg_loss = df[df["profit_pct"] <= 0]["profit_pct"].mean() if losses else 0

    gross_profit = df[df["profit_pct"] > 0]["profit_pct"].sum()
    gross_loss = abs(df[df["profit_pct"] <= 0]["profit_pct"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    total_return = df["profit_pct"].sum()
    avg_trade = df["profit_pct"].mean()

    # === Equity Curve ===
    balance = INITIAL_CAPITAL
    balances = [balance]
    for pct in df["profit_pct"]:
        balance *= 1 + pct / 100
        balances.append(balance)
    df["equity"] = balances[1:]

    # === Drawdown ===
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    # === Resultados ===
    print("\n📊 RESULTADOS DEL BOT")
    print("-" * 40)
    print(f"📈 Total de trades: {total_trades}")
    print(f"✅ Ganadores: {wins} ({win_rate:.2f}%)")
    print(f"❌ Perdedoras: {losses} ({100 - win_rate:.2f}%)")
    print(f"💰 Ganancia media: {avg_win:.2f}% | Pérdida media: {avg_loss:.2f}%")
    print(f"⚖️ Profit Factor: {profit_factor:.2f}")
    print(f"📊 Promedio por trade: {avg_trade:.2f}%")
    print(f"💵 Rentabilidad total: {total_return:.2f}%")
    print(f"📉 Máx. drawdown: {max_dd:.2f}%")
    print(f"💼 Balance final: {balance:.2f} USDT\n")

    # === Gráfica de Equity ===
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["equity"], label="Equity Curve", linewidth=2)
    plt.title("Curva de capital del bot")
    plt.xlabel("Fecha")
    plt.ylabel("Capital (USDT)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_results()

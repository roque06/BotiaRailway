import os, time, pandas as pd, joblib, numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
DATA_FILE = "AI_MODEL/trades_clean.csv"
MODEL_FILE = "AI_MODEL/model_trading.pkl"
CHECK_INTERVAL_HOURS = 6  # 🕒 cada 6 horas (ajustable)
MIN_TRADES_REQUIRED = 8  # número mínimo de operaciones para reentrenar
TELEGRAM_TOKEN = "7543685147:AAGtQjY-wA97qmUTsahux75MQ-8vYeDgcls"
TELEGRAM_CHAT_ID = "1216693645"

# ===========================
# UTILIDAD: enviar mensaje Telegram
# ===========================
import requests


def send_telegram_message(msg):
    try:
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except:
        pass


# ===========================
# FUNCIÓN PRINCIPAL: autoentrenar IA
# ===========================
def auto_update_ai():
    print(
        "🧠 Auto-IA iniciada: verificará nuevos datos cada",
        CHECK_INTERVAL_HOURS,
        "horas.",
    )
    while True:
        try:
            if not os.path.exists(DATA_FILE):
                print("⚠️ No existe trades_clean.csv aún.")
                time.sleep(60 * 60)
                continue

            df = pd.read_csv(DATA_FILE)
            if df.empty or len(df) < MIN_TRADES_REQUIRED:
                print(
                    f"⚠️ Solo {len(df)} operaciones disponibles. Esperando más para entrenar..."
                )
                time.sleep(60 * 60)
                continue

            # ===========================
            # Preparar datos
            # ===========================
            df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
            df["target"] = (df["profit"] > 0).astype(int)
            df = df.dropna(subset=["profit"])

            X = df[["price_entry", "price_exit", "profit"]].copy()
            y = df["target"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ===========================
            # Entrenamiento IA
            # ===========================
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.25, random_state=42
            )

            model = RandomForestClassifier(n_estimators=80, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100

            # Guardar modelo
            joblib.dump(model, MODEL_FILE)
            print(f"✅ Auto-IA reentrenada correctamente ({acc:.2f}% precisión)")
            send_telegram_message(
                f"🧠 Auto-IA actualizada | Precisión {acc:.2f}% | {len(df)} trades usados"
            )

            # Esperar hasta el próximo ciclo
            time.sleep(CHECK_INTERVAL_HOURS * 3600)

        except Exception as e:
            print("⚠️ Error en auto_update_ai:", e)
            send_telegram_message(f"⚠️ Error Auto-IA: {e}")
            time.sleep(3600)  # espera 1h antes de reintentar

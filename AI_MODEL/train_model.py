import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Cargar datos del bot ===
df = pd.read_csv("trades_log.csv")

# === Limpiar datos ===
df = df.dropna()

# === Definir variables ===
features = ["ema9", "ema21", "ema200", "rsi", "atr"]
df["target"] = (df["profit"] > 0).astype(int)  # 1=ganó, 0=perdió

X = df[features]
y = df["target"]

# === Crear modelo IA ===
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X, y)

# === Guardar modelo entrenado ===
joblib.dump(model, "AI_MODEL/model_trading.pkl")

print("✅ Modelo IA entrenado y guardado con éxito.")

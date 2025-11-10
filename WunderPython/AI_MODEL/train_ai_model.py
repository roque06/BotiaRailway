import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# === Cargar datos ===
df = pd.read_csv("AI_MODEL/trades_clean.csv")

# === Simular variables técnicas ===
# ⚠️ En esta versión inicial, simulamos EMA/RSI/ATR si no están
# más adelante lo haremos real tomando los valores de tu bot
import numpy as np

if "ema9" not in df.columns:
    df["ema9"] = df["price_entry"] * (1 + np.random.uniform(-0.002, 0.002, len(df)))
if "ema21" not in df.columns:
    df["ema21"] = df["price_entry"] * (1 + np.random.uniform(-0.003, 0.003, len(df)))
if "ema200" not in df.columns:
    df["ema200"] = df["price_entry"] * (1 + np.random.uniform(-0.005, 0.005, len(df)))
if "rsi" not in df.columns:
    df["rsi"] = np.random.uniform(30, 70, len(df))
if "atr" not in df.columns:
    df["atr"] = np.random.uniform(100, 500, len(df))

# === Crear columna objetivo ===
df["target"] = (df["profit"] > 0).astype(int)  # 1 = ganancia, 0 = pérdida

# === Seleccionar features ===
features = ["ema9", "ema21", "ema200", "rsi", "atr"]
X = df[["ema9", "ema21", "ema200", "rsi", "atr"]]
y = df["target"]

# === Dividir en entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === Entrenar modelo IA ===
model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# === Evaluar ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Precisión del modelo: {acc:.2%}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Guardar modelo ===
joblib.dump(model, "AI_MODEL/model_trading.pkl")
print("💾 Modelo guardado en: AI_MODEL/model_trading.pkl")

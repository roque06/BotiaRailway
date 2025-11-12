import sys
import os
import time
import json
import requests
import pandas as pd
import ta
from datetime import datetime, UTC, timedelta
import http.server
import socketserver
import threading
import subprocess
import joblib
import math
import numpy as np


# ✅ Corrige el path para que Python encuentre AI_MODEL/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI_MODEL.auto_ai_manager import auto_update_ai
from AI_MODEL.market_context_ai import market_okay
from AI_MODEL.online_learner import OnlineLearner


# === RUTAS PERSISTENTES AUTOMÁTICAS ===
# Si Railway monta /mnt/data, guardamos allí los logs y modelos.
if os.path.exists("/mnt/data"):
    BASE_PATH = "/mnt/data"
else:
    BASE_PATH = os.getcwd()

LOG_CSV = os.path.join(BASE_PATH, "trades_log.csv")
PARAMS_FILE = os.path.join(BASE_PATH, "params.json")
STATE_FILE_TPL = os.path.join(BASE_PATH, "state_{symbol}.json")
AI_MODEL_PATH = os.path.join(BASE_PATH, "AI_MODEL/model_trading.pkl")

print(f"📁 Base path activo: {BASE_PATH}", flush=True)


# Inicia IA secundaria
threading.Thread(target=auto_update_ai, daemon=True).start()

# Modelo IA offline (RandomForest)
try:
    IA_MODEL_PATH = "AI_MODEL/model_trading.pkl"
    ia_model = joblib.load(IA_MODEL_PATH)
    print("🧠 Modelo IA cargado correctamente.")
except Exception as e:
    ia_model = None
    print("⚠️ No se pudo cargar el modelo IA:", e)

# Modelo IA on-line (aprendizaje incremental)
online_ai = OnlineLearner()
online_ai.is_warmed = True

# ✅ Entrenamiento inicial completo (scaler + modelo)
import numpy as np
X_init = np.array([[0, 0, 0, 50, 1]])  # ema9, ema21, ema200, rsi, atr
y_init = np.array([0])  # clase neutra

try:
    # Fittea el scaler antes de entrenar
    online_ai.scaler.fit(X_init)
    online_ai.clf.partial_fit(X_init, y_init, classes=[0, 1])
    print("🧩 IA online inicializada correctamente (scaler y coef_ listos).")
except Exception as e:
    print(f"⚠️ Error inicializando IA online: {e}")


CONTEXT_PATH = "AI_MODEL/context_state.json"
NOW_UTC = lambda: pd.Timestamp.now(tz="UTC")


def load_context():
    if os.path.exists(CONTEXT_PATH):
        try:
            return json.load(open(CONTEXT_PATH, "r", encoding="utf-8"))
        except Exception:
            pass
    # defaults
    return {
        "sentiment_bias": 0.5,  # 0 (muy bajista) .. 1 (muy alcista)
        "wins_short": 0,
        "wins_long": 0,
        "loss_short": 0,
        "loss_long": 0,
    }


def save_context(ctx: dict):
    try:
        json.dump(ctx, open(CONTEXT_PATH, "w", encoding="utf-8"))
    except Exception:
        pass


context = load_context()

# =====================================================
# SMART TRADING BOT v6 ULTIMATE (BTCUSDT) - Adaptativo
# =====================================================
# - Régimen de mercado (trend / range / explosive)
# - Multi-timeframe (15m base, 5m confirm, 1h macro)
# - ML-lite (clasificador de confluencia)
# - Drawdown diario/semanal + profit-lock
# - Auto-optimización semanal (crea/actualiza params.json)
# - Reload de parámetros en caliente
# - Riesgo adaptativo (régimen x racha x equity x profit-lock)
# - Trailing / Break-even / Parciales / Cooldown / Filtros ADX & ATR
# - Señales WunderTrading + Telegram


def ia_decision(
    ema9,
    ema21,
    ema200,
    rsi,
    atr,
    *,
    rsi_slope_now=None,
    atr_stable=None,
    vol_now=None,
    vol_ma=None,
    ema_fast_ref=None,
    ema_slow_ref=None,
):
    """
    Mezcla:
      - ia_model (offline)  -> p_off
      - online_ai (on-line) -> p_on
      - sesgo de contexto   -> bias
    """
    # ------------------
    # OFFLINE (con compat layer)
    p_off = 0.5
    try:
        if ia_model is not None and OFFLINE_FEATURES:
            # Usa ema_fast_ref/ema_slow_ref si fueron pasados; si no, cae a ema9/ema21
            ef = ema_fast_ref if ema_fast_ref is not None else ema9
            es = ema_slow_ref if ema_slow_ref is not None else ema21

            X_off = build_offline_feature_row(
                OFFLINE_FEATURES,
                ema_f=ef,
                ema_s=es,
                rsi_slope_now=(0.0 if rsi_slope_now is None else rsi_slope_now),
                atr_fast=atr,
                atr_stable=(atr if atr_stable is None else atr_stable),
                vol_now=(1.0 if vol_now is None else vol_now),
                vol_ma=(1.0 if vol_ma is None else vol_ma),
            )
            p_off = float(ia_model.predict_proba(X_off)[0][1])
    except Exception as e:
        print("⚠️ IA offline error:", e)

    # ------------------
    # ONLINE (usa 5 señales crudas como antes)
    try:
        X = [ema9, ema21, ema200, rsi, atr]
        p_on = float(online_ai.predict_proba([X])[0][1])
    except Exception as e:
        print("⚠️ IA online error:", e)
        p_on = 0.5

    # ------------------
    # Sesgo de contexto
    bias = float(context.get("sentiment_bias", 0.5))
    bias = max(0.35, min(0.65, bias))

    # Mezcla ponderada
    w_on = 0.5 if online_ai.is_warmed else 0.25
    w_off = 0.4
    w_bias = 0.1
    p = w_on * p_on + w_off * p_off + w_bias * bias
    return max(0.0, min(1.0, p))


def auto_train_ai_model():
    """Ejecuta conversión y entrenamiento IA automáticamente si hay nuevos trades"""
    try:
        log_file = "trades_log.csv"
        model_file = "AI_MODEL/model_trading.pkl"

        # Solo reentrena si hay trades nuevos o el modelo no existe
        if os.path.exists(log_file):
            log_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            model_time = (
                datetime.fromtimestamp(os.path.getmtime(model_file))
                if os.path.exists(model_file)
                else datetime.fromtimestamp(0)
            )

            if log_time > model_time:
                print("🧠 Detectados nuevos trades, reentrenando IA automáticamente...")
                subprocess.run(["python", "AI_MODEL/convert_trades.py"], check=True)
                subprocess.run(["python", "AI_MODEL/train_ai_model_advanced.py"], check=True)
                print("✅ IA reentrenada automáticamente con los nuevos datos.")
                send_telegram_message(
                    "🧠 IA reentrenada automáticamente con nuevos trades ✅"
                )
            else:
                print("ℹ️ IA ya está actualizada, sin nuevos trades.")
        else:
            print("⚠️ No se encontró trades_log.csv, no se puede reentrenar IA.")
    except Exception as e:
        print("⚠️ Error en auto-entrenamiento IA:", e)


def auto_retrain_ai(interval_hours=6):
    """Reentrena la IA cada N horas con los nuevos trades."""

    def retrain_loop():
        while True:
            try:
                print("🧠 Reentrenando IA con nuevos datos...")
                subprocess.run(
                    ["python", "AI_MODEL/train_ai_model_advanced.py"], check=True
                )
                print("✅ Auto-IA reentrenada correctamente.")
            except Exception as e:
                print(f"⚠️ Error reentrenando IA: {e}")
            time.sleep(interval_hours * 3600)

    t = threading.Thread(target=retrain_loop, daemon=True)
    t.start()


def estimate_sentiment_from_1h(df1h):
    """
    0..1. Usa slope EMA200 y cruce EMA9/EMA21.
    """
    ema_fast = float(df1h["ema_fast"].iloc[-1])
    ema_slow = float(df1h["ema_slow"].iloc[-1])
    ema200_now = float(df1h["ema_long"].iloc[-1])
    ema200_prev = float(df1h["ema_long"].iloc[-10]) if len(df1h) > 10 else ema200_now

    slope = (ema200_now - ema200_prev) / max(1e-9, ema200_prev)
    slope_score = (math.tanh(slope * 5000) + 1) / 2.0  # 0..1

    cross_score = 0.7 if ema_fast > ema_slow else 0.3
    sentiment = 0.6 * slope_score + 0.4 * cross_score
    return max(0.0, min(1.0, sentiment))


def apply_context_bias_from_result(side: str, profit_pct: float):
    """
    Actualiza memoria de contexto según resultado del trade.
    """
    global context
    win = profit_pct > 0
    if side.upper() == "LONG":
        context["wins_long"] += int(win)
        context["loss_long"] += int(not win)
    else:
        context["wins_short"] += int(win)
        context["loss_short"] += int(not win)

    # bias hacia 0..1; empuja un poco hacia LONG si ganan LONGs, hacia 0 si ganan SHORTs
    wl_long = context["wins_long"] - context["loss_long"]
    wl_short = context["wins_short"] - context["loss_short"]
    # mapea a [-0.1..+0.1]
    delta = max(-0.1, min(0.1, (wl_long - wl_short) / 50.0))
    context["sentiment_bias"] = max(0.0, min(1.0, 0.5 + delta))
    save_context(context)


# =====================================================
# AUTO-VALIDACIÓN Y REENTRENAMIENTO IA
# =====================================================
# arriba de todo, globals:
OFFLINE_FEATURES = None


def ensure_ia_model():
    """Carga el modelo IA y recuerda sus columnas esperadas (feature_names_in_)."""
    global ia_model, OFFLINE_FEATURES
    try:
        model_path = "AI_MODEL/model_trading.pkl"
        if not os.path.exists(model_path):
            print("⚠️ Modelo IA no encontrado; entrenando...")
            subprocess.run(["python", "AI_MODEL/convert_trades.py"], check=True)
            subprocess.run(
                ["python", "AI_MODEL/train_ai_model_advanced.py"], check=True
            )

        ia_model = joblib.load(model_path)
        # Detectar columnas esperadas
        OFFLINE_FEATURES = list(getattr(ia_model, "feature_names_in_", [])) or None
        print(f"🧠 Modelo IA cargado. Features esperadas: {OFFLINE_FEATURES}")
    except Exception as e:
        print(f"⚠️ Error al validar/cargar IA: {e}")
        ia_model = None
        OFFLINE_FEATURES = None


# Ejecutar verificación IA al inicio
ensure_ia_model()


def build_offline_feature_row(
    OFFLINE_FEATURES, ema_f, ema_s, rsi_slope_now, atr_fast, atr_stable, vol_now, vol_ma
):
    """
    Devuelve un DataFrame de una fila con las columnas que el modelo espera.
    Soporta al menos: atr_norm, ema_diff, rsi_slope, vol_ratio
    (y fácilmente extensible si tu modelo futuro añade más).
    """
    vals = {}
    for fname in OFFLINE_FEATURES or []:
        if fname == "atr_norm":
            vals[fname] = float(atr_fast) / max(float(atr_stable), 1e-9)
        elif fname == "ema_diff":
            # diferencia relativa entre 9 y 21
            vals[fname] = (float(ema_f) - float(ema_s)) / max(abs(float(ema_s)), 1e-9)
        elif fname == "rsi_slope":
            vals[fname] = float(rsi_slope_now)
        elif fname == "vol_ratio":
            vals[fname] = float(vol_now) / max(float(vol_ma), 1e-9)
        else:
            # fallback en caso de futuras features: 0.0
            vals[fname] = 0.0

    # Asegura orden de columnas consistente con el modelo
    df = pd.DataFrame(
        [[vals.get(k, 0.0) for k in OFFLINE_FEATURES]], columns=OFFLINE_FEATURES
    )
    return df


# ==============================
# GESTIÓN DE POSICIÓN DINÁMICA
# ==============================
# Breakeven
BREAKEVEN_TRIGGER = 1.0  # % de ganancia para activar BE
BREAKEVEN_OFFSET = (
    0.10  # % por encima del precio de entrada (LONG) o por debajo (SHORT)
)

# Trailing base y niveles de apriete automático
TRAIL_BASE_PCT = 0.8  # % trailing por defecto
TRAIL_TIGHT_AT = [2.0, 3.0, 5.0]  # cuando la ganancia supere estos %...
TRAIL_VALUES = [0.6, 0.4, 0.3]  # ...aprieta a estos trailing %

# Seguridad
MIN_PROFIT_TO_CLOSE = 0.3  # %: si cae por debajo de esto después del BE, cierra


# ==============================
# CONFIG GENERAL
# ==============================
SYMBOLS = ["BTCUSDT"]  # Solo BTCUSDT
INTERVAL = "15m"  # timeframe base
CONFIRM_INTERVAL = "5m"  # confirmación táctica
CONFIRM_INTERVAL_MACRO = "1h"  # confirmación macro
WUNDER_WEBHOOK = "https://wtalerts.com/bot/custom"
POLL_SECONDS = 15
LOG_CSV = "trades_log.csv"
STATE_FILE_TPL = "state_{symbol}.json"
PARAMS_FILE = "params.json"
DUP_SIGNAL_COOLDOWN_SEC = 10

INITIAL_CAPITAL = 100.0
capital = INITIAL_CAPITAL

# ==============================
# ESTRATEGIA (parámetros base)
# ==============================
EMA_FAST, EMA_SLOW, EMA_LONG = 9, 21, 200
RSI_PERIOD = 14
RSI_LONG_MIN, RSI_LONG_MAX = 35, 76
RSI_SHORT_MIN, RSI_SHORT_MAX = 25, 62
EMA_DIFF_MARGIN = 0.0002

ATR_PERIOD = 14
ATR_SL_MULT = 1.6
ATR_TRAIL_MULT = 1.5
MIN_PROFIT_TO_CLOSE = 0.3  # %

# Break-even
BREAKEVEN_TRIGGER = 1.0  # %
BREAKEVEN_OFFSET = 0.15  # % (reservado para mejoras)

# ==============================
# COOLDOWN / SEGURIDAD
# ==============================
COOLDOWN_AFTER_EXIT_SEC = 300  # 5 min tras cualquier salida
MAX_CONSECUTIVE_LOSSES = 3  # autopausa tras N pérdidas seguidas
AUTO_PAUSE_SECONDS = 3600  # 1h

# ==============================
# RIESGO / VOLATILIDAD
# ==============================
RISK_PCT_BASE = 2.0  # % riesgo teórico base por operación
RISK_PCT_MIN, RISK_PCT_MAX = 0.5, 2.0
VOLATILITY_MULT_LIMIT = 1.9  # si ATR_f > ATR_MA * este múltiplo, no abrir
EQUITY_CURVE_LOOKBACK = 10  # trades para controlar curva de equity
EQUITY_DRAWDOWN_TH_PCT = -2.0  # si últimos N trades suman <-2%, bajar riesgo

# ==============================
# DRAWDOWN INSTITUCIONAL
# ==============================
DAILY_MAX_LOSS_PCT = 2.0  # límite de pérdidas diarias (% sumado del log)
WEEKLY_MAX_LOSS_PCT = 5.0  # límite de pérdidas semanales
PROFIT_LOCK_PCT = 2.0  # si en el día ya ganaste >=2% bloquea subir riesgo

# ==============================
# FILTROS AVANZADOS
# ==============================
ADX_MIN = 10  # fuerza mínima de tendencia
ATR_PCTL_WINDOW = 200
ATR_PCTL_THRESHOLD = 0.90
CANDLE_STRENGTH_MIN = 0.1
TRAIL_MIN_MOVE_ATR = 0.3
SKIP_UTC_HOURS = (0, 1, 2, 3)  # horas muertas
SKIP_WEEKENDS = False
PARTIAL_TAKE_PROFIT_PCT = 1.5
ENABLE_PARTIALS = True

# ==============================
# CÓDIGOS DE SEÑALES WUNDER
# ==============================
SIGNAL_CODES = {
    "BTCUSDT": {
        "ENTER_LONG": "ENTER-LONG_Binance_BTCUSDT_BOT-BTC_5M_d99fa795a087b1d1fa830920",
        "ENTER_SHORT": "ENTER-SHORT_Binance_BTCUSDT_BOT-BTC_5M_d99fa795a087b1d1fa830920",
        "EXIT_ALL": "EXIT-ALL_Binance_BTCUSDT_BOT-BTC_5M_d99fa795a087b1d1fa830920",
        # "TAKE_PROFIT_PARTIAL": "TP-PARTIAL_Binance_BTCUSDT_BTC-BOT_15M_xxx"
    }
}

# ==============================
# TELEGRAM
# ==============================
TELEGRAM_TOKEN = "7543685147:AAGtQjY-wA97qmUTsahux75MQ-8vYeDgcls"
TELEGRAM_CHAT_ID = "1216693645"


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass


def test_telegram():
    send_telegram_message(
        "✅ Bot v6 ULTIMATE conectado con Telegram.\n"
        "Cargando parámetros y verificando data…"
    )


# ==============================
# ESTADO (persistencia por símbolo)
# ==============================
def state_path(symbol: str) -> str:
    return STATE_FILE_TPL.format(symbol=symbol)


def load_state(symbol: str):
    path = state_path(symbol)
    if not os.path.exists(path):
        return {
            "last_side": None,
            "entry_price": None,
            "trail_price": None,
            "cooldown_until": 0,
            "breakeven_active": False,
            "consecutive_losses": 0,
            "last_pnl_pct": None,
            "sl_price": None,
            "partial_taken": False,
            "entry_snapshot": None,
            "regime": None,
            "last_signal": None,
            "last_signal_ts": 0,
            # para auto-optimización semanal:
            "last_autoopt_date": None,  # YYYY-MM-DD ejecutado por última vez (UTC)
            # en load_state() -> defaults y sane defaults:
            "highest_price": None,  # para LONG
            "lowest_price": None,  # para SHORT
            "dynamic_trail_pct": None,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
            # sane defaults
            s.setdefault("consecutive_losses", 0)
            s.setdefault("last_pnl_pct", None)
            s.setdefault("breakeven_active", False)
            s.setdefault("cooldown_until", 0)
            s.setdefault("sl_price", None)
            s.setdefault("partial_taken", False)
            s.setdefault("entry_snapshot", None)
            s.setdefault("regime", None)
            s.setdefault("last_signal", None)
            s.setdefault("last_signal_ts", 0)
            s.setdefault("last_autoopt_date", None)
            s.setdefault("highest_price", None)
            s.setdefault("lowest_price", None)
            s.setdefault("dynamic_trail_pct", None)
            return s
    except:
        return {
            "last_side": None,
            "entry_price": None,
            "trail_price": None,
            "cooldown_until": 0,
            "breakeven_active": False,
            "consecutive_losses": 0,
            "last_pnl_pct": None,
            "sl_price": None,
            "partial_taken": False,
            "entry_snapshot": None,
            "regime": None,
            "last_signal": None,
            "last_signal_ts": 0,
            "last_autoopt_date": None,
        }


def save_state(symbol: str, state: dict):
    path = state_path(symbol)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)


# ==============================
# PARAM RELOAD (auto-optimización externa)
# ==============================
def maybe_reload_params():
    try:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                p = json.load(f)
            changed = []
            for k, v in p.items():
                if k in globals() and globals()[k] != v:
                    globals()[k] = v
                    changed.append((k, v))
            if changed:
                send_telegram_message(
                    "♻️ Parámetros recargados: "
                    + ", ".join(f"{k}={v}" for k, v in changed)
                )
    except Exception as e:
        print("⚠️ Error recargando params:", e, flush=True)


# ==============================
# DESCARGA DE DATOS
# ==============================
def fetch_klines(symbol, interval, limit=500, retries=5, backoff=5):
    """Descarga velas con fallback US->Global, user-agent y limpieza de datos."""
    primary_url = "https://api.binance.us/api/v3/klines"
    backup_url = "https://api.binance.com/api/v3/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for i in range(retries):
        try:
            r = requests.get(primary_url, params=params, headers=headers, timeout=10)
            if r.status_code in (418, 451) or not r.ok:
                r = requests.get(backup_url, params=params, headers=headers, timeout=10)

            r.raise_for_status()
            data = r.json()

            clean_data = [
                row[:12] for row in data if isinstance(row, list) and len(row) >= 12
            ]
            df = pd.DataFrame(
                clean_data,
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
                    "taker_base",
                    "taker_quote",
                    "ignore",
                ],
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "taker_base",
                "taker_quote",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df.dropna(subset=["close"])
            return df

        except Exception as e:
            print(
                f"⚠️ Error Binance {symbol} (intento {i+1}/{retries}): {e}", flush=True
            )
            time.sleep(backoff * (i + 1))

    raise RuntimeError(f"⚠️ Binance no responde para {symbol} tras varios intentos.")


def record_entry(symbol: str, side: str, price: float, sl_price: float,
                 ema_f: float, ema_s: float, ema_long: float, rsi: float, atr_fast: float):
    state = load_state(symbol)
    state.update({
        "last_side": side,
        "entry_price": price,
        "trail_price": None,
        "sl_price": sl_price,
        "breakeven_active": False,
        "partial_taken": False,
        "highest_price": None,
        "lowest_price": None,
        "dynamic_trail_pct": None,
        "entry_snapshot": {
            "ema_fast": ema_f,
            "ema_slow": ema_s,
            "ema_long": ema_long,
            "rsi": rsi,
            "atr": atr_fast,
        },
        "cooldown_until": time.time() + 10,  # pequeño cooldown para evitar dobles entradas
    })
    save_state(symbol, state)


# ==============================
# SEÑALES (WunderTrading)
# ==============================
def send_signal(symbol: str, code: str) -> bool:
    state = load_state(symbol)
    now_ts = time.time()

    # anti-duplicados locales
    if (
        state.get("last_signal") == code
        and (now_ts - state.get("last_signal_ts", 0)) < DUP_SIGNAL_COOLDOWN_SEC
    ):
        print(f"↩️ {symbol} señal ignorada (cooldown antidupe).")
        return False

    try:
        r = requests.post(WUNDER_WEBHOOK, json={"code": code}, timeout=10)
        ok = (200 <= r.status_code < 300)
        print(f"[{datetime.now(UTC)}] {symbol} Signal -> {code} | status={r.status_code}", flush=True)

        if ok:
            state["last_signal"] = code
            state["last_signal_ts"] = now_ts
            save_state(symbol, state)
        else:
            # útil para depurar por qué el provider rechazó
            try:
                print(f"Body: {r.text[:300]}", flush=True)
            except Exception:
                pass
        return ok
    except Exception as e:
        print(f"⚠️ Error enviando señal {symbol}: {e}", flush=True)
        return False


# ==============================
# INDICADORES
# ==============================
def compute_indicators(df):
    # EMAs
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["ema_long"] = ta.trend.ema_indicator(df["close"], window=EMA_LONG)

    # RSI base + suavizado y pendiente
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["rsi_smooth"] = ta.trend.ema_indicator(df["rsi"], window=5)
    df["rsi_slope"] = df["rsi_smooth"].diff()

    # ATR y su media (estable)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=ATR_PERIOD
    )
    df["atr_ma"] = df["atr"].rolling(ATR_PERIOD).mean()

    # Percentil de ATR (riesgo extremo)
    df["atr_p90"] = df["atr"].rolling(ATR_PCTL_WINDOW).quantile(ATR_PCTL_THRESHOLD)

    # Volumen promedio 20 velas
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["vol_ma"] = df["volume"].rolling(20).mean()

    # ADX (fuerza de tendencia)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    return df


def current_profit_pct(side: str, entry: float, price: float) -> float:
    if not entry:
        return 0.0
    if side == "LONG":
        return (price - entry) / entry * 100.0
    else:
        return (entry - price) / entry * 100.0


def pick_dynamic_trailing(base_pct: float, profit_pct: float) -> float:
    """
    Devuelve el trailing % según los escalones definidos.
    A mayor ganancia, menor trailing (más apretado).
    """
    t = base_pct
    for trigger, tight in zip(TRAIL_TIGHT_AT, TRAIL_VALUES):
        if profit_pct >= trigger:
            t = min(t, tight)
    return t


def apply_breakeven_if_needed(state: dict, side: str, price: float, symbol: str):

    """
    Activa el breakeven cuando profit >= BREAKEVEN_TRIGGER.
    Ajusta sl_price al precio de entrada +/- offset.
    """
    if state.get("breakeven_active"):
        return

    entry = state.get("entry_price")
    if entry is None:
        return

    profit = current_profit_pct(side, entry, price)
    if profit >= BREAKEVEN_TRIGGER:
        if side == "LONG":
            be_price = entry * (1 + BREAKEVEN_OFFSET / 100.0)
        else:
            be_price = entry * (1 - BREAKEVEN_OFFSET / 100.0)

        state["sl_price"] = be_price
        state["breakeven_active"] = True
        send_telegram_message(
            f"🛡️ Breakeven activado a {be_price:.2f} ({profit:.2f}% ganancia)"
        )
        save_state(symbol, state)


def update_trailing(state: dict, side: str, price: float, symbol: str):
    """
    Actualiza el trailing interno:
      - registra máximo/mínimo a favor
      - elige trailing % dinámico según la ganancia
      - recalcula 'trail_price' acorde
    """
    entry = state.get("entry_price")
    if entry is None:
        return

    # 1) máximo/mínimo a favor
    if side == "LONG":
        state["highest_price"] = (
            price
            if state.get("highest_price") is None
            else max(state["highest_price"], price)
        )
        ref = state["highest_price"]
    else:
        state["lowest_price"] = (
            price
            if state.get("lowest_price") is None
            else min(state["lowest_price"], price)
        )
        ref = state["lowest_price"]

    # 2) elige trailing dinámico
    profit = current_profit_pct(side, entry, price)
    dyn_trail = pick_dynamic_trailing(TRAIL_BASE_PCT, profit)
    state["dynamic_trail_pct"] = dyn_trail

    # 3) calcula el precio de trailing
    if side == "LONG":
        new_trail = ref * (1 - dyn_trail / 100.0)
        # nunca aflojar: sólo sube el trailing
        if state.get("trail_price") is None or new_trail > state["trail_price"]:
            state["trail_price"] = new_trail
    else:
        new_trail = ref * (1 + dyn_trail / 100.0)
        if state.get("trail_price") is None or new_trail < state["trail_price"]:
            state["trail_price"] = new_trail

    # 4) opcional: aprieta también el SL si ya hay BE activo
    if state.get("breakeven_active"):
        if side == "LONG" and state["sl_price"] is not None:
            state["sl_price"] = max(state["sl_price"], new_trail)
        elif side == "SHORT" and state["sl_price"] is not None:
            state["sl_price"] = min(state["sl_price"], new_trail)

    save_state(symbol, state)


def should_exit_now(state: dict, side: str, price: float) -> tuple[bool, str]:
    """
    Devuelve (exit?, motivo)
    Sale si el precio rompe el trailing o el SL.
    Además, si ya había BE y la ganancia cae por debajo de MIN_PROFIT_TO_CLOSE.
    """
    entry = state.get("entry_price")
    trail = state.get("trail_price")
    sl_pr = state.get("sl_price")
    be_on = state.get("breakeven_active", False)

    # 1) trailing / SL
    if side == "LONG":
        if trail is not None and price <= trail:
            return True, f"Trailing hit ({trail:.2f})"
        if sl_pr is not None and price <= sl_pr:
            return True, f"StopLoss hit ({sl_pr:.2f})"
    else:
        if trail is not None and price >= trail:
            return True, f"Trailing hit ({trail:.2f})"
        if sl_pr is not None and price >= sl_pr:
            return True, f"StopLoss hit ({sl_pr:.2f})"

    # 2) si hay BE y el beneficio se “desinfla” demasiado, cerrar para asegurar
    if be_on and entry is not None:
        profit = current_profit_pct(side, entry, price)
        if profit <= MIN_PROFIT_TO_CLOSE:
            return True, f"Protección BE: profit cayó a {profit:.2f}%"

    return False, ""


# ==============================
# RÉGIMEN DE MERCADO
# ==============================
def detect_regime(df):
    """Devuelve: 'trend', 'range', 'explosive' según ADX, ATR y pendiente EMA200."""
    adx_now = float(df["adx"].iloc[-1])
    atr_now = float(df["atr"].iloc[-1])
    atr_ma = float(df["atr_ma"].iloc[-1])
    atr_p90 = df["atr_p90"].iloc[-1]
    ema200_now = float(df["ema_long"].iloc[-1])
    ema200_prev = float(df["ema_long"].iloc[-5]) if len(df) > 5 else ema200_now
    ema200_slope = (ema200_now - ema200_prev) / max(1e-9, ema200_prev)

    explosive = False
    if not pd.isna(atr_p90) and atr_now > float(atr_p90):
        explosive = True
    if atr_now > atr_ma * (VOLATILITY_MULT_LIMIT * 1.1):
        explosive = True

    if explosive:
        return "explosive"
    if adx_now >= 30 and abs(ema200_slope) > 0.0005:
        return "trend"
    if adx_now < ADX_MIN:
        return "range"
    return "range"


# ==============================
# EQUITY CURVE CONTROL
# ==============================
def equity_curve_adjustment():
    """Lee los últimos trades del LOG_CSV y devuelve multiplicador de riesgo (0.6~1.1)."""
    try:
        if not os.path.exists(LOG_CSV):
            return 1.0
        df = pd.read_csv(LOG_CSV)
        if "profit_pct" not in df.columns:
            return 1.0
        last = df.tail(EQUITY_CURVE_LOOKBACK)
        perf = float(last["profit_pct"].fillna(0).sum())
        if perf <= EQUITY_DRAWDOWN_TH_PCT:
            return 0.7  # baja riesgo si rinde mal
        if perf > 3.0:
            return 1.1  # sube levemente si rinde muy bien
        return 1.0
    except Exception:
        return 1.0


# ==============================
# DRAWDOWN DIARIO/SEMANAL
# ==============================
def read_trades():
    try:
        return pd.read_csv(LOG_CSV)
    except:
        return pd.DataFrame()


def check_drawdown_limits():
    df = read_trades()
    if df.empty:
        return (False, False, False)
    try:
        # Fuerza timestamps con zona UTC
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["profit_pct"] = pd.to_numeric(df["profit_pct"], errors="coerce")

        now_ts = pd.Timestamp.now(tz="UTC")                 # pandas Timestamp (aware)
        start_week = now_ts - pd.Timedelta(days=7)

        # “Hoy” comparando por fecha en UTC
        today = df[df["time"].dt.date == now_ts.date()]
        # Últimos 7 días con tipos compatibles
        week = df[df["time"] >= start_week]

        day_sum = today["profit_pct"].sum()
        week_sum = week["profit_pct"].sum()

        profit_lock = day_sum >= PROFIT_LOCK_PCT
        day_limit = day_sum <= -DAILY_MAX_LOSS_PCT
        week_limit = week_sum <= -WEEKLY_MAX_LOSS_PCT
        return day_limit, week_limit, profit_lock
    except Exception:
        return (False, False, False)


def log_trade(
    symbol, side, entry_price, exit_price, profit_pct, size_info=None, reason=""
):
    try:
        row = {
            "time": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_pct": profit_pct,
            "size_info": size_info if size_info is not None else "",
            "reason": reason,
        }
        pd.DataFrame([row]).to_csv(
            LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False
        )
    except Exception as e:
        print("⚠️ Error al escribir LOG_CSV:", e, flush=True)

    # === IA on-line: aprender del resultado si tenemos snapshot de entrada ===
    try:
        st = load_state(symbol)
        snap = st.get("entry_snapshot")
        if snap is not None and profit_pct is not None:
            # features iguales a la IA offline: [ema_fast, ema_slow, ema_long, rsi, atr]
            X = [
                float(snap["ema_fast"]),
                float(snap["ema_slow"]),
                float(snap["ema_long"]),
                float(snap["rsi"]),
                float(snap["atr"]),
            ]
            y = 1 if float(profit_pct) > 0 else 0
            online_ai.partial_update([X], [y])
            apply_context_bias_from_result(side, float(profit_pct))
            print(f"🟢 IA on-line actualizada con outcome={y} ({profit_pct:.2f}%)")
    except Exception as e:
        print("⚠️ Online-learning falló:", e)


def auto_tune_by_regime(regime: str):
    """
    Ajusta filtros según rendimiento reciente y régimen actual (últimos 30 trades).
    Conservador y acotado.
    """
    global ADX_MIN, RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX
    df = read_trades()
    if df.empty or "profit_pct" not in df.columns:
        return

    last = df.tail(30).copy()
    last["profit_pct"] = pd.to_numeric(last["profit_pct"], errors="coerce").fillna(0.0)
    wr = (last["profit_pct"] > 0).mean()
    avg = last["profit_pct"].mean()

    if regime == "range":
        # en rango, pide menos ADX, rsi más centrado
        if wr < 0.45 or avg < 0:
            ADX_MIN = max(8, ADX_MIN - 1)
            RSI_LONG_MIN = min(45, RSI_LONG_MIN + 1)
            RSI_LONG_MAX = max(55, RSI_LONG_MAX - 1)
            RSI_SHORT_MIN = min(35, RSI_SHORT_MIN + 1)
            RSI_SHORT_MAX = max(55, RSI_SHORT_MAX - 1)
    elif regime == "trend":
        # en tendencia, endurece un poco si rinde mal
        if wr < 0.45 or avg < 0:
            ADX_MIN = min(35, ADX_MIN + 1)
            # separa LONG/SHORT de los rangos medios
            RSI_LONG_MIN = max(30, RSI_LONG_MIN - 1)
            RSI_LONG_MAX = min(75, RSI_LONG_MAX + 1)
            RSI_SHORT_MIN = max(25, RSI_SHORT_MIN - 1)
            RSI_SHORT_MAX = min(65, RSI_SHORT_MAX + 1)
    else:  # explosive
        # reduce señales si rinde mal en alta volatilidad
        if wr < 0.5 or avg < 0:
            ADX_MIN = min(35, ADX_MIN + 1)

    # persistir
    try:
        base = {}
        if os.path.exists(PARAMS_FILE):
            base = json.load(open(PARAMS_FILE, "r", encoding="utf-8"))
        base.update(
            {
                "ADX_MIN": ADX_MIN,
                "RSI_LONG_MIN": RSI_LONG_MIN,
                "RSI_LONG_MAX": RSI_LONG_MAX,
                "RSI_SHORT_MIN": RSI_SHORT_MIN,
                "RSI_SHORT_MAX": RSI_SHORT_MAX,
            }
        )
        json.dump(base, open(PARAMS_FILE, "w", encoding="utf-8"))
    except Exception as e:
        print("⚠️ auto_tune_by_regime persistencia:", e)


def ml_score(features, ia_prob: float = 0.5):
    """
    Calcula una puntuación heurística de entrada (ML-lite) basada en los indicadores.
    Incluye:
      - RSI, pendiente RSI
      - ADX (fuerza de tendencia)
      - EMA trend (cruce 9/21)
      - Volumen relativo
      - Régimen de mercado (trend / range / explosive)
      - IA probabilística (p_ref)
    """
    w = {
        "rsi": 0.04,           # sensibilidad RSI
        "rsi_slope": 2.0,      # impulso del RSI
        "adx": 0.03,           # fuerza de tendencia
        "ema_trend": 0.8,      # cruce de medias
        "vol_rel": 0.5,        # actividad de volumen
        "regime_trend": 0.6,   # peso del régimen
        "ia_boost": 2.0,       # refuerzo IA
    }

    score = 0.0
    score += w["rsi"] * (features["rsi"] - 50)
    score += w["rsi_slope"] * features["rsi_slope"]
    score += w["adx"] * (features["adx"] - 20)
    score += w["ema_trend"] * (1 if features["ema_trend"] else -1)
    score += w["vol_rel"] * features["vol_rel"]

    # --- Régimen: flexible ---
    regime = features.get("regime", "range")
    if regime == "trend":
        score += w["regime_trend"] * 1.0
    elif regime == "explosive":
        # solo penaliza si la IA no está convencida
        score += w["regime_trend"] * (0.6 if ia_prob >= 0.60 else -0.3)
    else:  # range
        score += w["regime_trend"] * 0.0

    # --- Refuerzo IA ---
    score += w["ia_boost"] * (ia_prob - 0.5)

    return score

    return score


ML_THRESHOLD = 0.0  # si quieres ser más estricto, súbelo a 1.0


# ==============================
# AUTO-OPTIMIZACIÓN SEMANAL
# ==============================
def auto_optimize_params():
    df = read_trades()
    # Normaliza tiempos a UTC (pandas Timestamp)
    df["time"] = pd.to_datetime(df.get("time", pd.Series([])), errors="coerce", utc=True)

    now_ts = pd.Timestamp.now(tz="UTC")
    week_mask = df["time"] >= (now_ts - pd.Timedelta(days=7))
    week_df = df[week_mask]
    if week_df.empty or "profit_pct" not in week_df.columns:
        return None

    week_df["profit_pct"] = pd.to_numeric(
        week_df["profit_pct"], errors="coerce"
    ).fillna(0.0)
    n = len(week_df)
    wins = (week_df["profit_pct"] > 0).sum()
    winrate = wins / max(n, 1)
    avg_pnl = week_df["profit_pct"].mean()

    # Copiamos los valores actuales
    new_params = {
        "RSI_LONG_MIN": RSI_LONG_MIN,
        "RSI_LONG_MAX": RSI_LONG_MAX,
        "ADX_MIN": ADX_MIN,
        "RISK_PCT_BASE": RISK_PCT_BASE,
    }

    # Ajustes de RSI: si exceso de whipsaw (bajo winrate), endurecer filtros
    if winrate < 0.40 or avg_pnl < -0.5:
        # endurecer LONG: sube piso y baja techo
        new_params["RSI_LONG_MIN"] = min(max(RSI_LONG_MIN + 2, 30), 50)
        new_params["RSI_LONG_MAX"] = max(min(RSI_LONG_MAX - 2, 80), 55)
        # pedir más tendencia
        new_params["ADX_MIN"] = min(max(ADX_MIN + 2, 10), 35)
        # baja riesgo base
        new_params["RISK_PCT_BASE"] = float(
            max(RISK_PCT_MIN, round(RISK_PCT_BASE - 0.2, 2))
        )

    # Si está funcionando bien, relajar levemente (sin exceder límites)
    elif winrate > 0.55 and avg_pnl > 0.5:
        new_params["RSI_LONG_MIN"] = min(max(RSI_LONG_MIN - 1, 30), 50)
        new_params["RSI_LONG_MAX"] = max(min(RSI_LONG_MAX + 1, 80), 55)
        new_params["ADX_MIN"] = min(max(ADX_MIN - 1, 10), 35)
        new_params["RISK_PCT_BASE"] = float(
            min(RISK_PCT_MAX, round(RISK_PCT_BASE + 0.1, 2))
        )

    # Guardar params.json
    try:
        # mezclar con params existentes si los hay
        base = {}
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                base = json.load(f)
        base.update(new_params)
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False)
        return {
            "n": n,
            "winrate": round(winrate, 3),
            "avg": round(avg_pnl, 3),
            "new_params": new_params,
        }
    except Exception as e:
        print("⚠️ Error guardando params.json:", e, flush=True)
        return None


def is_sunday_utc(dt: datetime) -> bool:
    # Monday=0 ... Sunday=6
    return dt.weekday() == 6


def maybe_run_weekly_autoopt(symbol: str, state: dict):
    """
    Ejecuta la auto-optimización una sola vez cada domingo UTC.
    Guarda en el state la fecha (YYYY-MM-DD) de última ejecución.
    """
    now = datetime.now(UTC)
    if not is_sunday_utc(now):
        return state  # solo domingos

    today_str = now.date().isoformat()
    last_run = state.get("last_autoopt_date")
    if last_run == today_str:
        return state  # ya se ejecutó hoy

    result = auto_optimize_params()
    if result:
        msg = f"🧠 Auto-optimización semanal: trades={result['n']}, WinRate={result['winrate']*100:.1f}%, AvgPnL={result['avg']:.2f}%\n"
        msg += "Nuevos parámetros: " + ", ".join(
            [f"{k}={v}" for k, v in result["new_params"].items()]
        )
        send_telegram_message(msg)
        print(msg, flush=True)
    else:
        send_telegram_message(
            "🧠 Auto-optimización semanal: sin datos suficientes o sin cambios."
        )

    state["last_autoopt_date"] = today_str
    save_state(symbol, state)
    # recargar inmediatamente los nuevos parámetros si existen
    maybe_reload_params()
    return state


def main():
    print("🚀 Bot v6 ULTIMATE: multiTF + ML-lite + DD + riesgo adaptativo + IA Copiloto + On-line + Auto-tuning.", flush=True)
    auto_retrain_ai(interval_hours=6)
    print("🧠 Aprendizaje continuo activado cada 6 horas.")
    test_telegram()
    send_telegram_message("🤖 v6 ULTIMATE con IA Copiloto + On-line activo (15m/5m/1h).")
    maybe_reload_params()

    # --- Parámetros de control de reversión (flip) ---
    FLIP_MIN_IA = 0.70           # IA mínima para permitir flip
    FLIP_REQUIRE_ADX = 18        # fuerza mínima para flip
    FLIP_COOLDOWN_SEC = 5        # micro-pausa entre EXIT y ENTER opuesto

    consecutive_fetch_errors = 0

    while True:
        try:
            for SYMBOL in SYMBOLS:
                state = load_state(SYMBOL)

                # Auto-optimización semanal + recarga params
                state = maybe_run_weekly_autoopt(SYMBOL, state)
                maybe_reload_params()

                now_utc = datetime.now(UTC)

                # =============== DATOS MULTI-TF ===============
                df15 = compute_indicators(fetch_klines(SYMBOL, INTERVAL, 900))
                df5  = compute_indicators(fetch_klines(SYMBOL, CONFIRM_INTERVAL, 900))
                df1h = compute_indicators(fetch_klines(SYMBOL, CONFIRM_INTERVAL_MACRO, 900))
                consecutive_fetch_errors = 0

                # =============== Indicadores ===============
                price       = float(df15["close"].iloc[-1])
                ema_f       = float(df15["ema_fast"].iloc[-1])
                ema_s       = float(df15["ema_slow"].iloc[-1])
                ema_long    = float(df15["ema_long"].iloc[-1])
                rsi         = float(df15["rsi"].iloc[-1])
                rsi_slope   = float(df15["rsi_slope"].iloc[-1])
                adx_now     = float(df15["adx"].iloc[-1])
                atr_fast    = float(df15["atr"].iloc[-1])
                atr_stable  = float(df15["atr_ma"].iloc[-1])
                vol_now     = float(df15["volume"].iloc[-1])
                vol_ma      = float(df15["vol_ma"].iloc[-1])

                ema_f_5, ema_s_5   = float(df5["ema_fast"].iloc[-1]), float(df5["ema_slow"].iloc[-1])
                rsi_5, rsi_slope_5 = float(df5["rsi"].iloc[-1]), float(df5["rsi_slope"].iloc[-1])
                ema_f_1h, ema_s_1h = float(df1h["ema_fast"].iloc[-1]), float(df1h["ema_slow"].iloc[-1])
                adx_1h             = float(df1h["adx"].iloc[-1])

                # =============== Régimen + auto-tuning ===============
                regime = detect_regime(df15)
                state["regime"] = regime
                save_state(SYMBOL, state)
                auto_tune_by_regime(regime)

                # =============== IA y contexto ===============
                sentiment = estimate_sentiment_from_1h(df1h)
                context_bias = float(context.get("sentiment_bias", 0.5))
                print(f"⏱️ {now_utc} | {SYMBOL} | P={price:.2f} | RSI={rsi:.1f}/{rsi_5:.1f} | ADX={adx_now:.1f} | regime={regime} | sentiment={sentiment:.2f} | bias_mem={context_bias:.2f}", flush=True)

                ia_prob = ia_decision(
                    ema_f, ema_s, ema_long, rsi, atr_fast,
                    rsi_slope_now=rsi_slope,
                    atr_stable=atr_stable,
                    vol_now=vol_now, vol_ma=vol_ma,
                    ema_fast_ref=ema_f, ema_slow_ref=ema_s,
                )
                print(f"🧩 IA → probabilidad de éxito: {ia_prob:.2%}")
                print(f"🤖 warmed={online_ai.is_warmed} | modelo={'OK' if ia_model else 'None'}")

                # Copiloto IA: micro-ajustes de filtros
                global RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX, ADX_MIN, RISK_PCT_BASE
                if ia_prob > 0.75:
                    RSI_LONG_MIN = max(30, RSI_LONG_MIN - 2)
                    RSI_LONG_MAX = min(80, RSI_LONG_MAX + 2)
                    ADX_MIN      = max(8, ADX_MIN - 1)
                elif ia_prob < 0.20:
                    RSI_LONG_MIN = min(45, RSI_LONG_MIN + 2)
                    RSI_LONG_MAX = max(60, RSI_LONG_MAX - 2)
                    ADX_MIN      = min(35, ADX_MIN + 2)

                # Límites de seguridad
                day_limit, week_limit, profit_lock = check_drawdown_limits()

                # Filtros de ruido
                if atr_fast > atr_stable * 2.5 or adx_now < 8:
                    continue

                # =============== Señales base ===============
                ema_cross_up = ema_f > ema_s * (1 + EMA_DIFF_MARGIN)
                ema_cross_dn = ema_f < ema_s * (1 - EMA_DIFF_MARGIN)

                long_align_5m  = (ema_f_5 > ema_s_5) and (rsi_slope_5 > 0) and (rsi_5 >= 40)
                short_align_5m = (ema_f_5 < ema_s_5) and (rsi_slope_5 < 0) and (rsi_5 <= 60)

                bullish_ok = (
                    ema_cross_up and (RSI_LONG_MIN <= rsi <= RSI_LONG_MAX) and (rsi_slope > 0)
                    and (price > ema_long) and (vol_now > vol_ma) and long_align_5m
                )
                bearish_ok = (
                    ema_cross_dn and (RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX) and (rsi_slope < 0)
                    and (price < ema_long) and (vol_now > vol_ma) and short_align_5m
                )

                if ia_prob < 0.25:
                    bullish_ok = bearish_ok = False

                # Puntuación ML-lite
                features = {
                    "rsi": rsi,
                    "rsi_slope": rsi_slope,
                    "adx": adx_now,
                    "ema_trend": ema_f > ema_s,
                    "vol_rel": (vol_now / max(vol_ma, 1e-9)) - 1.0,
                    "regime": regime,
                }
                score = ml_score(features, ia_prob=ia_prob)
                if not (bullish_ok or bearish_ok) or score < ML_THRESHOLD:
                    print(f"⏸️ {SYMBOL} sin señal clara. ML={score:.2f} | regime={regime} | ia={ia_prob:.2%}")

                # =============== Gestión de posición abierta ===============
                flip_to = None
                if state.get("last_side") and state.get("entry_price") is not None:
                    side  = state["last_side"]
                    entry = state["entry_price"]

                    apply_breakeven_if_needed(state, side, price, SYMBOL)
                    update_trailing(state, side, price, SYMBOL)
                    exit_now, reason = should_exit_now(state, side, price)

                    # Flip inteligente
                    if not exit_now and adx_now >= FLIP_REQUIRE_ADX and ia_prob >= FLIP_MIN_IA:
                        if side == "LONG" and bearish_ok:
                            flip_to = "SHORT"
                            reason = "Flip: señal contraria fuerte"
                            exit_now = True
                        elif side == "SHORT" and bullish_ok:
                            flip_to = "LONG"
                            reason = "Flip: señal contraria fuerte"
                            exit_now = True

                    # Salida o Flip
                    if exit_now:
                        pnl_pct = current_profit_pct(side, entry, price)
                        send_signal(SYMBOL, SIGNAL_CODES[SYMBOL]["EXIT_ALL"])
                        log_trade(SYMBOL, side, entry, price, pnl_pct, reason)
                        send_telegram_message(f"🔚 {SYMBOL} EXIT {side} @ {entry:.2f} → {price:.2f} | PnL={pnl_pct:.2f}% ({reason})")

                        # limpiar estado
                        state.update({
                            "last_side": None,
                            "entry_price": None,
                            "trail_price": None,
                            "sl_price": None,
                            "breakeven_active": False,
                            "highest_price": None,
                            "lowest_price": None,
                            "dynamic_trail_pct": None,
                            "cooldown_until": time.time() + (FLIP_COOLDOWN_SEC if flip_to else COOLDOWN_AFTER_EXIT_SEC),
                        })
                        save_state(SYMBOL, state)

                        # Flip inmediato
                        if flip_to:
                            time.sleep(FLIP_COOLDOWN_SEC)
                            sl_flip = (price - ATR_SL_MULT * atr_fast) if flip_to == "LONG" else (price + ATR_SL_MULT * atr_fast)
                            code = SIGNAL_CODES[SYMBOL]["ENTER_LONG"] if flip_to == "LONG" else SIGNAL_CODES[SYMBOL]["ENTER_SHORT"]
                            if send_signal(SYMBOL, code):
                                record_entry(SYMBOL, flip_to, price, sl_flip, ema_f, ema_s, ema_long, rsi, atr_fast)
                                send_telegram_message(f"🔁 {SYMBOL} FLIP → ENTER {flip_to} @ {price:.2f} | SL={sl_flip:.2f} | IA={ia_prob*100:.1f}%")
                                print(f"🔁 {SYMBOL} FLIP → ENTER {flip_to} @ {price:.2f}", flush=True)
                        continue

                # =============== Entrada nueva si no hay posición ===============
                state = load_state(SYMBOL)
                if not state.get("last_side") or state.get("entry_price") is None:
                    if not (bullish_ok or bearish_ok) or score < ML_THRESHOLD:
                        time.sleep(0.01)
                        continue

                    desired_side = "LONG" if bullish_ok else "SHORT"
                    now_ts = time.time()

                    if state.get("cooldown_until", 0) > now_ts:
                        print(f"⏸️ {SYMBOL} en cooldown.")
                        continue

                    sl_price = (price - ATR_SL_MULT * atr_fast) if desired_side == "LONG" else (price + ATR_SL_MULT * atr_fast)
                    enter_code = SIGNAL_CODES[SYMBOL]["ENTER_LONG"] if desired_side == "LONG" else SIGNAL_CODES[SYMBOL]["ENTER_SHORT"]

                    if send_signal(SYMBOL, enter_code):
                        record_entry(SYMBOL, desired_side, price, sl_price, ema_f, ema_s, ema_long, rsi, atr_fast)
                        state = load_state(SYMBOL)
                        send_telegram_message(f"✅ {SYMBOL} ENTER {desired_side} @ {price:.2f} | SL={sl_price:.2f} | ADX={adx_now:.1f} | IA={ia_prob*100:.1f}% | Regime={regime}")
                        print(f"✅ {SYMBOL} ENTER {desired_side} @ {price:.2f} | SL={sl_price:.2f} | ADX={adx_now:.1f} | IA={ia_prob*100:.1f}% | Regime={regime}", flush=True)
                    else:
                        print(f"⚠️ {SYMBOL} NO se pudo enviar la señal ({desired_side}).", flush=True)
                        send_telegram_message(f"⚠️ {SYMBOL} NO se pudo enviar la señal ({desired_side}).")

            time.sleep(POLL_SECONDS)

        except Exception as e:
            consecutive_fetch_errors += 1
            if consecutive_fetch_errors in (3, 10, 20):
                send_telegram_message(f"⚠️ Error repetido de datos ({consecutive_fetch_errors}): {e}")
            print("⚠️ Error general:", e, flush=True)
            time.sleep(15)

# ===============================
# 🌐 Servidor HTTP unificado
# ===============================
class UnifiedHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # --- 1️⃣ IP pública ---
        if self.path == "/ip":
            try:
                ip = requests.get("https://ifconfig.me", timeout=5).text.strip()
            except Exception:
                ip = os.popen("hostname -I").read().strip() or "unknown"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(ip.encode("utf-8"))

        # --- 2️⃣ Estado del bot (posición abierta) ---
        elif self.path == "/state":
            state_file = state_path("BTCUSDT")
            if os.path.exists(state_file):
                with open(state_file, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No state_BTCUSDT.json yet.")

        # --- 3️⃣ Log de operaciones ---
        elif self.path == "/trades":
            csv_path = "/mnt/data/trades_log.csv"
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"No trades_log.csv yet.")

        # --- 4️⃣ Página raíz ---
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write("✅ Bot IA online - Endpoints disponibles: /ip /state /trades".encode("utf-8"))

# ===============================
# 🚀 Lanzar servidor en hilo paralelo
# ===============================
def start_http_server():
    PORT = 8080
    try:
        with socketserver.TCPServer(("", PORT), UnifiedHandler) as httpd:
            print(f"🌐 Servidor HTTP escuchando en puerto {PORT} (/ip, /state, /trades disponibles)", flush=True)
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️ Puerto {PORT} ya en uso, se omite iniciar nuevo servidor HTTP.")
        else:
            print(f"⚠️ Error iniciando servidor HTTP: {e}", flush=True)


# ===============================
# 🧠 Punto de entrada principal
# ===============================
if __name__ == "__main__":
    print("Binance OK, iniciando v6 ULTIMATE...", flush=True)
    auto_train_ai_model()

    # Iniciar servidor HTTP en hilo paralelo
    threading.Thread(target=start_http_server, daemon=True).start()

    # Iniciar bot principal
    main()


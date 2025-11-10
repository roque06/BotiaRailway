import requests
import numpy as np


def get_market_sentiment():
    """
    Retorna un puntaje de sentimiento global (0-1):
    1 = muy optimista, 0 = muy pesimista.
    """
    try:
        # Ejemplo: usar Fear & Greed Index (API gratuita)
        url = "https://api.alternative.me/fng/?limit=1"
        data = requests.get(url, timeout=10).json()
        value = float(data["data"][0]["value"])
        score = value / 100.0  # normalizar entre 0 y 1
        return round(score, 2)
    except Exception:
        # fallback aleatorio si falla conexión
        return np.random.uniform(0.4, 0.6)


def market_okay():
    """
    Devuelve True si el sentimiento del mercado permite operar.
    """
    score = get_market_sentiment()
    if score < 0.3:
        print(f"🌪️ Sentimiento bajista global ({score}) → evitar LONGS.")
        return False
    elif score > 0.8:
        print(f"🚀 Sentimiento eufórico global ({score}) → cuidado con reversión.")
        return False
    else:
        print(f"🧭 Sentimiento neutro ({score}), condiciones normales.")
        return True

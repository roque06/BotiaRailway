# ======================================
#  Dockerfile ÚNICO para BOT IA BTCUSDT
#  (todo integrado, sin requirements.txt)
# ======================================

FROM python:3.11-slim

# Crear carpeta de trabajo
WORKDIR /app

# Copiar todo el código local (bot + IA_MODEL + CSVs)
COPY . .

# Instalar dependencias necesarias
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    requests==2.31.0 \
    ta==0.11.0 \
    joblib==1.4.2 \
    scikit-learn==1.3.2 \
    python-telegram-bot==20.7

# Crear ruta persistente (si Railway asigna un volumen /mnt/data)
RUN mkdir -p /mnt/data

# Puerto opcional del HTTP interno (para endpoint /ip)
EXPOSE 8080

# Comando principal
CMD ["python", "WunderPython/smart_trading_bot.py"]

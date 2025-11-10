# ============================================
#  Dockerfile para WunderPython Bot con IA
# ============================================

FROM python:3.11-slim

# Crear carpeta de trabajo
WORKDIR /app

# Copiar TODO el proyecto (incluye AI_MODEL y WunderPython)
COPY . /app/

# Instalar dependencias necesarias
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    requests==2.31.0 \
    ta==0.11.0 \
    joblib==1.4.2 \
    scikit-learn==1.3.2 \
    python-telegram-bot==20.7

# Crear ruta persistente
RUN mkdir -p /mnt/data

# Exponer el puerto del servidor interno (/ip)
EXPOSE 8080

# Iniciar el bot
CMD ["python", "WunderPython/smart_trading_bot.py"]

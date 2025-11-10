# ============================================
#  Dockerfile completo para Railway + IA Model
# ============================================

FROM python:3.11-slim

# Crear carpeta de trabajo
WORKDIR /app

# Copiar TODO el proyecto, incluyendo WunderPython y AI_MODEL
COPY . /app/

# Instalar dependencias del bot
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    requests==2.31.0 \
    ta==0.11.0 \
    joblib==1.4.2 \
    scikit-learn==1.3.2 \
    python-telegram-bot==20.7

# Crear ruta persistente (Railway monta /mnt/data)
RUN mkdir -p /mnt/data

# Exponer el puerto del servidor interno /ip
EXPOSE 8080

# Comando de arranque
CMD ["python", "WunderPython/smart_trading_bot.py"]

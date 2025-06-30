FROM python:3.10-slim

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Crea cartella di lavoro
WORKDIR /app

# Copia i file nel container
COPY requirements.txt .
COPY vision_server.py .


# Installa le dipendenze Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Espone la porta usata da Flask
EXPOSE 5000

# Avvia il server Flask
CMD ["python", "vision_server.py"]

from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
from pathlib import Path
import requests
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# === CARICAMENTO MODELLO DA HUGGING FACE (senza ?download=true) ===
model_url = "https://huggingface.co/Alecet/yolov8n/resolve/main/yolov8n.pt"
model_path = Path("yolov8n.pt")

if not model_path.exists():
    print("Scarico il modello da Hugging Face...")
    response = requests.get(model_url)
    model_path.write_bytes(response.content)

# === CARICAMENTO MODELLO YOLOv8 ===
model = YOLO(str(model_path))

# === ENDPOINT /predict ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'Image not found in request'}), 400

    # Decodifica immagine base64
    image_data = data['image']
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Inference
    results = model(image)

    # Estrazione risultati
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = float(box.conf[0])
        detections.append({
            "label": label,
            "confidence": round(confidence, 3)
        })

    return jsonify({"detections": detections})

# === AVVIO LOCALE ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
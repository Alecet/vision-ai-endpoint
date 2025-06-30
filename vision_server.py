
import os
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# URL del modello caricato su Hugging Face
MODEL_URL = "https://huggingface.co/Alecet/yolov8n/resolve/main/yolov8n.pt"
MODEL_PATH = "yolov8n.pt"

# Scarica il modello se non esiste
if not os.path.exists(MODEL_PATH):
    print("Scaricamento modello YOLOv8n da Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Modello scaricato correttamente.")

# Caricamento del modello YOLOv8n
model = YOLO(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nessuna immagine inviata."}), 400

    image_file = request.files["image"]
    image_path = "temp_image.jpg"
    image_file.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.xyxy.tolist()

    os.remove(image_path)

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

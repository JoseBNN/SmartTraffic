from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
import cv2
import numpy as np
import base64
import uuid
import os

app = FastAPI()

# =========================
#  MONTAR CARPETAS STATIC
# =========================
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# =========================
#     MODELOS YOLO
# =========================

# Modelo de vehículos
car_model = YOLO("best_.pt")

# Modelo de carriles (segmentación)
lane_model = YOLO("best.pt")

# =========================
#   SERVIR FRONTEND
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# =========================
#   PREDICCIÓN UNIFICADA
# =========================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # ----------- Guardar imagen subida -----------
    img_id = str(uuid.uuid4())
    upload_path = f"uploads/{img_id}.jpg"

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # Leer imagen en OpenCV
    img = cv2.imread(upload_path)

    # ----------- PROCESO 1: DETECCIÓN VEHÍCULOS -----------
    car_res = car_model(img)[0]
    car_annot = car_res.plot()   # imagen con cajas de carros

    # ----------- PROCESO 2: DETECCIÓN CARRILES -----------
    lane_res = lane_model(img)[0]
    lane_annot = lane_res.plot()  # imagen con segmentación de carril

    # ----------- COMBINAR DETECCIONES -----------
    final = cv2.addWeighted(car_annot, 0.7, lane_annot, 0.7, 0)

    # ----------- ANALIZAR INVASIÓN -----------
    invasion = False
    estado = "Carril libre"

    if lane_res.masks is not None and len(lane_res.masks.data) > 0:

        # Tomar la primera máscara de carril
        mask = lane_res.masks.data[0].cpu().numpy().astype(np.uint8)

        # Ajustar máscara a tamaño original
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Revisar si alguna caja de carro invade la máscara
        for box in car_res.boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = map(int, box)

            # Asegurar límites dentro de la imagen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(mask.shape[1], x2)
            y2 = min(mask.shape[0], y2)

            crop = mask[y1:y2, x1:x2]

            if np.any(crop > 0):
                invasion = True
                estado = "Carril invadido"
                break

    # ----------- GUARDAR RESULTADO FINAL -----------
    result_path = f"results/{img_id}.png"
    cv2.imwrite(result_path, final)

    # Convertir a Base64 para mostrar directo en frontend
    _, buffer = cv2.imencode(".png", final)
    img_base64 = base64.b64encode(buffer).decode()

    # ----------- RESPUESTA JSON -----------
    return {
        "estado_carril": estado,
        "invasion": invasion,
        "processed_image": result_path,     # ruta por si deseas mostrar directo
        "imagen_base64": img_base64         # base64 para usar en <img>
    }

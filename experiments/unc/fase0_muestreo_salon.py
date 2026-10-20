"""
Fase 0 — Muestreo de caras del salón para anotación manual.

Toma 1 frame por segundo de cada video del salón, detecta caras con RetinaFace
y guarda los recortes con sus metadatos. NO reconoce identidades: las etiquetas
se asignan después por revisión humana/visual (evita el ground truth circular).

Salida:
  experiments/unc/salon_muestra/crops/<recorte_id>.jpg
  experiments/unc/salon_muestra/recortes.csv
"""

import csv
import os
import sys
from pathlib import Path

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector

RAIZ = Path(__file__).resolve().parents[2]
VIDEOS = [
    RAIZ / "data/unc/salon/VID_20260907_102851.mp4",
]
SALIDA = RAIZ / "experiments/unc/salon_muestra"
FPS_MUESTREO = 1.0
MARGEN = 0.2
TAMANO_MIN_CRUDO = 20


def recortar(frame, x1, y1, x2, y2, margen=MARGEN):
    h, w = frame.shape[:2]
    ancho, alto = x2 - x1, y2 - y1
    mx, my = int(ancho * margen), int(alto * margen)
    a, b = max(0, x1 - mx), max(0, y1 - my)
    c, d = min(w, x2 + mx), min(h, y2 + my)
    return frame[b:d, a:c]


def muestrear(detector, ruta_video, prefijo, filas, carpeta_crops):
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    intervalo = max(1, round(fps / FPS_MUESTREO))
    frame_idx = 0
    contador = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % intervalo == 0:
            for rostro in detector.detect(frame):
                x1, y1, x2, y2 = map(int, rostro.bbox)
                ancho = x2 - x1
                if ancho < TAMANO_MIN_CRUDO:
                    continue
                recorte = recortar(frame, x1, y1, x2, y2)
                if recorte.size == 0:
                    continue
                contador += 1
                recorte_id = f"{prefijo}_{contador:05d}"
                cv2.imwrite(str(carpeta_crops / f"{recorte_id}.jpg"), recorte)
                filas.append({
                    "recorte_id": recorte_id,
                    "video": ruta_video.name,
                    "frame_idx": frame_idx,
                    "segundo": round(frame_idx / fps, 2),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "ancho": ancho,
                    "confianza_det": round(float(rostro.confidence), 4),
                })
        frame_idx += 1

    cap.release()
    print(f"{ruta_video.name}: {frame_idx} frames, {contador} recortes")


def main():
    carpeta_crops = SALIDA / "crops"
    carpeta_crops.mkdir(parents=True, exist_ok=True)

    detector = create_detector("retinaface")
    filas = []
    for i, video in enumerate(VIDEOS, start=1):
        muestrear(detector, video, f"v{i}", filas, carpeta_crops)

    with open(SALIDA / "recortes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    print(f"Total recortes: {len(filas)} -> {SALIDA / 'recortes.csv'}")


if __name__ == "__main__":
    main()

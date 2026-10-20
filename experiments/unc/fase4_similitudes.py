"""
Fase 4 — Similitud de cada recorte etiquetado contra los 9 centroides.

Para cada recorte de la muestra (300), busca su detección en el frame del video,
calcula el embedding (mismo preprocesado que el motor) y guarda la similitud
contra cada uno de los 9 estudiantes.

Salida: experiments/unc/fase4/fase4_similitudes.csv
"""

import csv
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector, create_recognizer
from core.utils_facial import embedding_facial

RAIZ = Path(__file__).resolve().parents[2]
MUESTRA = RAIZ / "experiments/unc/salon_muestra/muestra_anotacion.csv"
RECORTES = RAIZ / "experiments/unc/salon_muestra/recortes.csv"
DB = RAIZ / "data/database_embeddings.pkl"
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
SALIDA = RAIZ / "experiments/unc/fase4"
UMBRAL_IOU = 0.5


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    with open(DB, "rb") as f:
        db = pickle.load(f)
    nombres = sorted(db)
    centroides = np.array([np.asarray(db[n]).reshape(-1) for n in nombres])

    with open(RECORTES, encoding="utf-8") as f:
        recortes = {r["recorte_id"]: r for r in csv.DictReader(f)}
    with open(MUESTRA, encoding="utf-8") as f:
        muestra = list(csv.DictReader(f))

    por_frame = defaultdict(list)
    for fila in muestra:
        r = recortes[fila["recorte_id"]]
        por_frame[int(r["frame_idx"])].append((fila, (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))))

    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    filas = []
    cap = cv2.VideoCapture(str(VIDEO))
    indice = 0
    ultimo = max(por_frame)
    while indice <= ultimo:
        ok, frame = cap.read()
        if not ok:
            break
        if indice in por_frame:
            caras = detector.detect(frame)
            cajas = [tuple(int(v) for v in c.bbox) for c in caras]
            for fila, caja in por_frame[indice]:
                if not cajas:
                    continue
                ious = [iou(caja, b) for b in cajas]
                k = int(np.argmax(ious))
                if ious[k] < UMBRAL_IOU:
                    continue
                vec = embedding_facial(recognizer, frame, caras[k].landmarks)
                sims = centroides @ vec
                registro = {"recorte_id": fila["recorte_id"], "etiqueta": fila["etiqueta"]}
                for n, s in zip(nombres, sims):
                    registro[f"sim_{n}"] = round(float(s), 4)
                filas.append(registro)
        indice += 1
    cap.release()

    with open(SALIDA / "fase4_similitudes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    print(f"Recortes con similitudes: {len(filas)}")
    print(f"Archivo: {SALIDA / 'fase4_similitudes.csv'}")


if __name__ == "__main__":
    main()

"""
Fase 0 — Reconocimiento sobre la muestra del salón y comparación con las etiquetas.

Para cada recorte emparejado (emparejamiento_iou.csv):
  1. Detecta la cara en su frame para obtener landmarks.
  2. Calcula el embedding con embedding_facial (mismo preprocesado que el motor).
  3. Compara contra los 9 centroides (producto punto).
  4. Decide: mejor similitud > 0.35 -> nombre; si no -> Desconocido.
  5. Clasifica contra la etiqueta humana.

Salida:
  experiments/unc/salon_muestra/reconocimiento_salon.csv
"""

import ast
import csv
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector, create_recognizer
from core.utils_facial import embedding_facial

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DIR = RAIZ / "experiments/unc/salon_muestra"
EMPAREJADOS = SALIDA_DIR / "emparejamiento_iou.csv"
MUESTRA = SALIDA_DIR / "muestra_anotacion.csv"
DB = RAIZ / "data/database_embeddings.pkl"
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
UMBRAL = 0.35
PERSONAS = {"alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"}


def clasificar(etiqueta, decision):
    if etiqueta in PERSONAS:
        if decision == etiqueta:
            return "acierto"
        if decision == "Desconocido":
            return "falso_negativo"
        return "error_identidad"
    if etiqueta == "desconocido":
        return "acierto" if decision == "Desconocido" else "falso_positivo"
    return "no_evaluable"


def main():
    with open(DB, "rb") as f:
        db = pickle.load(f)
    nombres = sorted(db)
    centroides = np.array([np.asarray(db[n]).reshape(-1) for n in nombres])

    with open(MUESTRA, encoding="utf-8") as f:
        etiquetas = {r["recorte_id"]: r["etiqueta"] for r in csv.DictReader(f)}

    with open(EMPAREJADOS, encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["emparejado"] == "True"]
    for r in filas:
        r["etiqueta"] = etiquetas[r["recorte_id"]]

    por_frame = defaultdict(list)
    for r in filas:
        por_frame[int(r["frame_idx"])].append(r)

    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")
    salida = []

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
            for r in por_frame[indice]:
                caja_det = ast.literal_eval(r["caja_deteccion"])
                k = cajas.index(caja_det) if caja_det in cajas else None
                if k is None:
                    continue
                vec = embedding_facial(recognizer, frame, caras[k].landmarks)
                sims = centroides @ vec
                orden = np.argsort(-sims)
                mejor, segunda = orden[0], orden[1]
                sim_mejor = float(sims[mejor])
                decision = nombres[mejor] if sim_mejor > UMBRAL else "Desconocido"
                salida.append({
                    "recorte_id": r["recorte_id"],
                    "etiqueta": r["etiqueta"],
                    "frame_idx": indice,
                    "mejor_nombre": nombres[mejor],
                    "sim_mejor": round(sim_mejor, 4),
                    "segunda_nombre": nombres[segunda],
                    "sim_segunda": round(float(sims[segunda]), 4),
                    "decision": decision,
                    "resultado": clasificar(r["etiqueta"], decision),
                })
        indice += 1
    cap.release()

    with open(SALIDA_DIR / "reconocimiento_salon.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(salida[0].keys()))
        writer.writeheader()
        writer.writerows(salida)

    print(f"Recortes procesados: {len(salida)}")
    print("Resultados globales:", dict(Counter(s["resultado"] for s in salida)))
    print("\nPor etiqueta (decision del sistema):")
    tabla = defaultdict(Counter)
    for s in salida:
        tabla[s["etiqueta"]][s["decision"]] += 1
    for etiqueta in sorted(tabla, key=lambda e: -sum(tabla[e].values())):
        print(f"  {etiqueta:12s} {dict(tabla[etiqueta])}")


if __name__ == "__main__":
    main()

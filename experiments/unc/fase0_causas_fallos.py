"""
Fase 0 — Posibles causas de los falsos negativos.

Para cada recorte con nombre (persona registrada) mide:
  - ancho de la caja (px)
  - nitidez del recorte (varianza del Laplaciano)
  - pose: desviación de la nariz respecto al punto medio de los ojos, dividida
    entre la distancia entre ojos (0 = de frente; mayor = más de perfil)
Luego compara aciertos contra falsos negativos.

Salida:
  experiments/unc/salon_muestra/causas_fallos.csv
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DIR = RAIZ / "experiments/unc/salon_muestra"
RECONOCIMIENTO = SALIDA_DIR / "reconocimiento_salon.csv"
RECORTES = SALIDA_DIR / "recortes.csv"
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
CROPS = SALIDA_DIR / "crops"
PERSONAS = {"alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"}


def nitidez(ruta):
    gris = cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def pose(landmarks):
    ojo_izq, ojo_der, nariz = landmarks[0], landmarks[1], landmarks[2]
    dist = np.linalg.norm(np.asarray(ojo_der) - np.asarray(ojo_izq))
    centro = (np.asarray(ojo_izq) + np.asarray(ojo_der)) / 2
    return float(abs(nariz[0] - centro[0]) / dist) if dist > 0 else None


def main():
    with open(RECONOCIMIENTO, encoding="utf-8") as f:
        rec = {r["recorte_id"]: r for r in csv.DictReader(f)}
    with open(RECORTES, encoding="utf-8") as f:
        info = {r["recorte_id"]: r for r in csv.DictReader(f)}

    evaluables = {k: v for k, v in rec.items() if v["etiqueta"] in PERSONAS}
    por_frame = defaultdict(list)
    for rid, v in evaluables.items():
        por_frame[int(v["frame_idx"])].append(rid)

    detector = create_detector("retinaface")
    poses = {}
    cap = cv2.VideoCapture(str(VIDEO))
    indice = 0
    while indice <= max(por_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if indice in por_frame:
            caras = detector.detect(frame)
            cajas = [tuple(int(v) for v in c.bbox) for c in caras]
            for rid in por_frame[indice]:
                caja = (int(info[rid]["x1"]), int(info[rid]["y1"]), int(info[rid]["x2"]), int(info[rid]["y2"]))
                if caja in cajas:
                    poses[rid] = pose(caras[cajas.index(caja)].landmarks)
        indice += 1
    cap.release()

    filas = []
    for rid, v in evaluables.items():
        filas.append({
            "recorte_id": rid,
            "etiqueta": v["etiqueta"],
            "resultado": v["resultado"],
            "ancho": int(info[rid]["ancho"]),
            "nitidez": round(nitidez(CROPS / f"{rid}.jpg"), 1),
            "pose": round(poses[rid], 3) if rid in poses and poses[rid] is not None else "",
        })

    with open(SALIDA_DIR / "causas_fallos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    grupos = {"acierto": [], "falso_negativo": []}
    for f_ in filas:
        if f_["resultado"] in grupos and f_["pose"] != "":
            grupos[f_["resultado"]].append(f_)

    print("Mediana por grupo:")
    for g, lista in grupos.items():
        print(f"  {g:15s} n={len(lista):3d} ancho={statistics.median([x['ancho'] for x in lista]):5.1f} "
              f"nitidez={statistics.median([x['nitidez'] for x in lista]):7.1f} "
              f"pose={statistics.median([x['pose'] for x in lista]):.3f}")

    print("\nTasa de fallo por pose (persona registrada):")
    rangos = [(0, 0.1, "frente (<0.10)"), (0.1, 0.2, "leve (0.10-0.20)"), (0.2, 9, "perfil (>=0.20)")]
    for lo, hi, nombre in rangos:
        en = [x for x in filas if x["pose"] != "" and lo <= x["pose"] < hi and x["resultado"] in ("acierto", "falso_negativo")]
        fn = sum(1 for x in en if x["resultado"] == "falso_negativo")
        if en:
            print(f"  {nombre:18s} casos={len(en):3d} falsos negativos={fn:3d} tasa de fallo={fn / len(en):.1%}")

    print("\nTasa de fallo por ancho:")
    for lo, hi, nombre in [(0, 30, "<30 px"), (30, 40, "30-39 px"), (40, 60, "40-59 px"), (60, 999, ">=60 px")]:
        en = [x for x in filas if lo <= x["ancho"] < hi and x["resultado"] in ("acierto", "falso_negativo")]
        fn = sum(1 for x in en if x["resultado"] == "falso_negativo")
        if en:
            print(f"  {nombre:10s} casos={len(en):3d} falsos negativos={fn:3d} tasa de fallo={fn / len(en):.1%}")


if __name__ == "__main__":
    main()

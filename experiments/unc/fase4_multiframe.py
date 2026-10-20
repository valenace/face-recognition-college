"""
Fase 4b — Comparación de umbrales con decisión de 1 frame vs 3 frames consecutivos.

Para cada recorte etiquetado se toman 3 frames consecutivos (el del recorte y los dos
siguientes), se sigue la cara por IoU y se calcula su similitud con los 9 centroides.

Dos reglas de decisión:
  - 1 frame: gana el argmax; se acepta si su similitud >= umbral.
  - 3 frames: se acepta solo si el mismo candidato gana y supera el umbral en los 3
    frames (la regla de votos del motor).

Salida: experiments/unc/fase4/fase4_multiframe.csv
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
UMBRALES = [0.28, 0.30, 0.35]
PERSONAS = ["alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"]
IOU_INICIAL = 0.5
IOU_SEGUIMIENTO = 0.3


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decidir(sims, umbral):
    k = int(np.argmax(sims))
    return (PERSONAS[k], float(sims[k])) if sims[k] >= umbral else ("Desconocido", float(sims[k]))


def main():
    with open(DB, "rb") as f:
        db = pickle.load(f)
    nombres = sorted(db)
    assert nombres == PERSONAS
    centroides = np.array([np.asarray(db[n]).reshape(-1) for n in nombres])

    with open(RECORTES, encoding="utf-8") as f:
        recortes = {r["recorte_id"]: r for r in csv.DictReader(f)}
    with open(MUESTRA, encoding="utf-8") as f:
        muestra = [r for r in csv.DictReader(f) if r["etiqueta"] in PERSONAS or r["etiqueta"] == "desconocido"]

    pedidos = defaultdict(list)
    for fila in muestra:
        r = recortes[fila["recorte_id"]]
        f0 = int(r["frame_idx"])
        pedidos[f0].append((fila["recorte_id"], fila["etiqueta"],
                            (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))))

    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    # series[recorte_id] = lista de (frame, [9 similitudes]) para los 3 frames
    series = defaultdict(list)
    necesarios = set()
    for f0 in pedidos:
        for d in range(3):
            necesarios.add(f0 + d)

    cap = cv2.VideoCapture(str(VIDEO))
    indice = 0
    ultimo = max(necesarios)
    cajas_previas = {}
    while indice <= ultimo:
        ok, frame = cap.read()
        if not ok:
            break
        if indice in necesarios:
            caras = detector.detect(frame)
            cajas = [tuple(int(v) for v in c.bbox) for c in caras]
            for recorte_id, etiqueta, caja0 in pedidos.get(indice, []):
                if not cajas:
                    continue
                ious = [iou(caja0, b) for b in cajas]
                k = int(np.argmax(ious))
                if ious[k] < IOU_INICIAL:
                    continue
                cajas_previas[recorte_id] = cajas[k]
                vec = embedding_facial(recognizer, frame, caras[k].landmarks)
                series[recorte_id].append((indice, centroides @ vec))
            # seguimiento en los frames +1 y +2
            for f0, lista in pedidos.items():
                if f0 + 1 <= indice <= f0 + 2:
                    for recorte_id, etiqueta, _ in lista:
                        previa = cajas_previas.get(recorte_id)
                        if previa is None or not cajas:
                            continue
                        ious = [iou(previa, b) for b in cajas]
                        k = int(np.argmax(ious))
                        if ious[k] < IOU_SEGUIMIENTO:
                            continue
                        cajas_previas[recorte_id] = cajas[k]
                        vec = embedding_facial(recognizer, frame, caras[k].landmarks)
                        series[recorte_id].append((indice, centroides @ vec))
        indice += 1
    cap.release()

    # Evaluación
    resultados = {}
    for umbral in UMBRALES:
        tp1 = fp1 = fn1 = tn1 = 0
        tp3 = fp3 = fn3 = tn3 = 0
        for recorte_id, etiqueta, _ in [(r["recorte_id"], r["etiqueta"], recortes[r["recorte_id"]]) for r in muestra]:
            serie = sorted(series.get(recorte_id, []))
            if not serie:
                continue
            # 1 frame
            dec1, _ = decidir(serie[0][1], umbral)
            # 3 frames
            if len(serie) == 3:
                decisiones = [decidir(s, umbral)[0] for _, s in serie]
                dec3 = decisiones[0] if len(set(decisiones)) == 1 and decisiones[0] != "Desconocido" else "Desconocido"
            else:
                dec3 = "Desconocido"

            if etiqueta in PERSONAS:
                if dec1 == etiqueta: tp1 += 1
                else: fn1 += 1
                if dec3 == etiqueta: tp3 += 1
                else: fn3 += 1
            else:  # desconocido
                if dec1 == "Desconocido": tn1 += 1
                else: fp1 += 1
                if dec3 == "Desconocido": tn3 += 1
                else: fp3 += 1

        def metricas(tp, fp, fn):
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            return p, r, f1

        p1, r1, f1_1 = metricas(tp1, fp1, fn1)
        p3, r3, f1_3 = metricas(tp3, fp3, fn3)
        resultados[umbral] = {
            "umbral": umbral,
            "1f_TP": tp1, "1f_FP": fp1, "1f_FN": fn1, "1f_P": round(p1, 3), "1f_R": round(r1, 3), "1f_F1": round(f1_1, 3),
            "3f_TP": tp3, "3f_FP": fp3, "3f_FN": fn3, "3f_P": round(p3, 3), "3f_R": round(r3, 3), "3f_F1": round(f1_3, 3),
        }

    with open(SALIDA / "fase4_multiframe.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[UMBRALES[0]].keys()))
        writer.writeheader()
        writer.writerows(resultados.values())

    print(f"{'umbral':>7} | {'1 frame: P R F1':>22} | {'3 frames: P R F1':>23}")
    for u, r in resultados.items():
        print(f"{u:7.2f} | {r['1f_P']:.3f} {r['1f_R']:.3f} {r['1f_F1']:.3f}   {r['1f_TP']:3d}TP {r['1f_FP']:2d}FP {r['1f_FN']:3d}FN | "
              f"{r['3f_P']:.3f} {r['3f_R']:.3f} {r['3f_F1']:.3f}   {r['3f_TP']:3d}TP {r['3f_FP']:2d}FP {r['3f_FN']:3d}FN")

    completos = sum(1 for r in series.values() if len(r) == 3)
    print(f"\nRecortes con los 3 frames: {completos} de {len(muestra)}")


if __name__ == "__main__":
    main()

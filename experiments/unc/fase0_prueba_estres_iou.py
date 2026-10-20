"""
Fase 0 — Prueba de estrés del emparejamiento por IoU.

Mueve al azar la caja de cada recorte (semilla fija) y comprueba si el emparejamiento
sigue eligiendo la misma detección del frame que sin perturbar.

Salida:
  experiments/unc/salon_muestra/prueba_estres_iou.csv
"""

import csv
import os
import random
import sys
from pathlib import Path

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DIR = RAIZ / "experiments/unc/salon_muestra"
MUESTRA = SALIDA_DIR / "muestra_anotacion.csv"
RECORTES = SALIDA_DIR / "recortes.csv"
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
PERTURBACIONES = 20
MAX_DESPLAZAMIENTO = 0.10
SEMILLA = 42


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def perturbar(caja, rng):
    x1, y1, x2, y2 = caja
    ancho, alto = x2 - x1, y2 - y1
    return (
        x1 + rng.uniform(-MAX_DESPLAZAMIENTO, MAX_DESPLAZAMIENTO) * ancho,
        y1 + rng.uniform(-MAX_DESPLAZAMIENTO, MAX_DESPLAZAMIENTO) * alto,
        x2 + rng.uniform(-MAX_DESPLAZAMIENTO, MAX_DESPLAZAMIENTO) * ancho,
        y2 + rng.uniform(-MAX_DESPLAZAMIENTO, MAX_DESPLAZAMIENTO) * alto,
    )


def main():
    with open(RECORTES, encoding="utf-8") as f:
        recortes = {r["recorte_id"]: r for r in csv.DictReader(f)}
    with open(MUESTRA, encoding="utf-8") as f:
        muestra = list(csv.DictReader(f))

    por_frame = {}
    for fila in muestra:
        r = recortes[fila["recorte_id"]]
        caja = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        por_frame.setdefault(int(fila["frame_idx"]), []).append((fila, caja))

    detector = create_detector("retinaface")
    rng = random.Random(SEMILLA)
    filas_salida = []

    cap = cv2.VideoCapture(str(VIDEO))
    indice = 0
    ultimo = max(por_frame)
    while indice <= ultimo:
        ok, frame = cap.read()
        if not ok:
            break
        if indice in por_frame:
            cajas = [tuple(int(v) for v in c.bbox) for c in detector.detect(frame)]
            for fila, caja in por_frame[indice]:
                if not cajas:
                    continue
                ious_orig = [iou(caja, b) for b in cajas]
                correcta = max(range(len(cajas)), key=lambda k: ious_orig[k])
                aciertos = 0
                ious_elegidas = []
                for _ in range(PERTURBACIONES):
                    p = perturbar(caja, rng)
                    ious_p = [iou(p, b) for b in cajas]
                    elegida = max(range(len(cajas)), key=lambda k: ious_p[k])
                    if elegida == correcta:
                        aciertos += 1
                    ious_elegidas.append(ious_p[elegida])
                filas_salida.append({
                    "recorte_id": fila["recorte_id"],
                    "etiqueta": fila["etiqueta"],
                    "frame_idx": indice,
                    "caras_en_frame": len(cajas),
                    "aciertos": aciertos,
                    "perturbaciones": PERTURBACIONES,
                    "iou_mediano_elegido": round(sorted(ious_elegidas)[len(ious_elegidas) // 2], 4),
                })
        indice += 1
    cap.release()

    with open(SALIDA_DIR / "prueba_estres_iou.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_salida[0].keys()))
        writer.writeheader()
        writer.writerows(filas_salida)

    total = sum(r["perturbaciones"] for r in filas_salida)
    aciertos = sum(r["aciertos"] for r in filas_salida)
    print(f"Recortes evaluados: {len(filas_salida)}")
    print(f"Perturbaciones totales: {total}")
    print(f"Acierto global del emparejamiento: {aciertos / total:.4f}")
    sin_perfecto = [r for r in filas_salida if r["aciertos"] < r["perturbaciones"]]
    print(f"Recortes con algún fallo: {len(sin_perfecto)}")
    if sin_perfecto:
        peor = min(sin_perfecto, key=lambda r: r["aciertos"])
        print(f"Peor caso: {peor['recorte_id']} ({peor['etiqueta']}) acierto {peor['aciertos']}/{peor['perturbaciones']} con {peor['caras_en_frame']} caras en el frame")


if __name__ == "__main__":
    main()

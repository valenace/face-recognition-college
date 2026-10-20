"""
Fase 0 — Emparejar cada recorte etiquetado con la detección de su frame, usando IoU.

Para cada fila de muestra_anotacion.csv:
  1. Toma la caja original del recorte (recortes.csv: x1, y1, x2, y2).
  2. Lee el frame correspondiente del video del salón.
  3. Detecta todas las caras del frame con RetinaFace.
  4. Calcula el IoU entre la caja del recorte y cada detección.
  5. Se queda con la detección de mayor IoU. Si IoU >= 0.5, queda emparejada.

Salida:
  experiments/unc/salon_muestra/emparejamiento_iou.csv
"""

import csv
import os
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
UMBRAL_EMPAREJADO = 0.5


def iou(a, b):
    """IoU entre dos cajas (x1, y1, x2, y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cargar_muestra():
    with open(MUESTRA, encoding="utf-8") as f:
        muestra = list(csv.DictReader(f))
    with open(RECORTES, encoding="utf-8") as f:
        recortes = {r["recorte_id"]: r for r in csv.DictReader(f)}
    for fila in muestra:
        r = recortes[fila["recorte_id"]]
        fila["caja"] = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
    return muestra


def main():
    muestra = cargar_muestra()

    # Agrupamos por frame: así leemos cada frame una sola vez
    por_frame = {}
    for fila in muestra:
        por_frame.setdefault(int(fila["frame_idx"]), []).append(fila)

    detector = create_detector("retinaface")
    resultados = {}

    # Leemos el video en orden, sin saltos, para que el índice de frame sea exacto
    cap = cv2.VideoCapture(str(VIDEO))
    indice = 0
    ultimo_necesario = max(por_frame)
    while indice <= ultimo_necesario:
        ok, frame = cap.read()
        if not ok:
            break
        if indice in por_frame:
            caras = detector.detect(frame)
            cajas = [tuple(int(v) for v in c.bbox) for c in caras]
            confianzas = [float(c.confidence) for c in caras]

            for fila in por_frame[indice]:
                if not cajas:
                    resultados[fila["recorte_id"]] = (0.0, None, None, 0, fila)
                    continue
                ious = [iou(fila["caja"], b) for b in cajas]
                mejor = max(range(len(ious)), key=lambda k: ious[k])
                resultados[fila["recorte_id"]] = (
                    ious[mejor], cajas[mejor], confianzas[mejor], len(cajas), fila
                )
        indice += 1
    cap.release()

    with open(SALIDA_DIR / "emparejamiento_iou.csv", "w", newline="", encoding="utf-8") as f:
        campos = ["recorte_id", "etiqueta", "frame_idx", "caja_recorte", "iou_max",
                  "caja_deteccion", "confianza_det", "caras_en_frame", "emparejado"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in muestra:
            iou_max, caja_det, conf, n_caras, _ = resultados[fila["recorte_id"]]
            writer.writerow({
                "recorte_id": fila["recorte_id"],
                "etiqueta": fila["etiqueta"],
                "frame_idx": fila["frame_idx"],
                "caja_recorte": fila["caja"],
                "iou_max": round(iou_max, 4),
                "caja_deteccion": caja_det,
                "confianza_det": round(conf, 4) if conf is not None else "",
                "caras_en_frame": n_caras,
                "emparejado": iou_max >= UMBRAL_EMPAREJADO,
            })

    ious = [r[0] for r in resultados.values()]
    emparejados = sum(1 for v in ious if v >= UMBRAL_EMPAREJADO)
    print(f"Filas: {len(ious)}")
    print(f"Emparejadas (IoU >= {UMBRAL_EMPAREJADO}): {emparejados}")
    print(f"Sin emparejar: {len(ious) - emparejados}")
    print(f"IoU mínimo: {min(ious):.4f} | mediano: {sorted(ious)[len(ious)//2]:.4f} | máximo: {max(ious):.4f}")


if __name__ == "__main__":
    main()

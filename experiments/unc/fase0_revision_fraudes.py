"""
Revisión de las alarmas de FRAUDE del video completo (filtro 20 px).

Para cada pista con alguna decisión FRAUDE:
  - toma el primer frame de la alarma
  - recalcula el score de anti-spoofing sobre esa caja
  - guarda un recorte para revisión visual

Salida:
  experiments/unc/demo/revision_fraudes/*.jpg
  experiments/unc/demo/revision_fraudes.csv
  experiments/unc/demo/revision_fraudes_resumen.jpg
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface.spoofing import create_spoofer

RAIZ = Path(__file__).resolve().parents[2]
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
DECISIONES = RAIZ / "experiments/unc/demo/decisiones_0s_107s_filtro20.csv"
SALIDA = RAIZ / "experiments/unc/demo/revision_fraudes"


def main():
    with open(DECISIONES, encoding="utf-8") as f:
        decisiones = list(csv.DictReader(f))

    fraudes = defaultdict(list)
    for d in decisiones:
        if d["estado"] == "FRAUDE":
            fraudes[d["track_id"]].append(d)

    SALIDA.mkdir(parents=True, exist_ok=True)
    spoofer = create_spoofer()

    filas = []
    for track_id, eventos in sorted(fraudes.items(), key=lambda kv: -len(kv[1])):
        primero = eventos[0]
        frame_idx = int(primero["frame_video"])
        bbox = (int(primero["x1"]), int(primero["y1"]), int(primero["x2"]), int(primero["y2"]))

        cap = cv2.VideoCapture(str(VIDEO))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue

        res = spoofer.predict(frame, bbox)
        ancho = bbox[2] - bbox[0]
        alto = bbox[3] - bbox[1]
        recorte = frame[max(0, bbox[1] - 10):bbox[3] + 10, max(0, bbox[0] - 10):bbox[2] + 10]
        cv2.imwrite(str(SALIDA / f"pista_{track_id}_seg{float(primero['segundo']):.0f}.jpg"),
                    cv2.resize(recorte, (200, 200)))

        filas.append({
            "track_id": track_id,
            "alarmas": len(eventos),
            "primer_segundo": primero["segundo"],
            "ancho": ancho,
            "alto": alto,
            "es_real_modelo": bool(res.is_real),
            "confianza_real": round(float(res.confidence), 4),
            "umbral": 0.90,
            "frame_idx": frame_idx,
        })

    with open(SALIDA.parent / "revision_fraudes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    imagenes = []
    for fila in filas:
        ruta = SALIDA / f"pista_{fila['track_id']}_seg{float(fila['primer_segundo']):.0f}.jpg"
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        texto = f"pista {fila['track_id']} x{fila['alarmas']} | {fila['ancho']}px | real={fila['es_real_modelo']} conf={fila['confianza_real']}"
        cv2.putText(img, texto, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        imagenes.append(img)
    if imagenes:
        while len(imagenes) % 4:
            imagenes.append(np.full((200, 200, 3), 255, np.uint8))
        filas_img = [np.hstack(imagenes[i:i + 4]) for i in range(0, len(imagenes), 4)]
        cv2.imwrite(str(SALIDA.parent / "revision_fraudes_resumen.jpg"), np.vstack(filas_img))

    for fila in filas:
        print(fila)


if __name__ == "__main__":
    main()

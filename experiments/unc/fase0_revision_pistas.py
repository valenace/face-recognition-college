"""
Revisión de las pistas de seguimiento confirmadas como una persona.

Uso:
  venv/bin/python experiments/unc/fase0_revision_pistas.py [nombre] [filtro]

Toma las decisiones del video completo (decisiones_0s_107s_filtro<N>.csv),
agrupa las pistas confirmadas o ancladas del nombre elegido, y genera:
  - un CSV con el resumen por pista (duración, frames, posición, ancho)
  - una cuadrícula con un frame por pista (cara marcada con rectángulo)
para verificar visualmente si todas son la misma persona.
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAIZ = Path(__file__).resolve().parents[2]
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
DEMO = RAIZ / "experiments/unc/demo"
LADO = 380
COLUMNAS = 5


def main():
    nombre = sys.argv[1] if len(sys.argv) > 1 else "bob"
    filtro = sys.argv[2] if len(sys.argv) > 2 else "20"
    decisiones = DEMO / f"decisiones_0s_107s_filtro{filtro}.csv"

    por_pista = defaultdict(list)
    with open(decisiones, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["nombre"] == nombre and d["estado"] in ("CONFIRMADO", "ANCLADO"):
                por_pista[d["track_id"]].append(d)

    resumen = []
    for t_id, eventos in sorted(por_pista.items(), key=lambda kv: -len(kv[1])):
        eventos.sort(key=lambda e: int(e["frame_video"]))
        primero, ultimo = eventos[0], eventos[-1]
        ancho = int(np.median([int(e["x2"]) - int(e["x1"]) for e in eventos]))
        x = int(np.median([(int(e["x1"]) + int(e["x2"])) // 2 for e in eventos]))
        y = int(np.median([(int(e["y1"]) + int(e["y2"])) // 2 for e in eventos]))
        resumen.append({
            "track_id": t_id,
            "frames": len(eventos),
            "seg_inicio": primero["segundo"],
            "seg_fin": ultimo["segundo"],
            "ancho_mediano": ancho,
            "x_mediana": x,
            "y_mediana": y,
            "frame_representativo": eventos[len(eventos) // 2]["frame_video"],
            "mejor_similitud": max(float(e["similitud"]) for e in eventos),
        })

    with open(DEMO / f"pistas_{nombre}_filtro{filtro}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        writer.writeheader()
        writer.writerows(resumen)

    cap = cv2.VideoCapture(str(VIDEO))
    imagenes = []
    for fila in resumen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fila["frame_representativo"]))
        ok, frame = cap.read()
        if not ok:
            continue
        evento = por_pista[fila["track_id"]][len(por_pista[fila["track_id"]]) // 2]
        x1, y1, x2, y2 = int(evento["x1"]), int(evento["y1"]), int(evento["x2"]), int(evento["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        vista = cv2.resize(frame, (LADO, int(LADO * frame.shape[0] / frame.shape[1])))
        texto = f"pista {fila['track_id']} {fila['frames']}f {fila['seg_inicio']}-{fila['seg_fin']}s {fila['ancho_mediano']}px"
        cv2.putText(vista, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        imagenes.append(vista)
    cap.release()

    alto = imagenes[0].shape[0]
    while len(imagenes) % COLUMNAS:
        imagenes.append(np.full((alto, LADO, 3), 255, np.uint8))
    filas = [np.hstack(imagenes[i:i + COLUMNAS]) for i in range(0, len(imagenes), COLUMNAS)]
    cv2.imwrite(str(DEMO / f"pistas_{nombre}_filtro{filtro}.jpg"), np.vstack(filas))

    print(f"Pistas de '{nombre}': {len(resumen)}")
    for fila in resumen:
        print(f"  pista {fila['track_id']:>4s}: {fila['frames']:4d} frames, {fila['seg_inicio']}-{fila['seg_fin']} s, "
              f"ancho {fila['ancho_mediano']} px, centro ({fila['x_mediana']},{fila['y_mediana']}), sim max {fila['mejor_similitud']:.2f}")


if __name__ == "__main__":
    main()

"""
Fase 7: hoja de revisión de las pistas NO confirmadas con ancho >= 40 px.

Son las candidatas a estudiantes no registrados, pero también pueden ser estudiantes
registrados en pose difícil que no llegaron a los 3 votos. Dos recortes por pista,
extraídos del video, para etiquetarlas a mano.

Salida: revision/pistas_sin_confirmar_NN.jpg y revision/pistas_sin_confirmar.csv
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
DEMO = RAIZ / "experiments/unc/demo"
ANCHO = 150
COLS = 8
PISTAS_POR_HOJA = 24


def ultima_corrida():
    carpetas = sorted(DEMO.glob("prueba_*_video3"))
    return carpetas[-1] if carpetas else None


def main():
    corrida = ultima_corrida()
    for a in sys.argv[1:]:
        if a.startswith("--corrida="):
            corrida = Path(a.split("=", 1)[1])
    if corrida is None:
        print("No encontré una corrida *_video3.")
        return

    info = defaultdict(lambda: {"sims": [], "anchos": [], "nombre": None, "n": 0})
    cajas = defaultdict(list)
    with open(corrida / "decisiones_video3.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            estado = row["estado"]
            clave_estado = "VERIFICANDO" if estado.startswith("Verificando") else estado
            if clave_estado == "CENTRATE":
                continue
            k = (int(row["parte"]), int(row["track_id"]))
            d = info[k]
            d["n"] += 1
            d["anchos"].append(int(row["x2"]) - int(row["x1"]))
            if row["similitud"]:
                d["sims"].append(float(row["similitud"]))
            if clave_estado in ("CONFIRMADO", "ANCLADO"):
                d["nombre"] = row["nombre"]
            if clave_estado in ("DESCONOCIDO", "VERIFICANDO"):
                cajas[k].append((int(row["frame_parte"]),
                                 (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))))

    pendientes = [(k, v) for k, v in info.items()
                  if not v["nombre"] and statistics.median(v["anchos"] or [0]) >= 40]
    pendientes.sort(key=lambda kv: (kv[0][0], kv[0][1]))
    print(f"Pistas sin confirmar con ancho >= 40 px: {len(pendientes)}")

    partes = sorted((RAIZ / "data/unc/salon").glob("lv_0_2026*.mp4"))
    imgs, filas = [], []
    for n, (k, v) in enumerate(pendientes, 1):
        frames = cajas.get(k, [])
        if not frames or k[0] - 1 >= len(partes):
            continue
        mx = max(v["sims"]) if v["sims"] else 0
        filas.append([k[0], k[1], v["n"], round(mx, 4),
                      round(statistics.median(v["anchos"]), 1), ""])
        cap = cv2.VideoCapture(str(partes[k[0] - 1]))
        for idx in (frames[len(frames) // 4], frames[3 * len(frames) // 4]):
            frame_parte, caja = idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_parte - 1)
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = caja
            m = int((x2 - x1) * 0.4)
            recorte = frame[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
            if recorte.size == 0:
                continue
            img = cv2.resize(recorte, (ANCHO, ANCHO))
            cv2.putText(img, f"p{k[0]} t{k[1]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
            imgs.append(img)
        cap.release()

    for hoja_n, inicio in enumerate(range(0, len(imgs), PISTAS_POR_HOJA * 2), 1):
        trozo = imgs[inicio:inicio + PISTAS_POR_HOJA * 2]
        while len(trozo) % COLS:
            trozo.append(np.full((ANCHO, ANCHO, 3), 255, np.uint8))
        filas_img = [np.hstack(trozo[i:i + COLS]) for i in range(0, len(trozo), COLS)]
        cv2.imwrite(str(corrida / f"revision/pistas_sin_confirmar_{hoja_n:02d}.jpg"),
                    np.vstack(filas_img))
        print(f"  hoja {hoja_n:02d}: {len(trozo)} recortes")

    with open(corrida / "revision/pistas_sin_confirmar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parte", "pista", "frames", "sim_max", "ancho_mediano", "etiqueta"])
        w.writerows(filas)
    print(f"Tabla: {corrida / 'revision/pistas_sin_confirmar.csv'}")


if __name__ == "__main__":
    main()

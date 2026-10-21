"""
Fase 7: revisión de pistas débiles y de las pistas de estudiantes desconocidos.

Arma dos hojas:
  pistas_debiles.jpg       las confirmaciones con menor similitud máxima (donde cabría
                           una confusión con alguien no registrado), 3 capturas por pista
  pistas_desconocidas.jpg  las pistas NO confirmadas con más frames y ancho >= 40 px,
                           que son los estudiantes no registrados, 2 capturas por pista
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
ANCHO = 200


def ultima_corrida():
    carpetas = sorted(DEMO.glob("prueba_*_video3"))
    return carpetas[-1] if carpetas else None


def leer_parte_track(nombre):
    parte = track = 0
    for trozo in nombre.replace(".jpg", "").split("_"):
        if trozo.startswith("p") and trozo[1:].isdigit():
            parte = int(trozo[1:])
        if trozo.startswith("t") and trozo[1:].isdigit():
            track = int(trozo[1:])
    return parte, track


def capturas_por_pista(carpeta):
    indice = defaultdict(list)
    for ruta in sorted(carpeta.glob("*/*.jpg")):
        parte, track = leer_parte_track(ruta.name)
        indice[(parte, track)].append(ruta)
    return indice


def hoja(imgs, destino, ancho=ANCHO, cols=6):
    if not imgs:
        return
    while len(imgs) % cols:
        imgs.append(np.full((ancho, ancho, 3), 255, np.uint8))
    filas = [np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
    cv2.imwrite(str(destino), np.vstack(filas))


def main():
    corrida = ultima_corrida()
    for a in sys.argv[1:]:
        if a.startswith("--corrida="):
            corrida = Path(a.split("=", 1)[1])
    if corrida is None:
        print("No encontré una corrida *_video3.")
        return

    ruta_csv = corrida / "decisiones_video3.csv"
    info = defaultdict(lambda: {"sims": [], "anchos": [], "nombre": None, "n": 0})
    with open(ruta_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            estado = row["estado"]
            estado = "VERIFICANDO" if estado.startswith("Verificando") else estado
            if estado == "CENTRATE":
                continue
            k = (int(row["parte"]), int(row["track_id"]))
            d = info[k]
            d["n"] += 1
            d["anchos"].append(int(row["x2"]) - int(row["x1"]))
            if row["similitud"]:
                d["sims"].append(float(row["similitud"]))
            if estado in ("CONFIRMADO", "ANCLADO"):
                d["nombre"] = row["nombre"]

    capturas = capturas_por_pista(corrida / "estudiantes")

    # 1) Confirmaciones más débiles
    confirmadas = [(k, v) for k, v in info.items() if v["nombre"]]
    confirmadas.sort(key=lambda kv: max(kv[1]["sims"]) if kv[1]["sims"] else 0)
    imgs, filas = [], []
    print("Confirmaciones más débiles:")
    for k, v in confirmadas[:18]:
        rutas = capturas.get(k, [])
        if not rutas:
            continue
        elegidas = [rutas[0], rutas[len(rutas) // 2], rutas[-1]]
        mx = max(v["sims"]) if v["sims"] else 0
        print(f"  p{k[0]:02d} t{k[1]:<4} {v['nombre']:<8} n={v['n']:4d} sim_max={mx:.3f} "
              f"ancho={statistics.median(v['anchos']):.0f}px capturas={len(rutas)}")
        filas.append([v["nombre"], k[0], k[1], v["n"], round(mx, 4),
                      round(statistics.median(v["anchos"]), 1)])
        for ruta in elegidas:
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            img = cv2.resize(img, (ANCHO, ANCHO))
            cv2.putText(img, f"{v['nombre']} p{k[0]} t{k[1]} {mx:.2f}", (3, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            imgs.append(img)
    hoja(imgs, corrida / "revision/pistas_debiles.jpg")
    with open(corrida / "revision/pistas_debiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "parte", "pista", "frames", "sim_max", "ancho_mediano"])
        w.writerows(filas)

    # 2) Pistas de estudiantes no registrados (sin confirmar, grandes y largas).
    #    No tienen capturas guardadas, así que se recortan del video.
    partes = sorted((RAIZ / "data/unc/salon").glob("lv_0_2026*.mp4"))
    cajas = defaultdict(list)
    with open(ruta_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            estado = row["estado"]
            if estado not in ("DESCONOCIDO", "VERIFICANDO") and not estado.startswith("Verificando"):
                continue
            k = (int(row["parte"]), int(row["track_id"]))
            cajas[k].append((int(row["frame_parte"]),
                             (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))))

    desconocidas = [(k, v) for k, v in info.items()
                    if not v["nombre"] and statistics.median(v["anchos"] or [0]) >= 40]
    desconocidas.sort(key=lambda kv: -kv[1]["n"])
    imgs, filas = [], []
    print("\nPistas no confirmadas con más frames (candidatas a estudiantes desconocidos):")
    for k, v in desconocidas[:18]:
        frames = cajas.get(k, [])
        if not frames:
            continue
        mx = max(v["sims"]) if v["sims"] else 0
        print(f"  p{k[0]:02d} t{k[1]:<4} n={v['n']:4d} sim_max={mx:.3f} "
              f"ancho={statistics.median(v['anchos']):.0f}px")
        filas.append([k[0], k[1], v["n"], round(mx, 4), round(statistics.median(v["anchos"]), 1)])
        if k[0] - 1 >= len(partes):
            continue
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
            cv2.putText(img, f"p{k[0]} t{k[1]} {mx:.2f}", (3, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            imgs.append(img)
        cap.release()
    hoja(imgs, corrida / "revision/pistas_desconocidas.jpg")
    with open(corrida / "revision/pistas_desconocidas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parte", "pista", "frames", "sim_max", "ancho_mediano"])
        w.writerows(filas)

    print(f"\nHojas y tablas en: {corrida / 'revision'}")


if __name__ == "__main__":
    main()

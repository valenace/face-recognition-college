"""
Fase 7: ¿cuántos estudiantes distintos hay entre las pistas no confirmadas?

Calcula un embedding por pista (promedio de 3 recortes) y agrupa las pistas por
similitud coseno. Devuelve cuántas personas distintas hay, cuántas pistas tiene cada
una y qué tan cerca quedó del estudiante registrado más parecido.

Salida: revision/clusters_sin_confirmar.csv y resumen por pantalla.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from uniface import create_detector, create_recognizer
from core.utils_facial import embedding_facial

DEMO = RAIZ / "experiments/unc/demo"
UMBRAL_CLUSTER = 0.50


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

    tabla = list(csv.DictReader(open(corrida / "revision/pistas_sin_confirmar.csv", encoding="utf-8")))
    print(f"Pistas a agrupar: {len(tabla)}")

    # frames y cajas de cada pista
    cajas = defaultdict(list)
    with open(corrida / "decisiones_video3.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            estado = row["estado"]
            if estado.startswith("Verificando"):
                estado = "VERIFICANDO"
            if estado not in ("DESCONOCIDO", "VERIFICANDO"):
                continue
            k = (int(row["parte"]), int(row["track_id"]))
            cajas[k].append((int(row["frame_parte"]),
                             (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))))

    partes = sorted((RAIZ / "data/unc/salon").glob("lv_0_2026*.mp4"))
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    vectores = []
    for fila in tabla:
        k = (int(fila["parte"]), int(fila["pista"]))
        frames = cajas.get(k, [])
        if not frames or k[0] - 1 >= len(partes):
            continue
        cap = cv2.VideoCapture(str(partes[k[0] - 1]))
        embs = []
        for pos in (len(frames) // 5, len(frames) // 2, 4 * len(frames) // 5):
            frame_parte, caja = frames[pos]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_parte - 1)
            ok, frame = cap.read()
            if not ok:
                continue
            rostros = detector.detect(frame)
            if not rostros:
                continue
            rostro = min(rostros, key=lambda r: sum((a - b) ** 2 for a, b in zip(r.bbox, caja)))
            embs.append(embedding_facial(recognizer, frame, rostro.landmarks).astype("float32"))
        cap.release()
        if not embs:
            continue
        v = np.mean(embs, axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        vectores.append({"parte": k[0], "pista": k[1], "frames": fila["frames"],
                         "sim_max": float(fila["sim_max"]), "ancho": fila["ancho_mediano"],
                         "vec": v})

    print(f"Pistas con embedding válido: {len(vectores)}")

    clusters = []
    for item in vectores:
        asignado = None
        for c in clusters:
            if float(np.dot(c["centroide"], item["vec"])) >= UMBRAL_CLUSTER:
                asignado = c
                break
        if asignado is None:
            clusters.append({"centroide": item["vec"].copy(), "miembros": [item]})
        else:
            asignado["miembros"].append(item)
            m = np.mean([x["vec"] for x in asignado["miembros"]], axis=0)
            asignado["centroide"] = m / (np.linalg.norm(m) + 1e-9)

    clusters.sort(key=lambda c: -len(c["miembros"]))
    print(f"\nPersonas distintas estimadas (cos >= {UMBRAL_CLUSTER}): {len(clusters)}")
    print(f"{'grupo':>5} {'pistas':>7} {'partes':>14} {'sim_max mediana':>16} {'ancho mediano':>14}")
    filas = []
    for n, c in enumerate(clusters, 1):
        miembros = c["miembros"]
        partes_c = sorted({m["parte"] for m in miembros})
        sims = [m["sim_max"] for m in miembros]
        anchos = [float(m["ancho"]) for m in miembros]
        print(f"{n:5d} {len(miembros):7d} {str(partes_c):>14} {statistics.median(sims):16.3f} "
              f"{statistics.median(anchos):14.0f}")
        for m in miembros:
            filas.append([n, m["parte"], m["pista"], m["frames"], m["sim_max"], m["ancho"]])

    with open(corrida / "revision/clusters_sin_confirmar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "parte", "pista", "frames", "sim_max", "ancho_mediano"])
        w.writerows(filas)


if __name__ == "__main__":
    main()

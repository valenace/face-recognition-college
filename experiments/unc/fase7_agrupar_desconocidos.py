"""
Fase 7: ¿a quién pertenecen los desconocidos cercanos al umbral?

Recalcula el embedding de cada recorte único guardado en desconocidos/ y lo compara
con los 9 centroides. Sirve para medir cuánto de la "zona gris" (sim 0.25-0.30) es
gente registrada en pose difícil y cuánto podría ser gente ajena.

Ojo: el embedding se recalcula sobre el recorte guardado (con margen), así que la
similitud no es idéntica a la del motor; el nombre más cercano sí es informativo.

Salida: revision/desconocidos_grupos.csv y un resumen por pantalla.
"""

import csv
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from uniface import create_detector, create_recognizer
from core.utils_facial import embedding_facial

DEMO = RAIZ / "experiments/unc/demo"


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

    with open(RAIZ / "data/database_embeddings.pkl", "rb") as f:
        db = pickle.load(f)
    nombres = sorted(db)
    matriz = np.stack([db[n] for n in nombres]).astype("float32")

    filas_csv = list(csv.DictReader(open(corrida / "revision/desconocidos_dedup.csv",
                                     encoding="utf-8")))
    print(f"Recortes únicos a analizar: {len(filas_csv)}")
    print("Cargando detector y reconocedor...")
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    resultados = []
    sin_rostro = 0
    for i, fila in enumerate(filas_csv, 1):
        ruta = corrida / "desconocidos" / fila["archivo"]
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        rostros = detector.detect(img)
        if not rostros:
            sin_rostro += 1
            continue
        rostro = max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
        vec = embedding_facial(recognizer, img, rostro.landmarks).astype("float32")
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        sims = matriz @ vec
        orden = np.argsort(-sims)
        resultados.append({
            "parte": fila["parte"], "segundo": fila["segundo"], "pista": fila["pista"],
            "sim_motor": fila["sim"], "top1": nombres[orden[0]], "sim_top1": round(float(sims[orden[0]]), 4),
            "top2": nombres[orden[1]], "sim_top2": round(float(sims[orden[1]]), 4),
            "archivo": fila["archivo"],
        })
        if i % 50 == 0:
            print(f"  {i}/{len(filas_csv)}")

    with open(corrida / "revision/desconocidos_grupos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados[0]))
        w.writeheader()
        w.writerows(resultados)

    conteo = Counter(r["top1"] for r in resultados)
    por_persona = defaultdict(list)
    for r in resultados:
        por_persona[r["top1"]].append(r["sim_top1"])

    print(f"\nSin rostro detectable al recalcular: {sin_rostro}")
    print(f"{'más cercano':<10} {'recortes':>9} {'sim mediana':>12} {'sim max':>8} {'>=0.30':>7}")
    for nombre, n in conteo.most_common():
        sims = por_persona[nombre]
        sobre = sum(1 for s in sims if s >= 0.30)
        print(f"{nombre:<10} {n:9d} {np.median(sims):12.3f} {np.max(sims):8.3f} {sobre:7d}")

    print(f"\nTotal clasificados: {len(resultados)}")
    print(f"Tabla completa: {corrida / 'revision/desconocidos_grupos.csv'}")


if __name__ == "__main__":
    main()

"""
Fase 2 — Dispersión interna por persona.

Para cada persona registrada, calcula la similitud coseno de cada una de sus fotos
con su propio centroide. Valores altos y juntos = fotos homogéneas; valores bajos
o muy separados = fotos heterogéneas (por ejemplo, con y sin accesorios).

Salida:
  experiments/unc/fase2_dispersion.csv
  experiments/unc/fase2_dispersion.png
"""

import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pickle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector, create_recognizer
from core.utils_facial import embedding_facial

RAIZ = Path(__file__).resolve().parents[2]
DB = RAIZ / "data/database_embeddings.pkl"
FRAMES = RAIZ / "data/dataset_cv/frames"
SALIDA = RAIZ / "experiments/unc"


def main():
    with open(DB, "rb") as f:
        db = pickle.load(f)

    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    resultados = {}
    for nombre in sorted(db):
        centroide = np.asarray(db[nombre]).reshape(-1)
        sims = []
        for ruta in sorted(glob.glob(str(FRAMES / nombre / "*.jpg"))):
            img = cv2.imread(ruta)
            rostros = detector.detect(img)
            if not rostros:
                continue
            mejor = max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
            vec = embedding_facial(recognizer, img, mejor.landmarks)
            sims.append(float(np.dot(vec, centroide)))
        resultados[nombre] = np.array(sims)

    with open(SALIDA / "fase2_dispersion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "fotos", "media", "mediana", "minimo", "maximo", "desviacion"])
        for n, s in resultados.items():
            writer.writerow([n, len(s), f"{s.mean():.4f}", f"{np.median(s):.4f}",
                             f"{s.min():.4f}", f"{s.max():.4f}", f"{s.std():.4f}"])

    fig, ax = plt.subplots(figsize=(10, 5))
    nombres = list(resultados)
    for i, n in enumerate(nombres):
        s = resultados[n]
        x = np.random.default_rng(0).normal(i, 0.06, size=len(s))
        ax.scatter(x, s, s=18, alpha=0.7)
        ax.hlines(s.mean(), i - 0.25, i + 0.25, colors="red")
    ax.set_xticks(range(len(nombres)))
    ax.set_xticklabels(nombres)
    ax.set_ylabel("Similitud de cada foto con su centroide")
    ax.set_title("Dispersión interna por persona (línea roja = media)")
    ax.axhline(0.35, color="gray", linestyle="--", label="umbral del motor 0.35")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SALIDA / "fase2_dispersion.png", dpi=120)

    for n in nombres:
        s = resultados[n]
        print(f"{n:8s} fotos={len(s):3d} media={s.mean():.3f} min={s.min():.3f} max={s.max():.3f} std={s.std():.3f}")


if __name__ == "__main__":
    main()

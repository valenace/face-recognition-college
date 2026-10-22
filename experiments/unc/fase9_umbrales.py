"""
Fase 9: umbral adaptativo por persona.

Calcula, para cada estudiante, la distribución de similitudes de sus 50 fotos de
registro contra su propio centroide (genuinas) y contra los centroides de los demás
(impostoras). Con eso se propone un umbral por persona que separe ambos mundos.

Uso:
    venv/bin/python experiments/unc/fase9_umbrales.py

No escribe en la base: solo imprime las distribuciones y los umbrales propuestos.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from uniface import create_detector, create_recognizer
from core.utils_facial import embedding_facial

FOTOS = RAIZ / "data/dataset_cv/frames"
DB = RAIZ / "data/database_embeddings.pkl"


def main():
    with open(DB, "rb") as f:
        centroides = pickle.load(f)

    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    embeddings = {}
    for persona in sorted(centroides):
        carpeta = FOTOS / persona
        vectores = []
        for ruta in sorted(carpeta.glob("*.jpg")):
            img = __import__("cv2").imread(str(ruta))
            if img is None:
                continue
            rostros = detector.detect(img)
            if not rostros:
                continue
            rostro = max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
            vec = embedding_facial(recognizer, img, rostro.landmarks)
            vectores.append(vec / (np.linalg.norm(vec) + 1e-9))
        embeddings[persona] = np.stack(vectores)
        print(f"  {persona:<8} fotos usadas: {len(vectores)}")

    print(f"\n{'persona':<8} {'genuina p10':>12} {'p50':>7} {'p90':>7} "
          f"{'impostora p99':>14} {'máx':>7} {'umbral prop.':>13}")
    propuestos = {}
    for persona in sorted(centroides):
        centroide = np.asarray(centroides[persona], dtype="float32")
        genuinas = embeddings[persona] @ centroide
        impostoras = []
        for otro in sorted(centroides):
            if otro == persona:
                continue
            impostoras.extend(embeddings[otro] @ centroide)
        impostoras = np.array(impostoras)

        p10 = np.percentile(genuinas, 10)
        p50 = np.percentile(genuinas, 50)
        p90 = np.percentile(genuinas, 90)
        imp99 = np.percentile(impostoras, 99)
        imp_max = impostoras.max()

        # Punto medio entre la cola genuina y la cola impostora, acotado.
        umbral = (p10 + imp99) / 2
        umbral = float(min(max(umbral, 0.28), 0.70))
        propuestos[persona] = round(umbral, 4)
        print(f"{persona:<8} {p10:12.3f} {p50:7.3f} {p90:7.3f} "
              f"{imp99:14.3f} {imp_max:7.3f} {umbral:13.4f}")

    print("\nUmbrales propuestos:", propuestos)


if __name__ == "__main__":
    main()

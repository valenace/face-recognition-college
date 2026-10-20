"""
Fase 1 — Diagnóstico del dataset UNC.

Para cada video de registro (data/unc/stds/) y el video del salón (102851):
  - muestrea 1 de cada 3 frames
  - cuenta caras por frame con RetinaFace
  - para frames con cara única: alinea, mide nitidez (umbral 60) y brillo
  - resume tasas y anchos de caja

Salida:
  experiments/unc/fase1_resumen_videos.csv
  experiments/unc/fase1_caras_salon.csv
  experiments/unc/fase1_tasa_deteccion.png
  experiments/unc/fase1_bbox_salon.png
"""

import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uniface import create_detector
from core.utils_facial import alinear_rostro, es_imagen_borrosa

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "experiments/unc"
STDS = sorted(glob.glob(str(RAIZ / "data/unc/stds/*/*.mp4")))
SALON = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
PASO = 3
UMBRAL_NITIDEZ = 60


def nitidez(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def brillo(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def analizar_video(detector, ruta, persona, caras_salon=None):
    cap = cv2.VideoCapture(str(ruta))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    muestreados = con_cara = cara_unica = util = 0
    anchos = []
    indice = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if indice % PASO == 0:
            muestreados += 1
            caras = detector.detect(frame)
            if caras:
                con_cara += 1
            if caras_salon is not None:
                caras_salon.append({
                    "frame_idx": indice,
                    "caras": len(caras),
                    "anchos": [int(c.bbox[2] - c.bbox[0]) for c in caras],
                })
            if len(caras) == 1:
                cara_unica += 1
                c = caras[0]
                anchos.append(int(c.bbox[2] - c.bbox[0]))
                alineado = alinear_rostro(frame, c.landmarks)
                if alineado is not None and not es_imagen_borrosa(alineado, umbral=UMBRAL_NITIDEZ):
                    util += 1
        indice += 1
    cap.release()
    return {
        "persona": persona,
        "frames_totales": total,
        "frames_muestreados": muestreados,
        "tasa_deteccion": round(con_cara / muestreados, 4) if muestreados else 0,
        "tasa_cara_unica": round(cara_unica / muestreados, 4) if muestreados else 0,
        "tasa_util": round(util / muestreados, 4) if muestreados else 0,
        "bbox_min": min(anchos) if anchos else "",
        "bbox_max": max(anchos) if anchos else "",
        "bbox_media": round(float(np.mean(anchos)), 1) if anchos else "",
        "frames_utiles_estimados": round(util * PASO),
    }


def main():
    detector = create_detector("retinaface")
    resumen = []
    for ruta in STDS:
        persona = Path(ruta).parent.name
        print(f"Analizando registro: {persona}")
        resumen.append(analizar_video(detector, ruta, persona))

    caras_salon = []
    print("Analizando salón: VID_20260907_102851")
    resumen.append(analizar_video(detector, SALON, "salon_102851", caras_salon))

    with open(SALIDA / "fase1_resumen_videos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        writer.writeheader()
        writer.writerows(resumen)

    with open(SALIDA / "fase1_caras_salon.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "caras", "anchos"])
        for fila in caras_salon:
            writer.writerow([fila["frame_idx"], fila["caras"], fila["anchos"]])

    personas = [r["persona"] for r in resumen if r["persona"] != "salon_102851"]
    det = [r["tasa_deteccion"] for r in resumen if r["persona"] != "salon_102851"]
    cu = [r["tasa_cara_unica"] for r in resumen if r["persona"] != "salon_102851"]
    ut = [r["tasa_util"] for r in resumen if r["persona"] != "salon_102851"]
    x = np.arange(len(personas))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.25, det, 0.25, label="con al menos 1 cara")
    ax.bar(x, cu, 0.25, label="cara única")
    ax.bar(x + 0.25, ut, 0.25, label="útil (única y nítida)")
    ax.set_xticks(x)
    ax.set_xticklabels(personas)
    ax.set_ylabel("Proporción de frames muestreados")
    ax.set_title("Fase 1: tasas por video de registro")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fase1_tasa_deteccion.png", dpi=120)

    anchos_salon = [a for f_ in caras_salon for a in f_["anchos"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(anchos_salon, bins=range(0, max(anchos_salon) + 10, 5))
    ax.axvline(40, color="red", linestyle="--", label="40 px")
    ax.set_xlabel("Ancho de caja (px)")
    ax.set_ylabel("Número de caras")
    ax.set_title("Fase 1: anchos de cara en el salón (102851)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fase1_bbox_salon.png", dpi=120)

    print("\nResumen registro:")
    for r in resumen:
        print(f"  {r['persona']:14s} det={r['tasa_deteccion']:.2f} unica={r['tasa_cara_unica']:.2f} util={r['tasa_util']:.2f} "
              f"bbox={r['bbox_min']}-{r['bbox_max']} media={r['bbox_media']} utiles_est={r['frames_utiles_estimados']}")
    print(f"\nSalón: frames muestreados={len(caras_salon)}, caras totales={sum(f_['caras'] for f_ in caras_salon)}, "
          f"mediana caras/frame={np.median([f_['caras'] for f_ in caras_salon]):.0f}, "
          f"ancho mediano={np.median(anchos_salon):.0f}, <40px={sum(a < 40 for a in anchos_salon)}")


if __name__ == "__main__":
    main()

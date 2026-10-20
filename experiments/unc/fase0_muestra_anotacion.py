"""
Fase 0 — Muestra estratificada para anotación y cuadrículas de revisión.

Selecciona una muestra aleatoria reproducible (semilla fija) de los recortes de
fase0_muestreo_salon.py y genera cuadrículas numeradas para revisión visual.
"""

import csv
import random
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "experiments/unc/salon_muestra"
SEMILLA = 42
TAMANO_MUESTRA = 300
COLUMNAS = 6
FILAS = 5
LADO = 160


def cargar_recortes():
    with open(SALIDA / "recortes.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generar_muestra(filas):
    rng = random.Random(SEMILLA)
    por_video = {}
    for fila in filas:
        por_video.setdefault(fila["video"], []).append(fila)
    muestra = []
    for video in sorted(por_video):
        cuota = TAMANO_MUESTRA // len(por_video)
        muestra.extend(rng.sample(por_video[video], min(cuota, len(por_video[video]))))
    rng.shuffle(muestra)
    return muestra


def cuadricula(muestra, indice_inicio, carpeta_salida):
    por_pagina = COLUMNAS * FILAS
    lote = muestra[indice_inicio:indice_inicio + por_pagina]
    lienzo = np.full((FILAS * (LADO + 30), COLUMNAS * LADO, 3), 255, dtype=np.uint8)
    for k, fila in enumerate(lote):
        r, c = divmod(k, COLUMNAS)
        img = cv2.imread(str(SALIDA / "crops" / f"{fila['recorte_id']}.jpg"))
        img = cv2.resize(img, (LADO, LADO))
        y, x = r * (LADO + 30), c * LADO
        lienzo[y + 30:y + 30 + LADO, x:x + LADO] = img
        cv2.putText(lienzo, str(k + 1), (x + 5, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ruta = carpeta_salida / f"cuadricula_{indice_inicio // por_pagina + 1:02d}.jpg"
    cv2.imwrite(str(ruta), lienzo)
    return ruta


def main():
    filas = cargar_recortes()
    muestra = generar_muestra(filas)

    carpeta = SALIDA / "cuadriculas"
    carpeta.mkdir(parents=True, exist_ok=True)

    with open(SALIDA / "muestra_anotacion.csv", "w", newline="", encoding="utf-8") as f:
        campos = ["orden", "recorte_id", "video", "frame_idx", "segundo", "ancho", "etiqueta", "notas"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for i, fila in enumerate(muestra, start=1):
            writer.writerow({
                "orden": i,
                "recorte_id": fila["recorte_id"],
                "video": fila["video"],
                "frame_idx": fila["frame_idx"],
                "segundo": fila["segundo"],
                "ancho": fila["ancho"],
                "etiqueta": "",
                "notas": "",
            })

    por_pagina = COLUMNAS * FILAS
    for inicio in range(0, len(muestra), por_pagina):
        cuadricula(muestra, inicio, carpeta)

    print(f"Muestra: {len(muestra)} recortes, {-(-len(muestra) // por_pagina)} cuadrículas en {carpeta}")


if __name__ == "__main__":
    main()

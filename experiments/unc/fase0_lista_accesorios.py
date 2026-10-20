"""
Fase 0 — Lista de trabajo para anotar accesorios (lentes y gorra).

Toma los recortes de personas registradas de reconocimiento_salon.csv y genera:
  - accesorios_muestra.csv: una fila por recorte, columnas lentes y gorra vacías
  - cuadriculas_accesorios/: tablas de 30 caras numeradas para revisar a simple vista
"""

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DIR = RAIZ / "experiments/unc/salon_muestra"
RECONOCIMIENTO = SALIDA_DIR / "reconocimiento_salon.csv"
CROPS = SALIDA_DIR / "crops"
CUADRICULAS = SALIDA_DIR / "cuadriculas_accesorios"
PERSONAS = {"alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"}
COLUMNAS = 6
FILAS = 5
LADO = 160


def main():
    with open(RECONOCIMIENTO, encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["etiqueta"] in PERSONAS]
    filas.sort(key=lambda r: r["recorte_id"])

    with open(SALIDA_DIR / "accesorios_muestra.csv", "w", newline="", encoding="utf-8") as f:
        campos = ["orden", "recorte_id", "etiqueta", "resultado", "lentes", "gorra", "notas"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for i, r in enumerate(filas, start=1):
            writer.writerow({
                "orden": i,
                "recorte_id": r["recorte_id"],
                "etiqueta": r["etiqueta"],
                "resultado": r["resultado"],
                "lentes": "",
                "gorra": "",
                "notas": "",
            })

    CUADRICULAS.mkdir(parents=True, exist_ok=True)
    por_pagina = COLUMNAS * FILAS
    for pagina, inicio in enumerate(range(0, len(filas), por_pagina), start=1):
        lote = filas[inicio:inicio + por_pagina]
        lienzo = np.full((FILAS * (LADO + 30), COLUMNAS * LADO, 3), 255, dtype=np.uint8)
        for k, r in enumerate(lote):
            fila_pos, col = divmod(k, COLUMNAS)
            img = cv2.resize(cv2.imread(str(CROPS / f"{r['recorte_id']}.jpg")), (LADO, LADO))
            y, x = fila_pos * (LADO + 30), col * LADO
            lienzo[y + 30:y + 30 + LADO, x:x + LADO] = img
            cv2.putText(lienzo, f"{inicio + k + 1}", (x + 5, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(CUADRICULAS / f"accesorios_{pagina:02d}.jpg"), lienzo)

    print(f"Recortes con nombre: {len(filas)}")
    print(f"Cuadrículas: {-(-len(filas) // por_pagina)} en {CUADRICULAS}")


if __name__ == "__main__":
    main()

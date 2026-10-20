"""
Fase 0 — Aplica las etiquetas visuales de primera pasada a muestra_anotacion.csv.

Códigos por cuadrícula (30 recortes cada una, en orden de la cuadrícula):
  D = desconocido  (cara clara que no coincide con ninguno de los 9 registrados)
  U = dudoso       (podría ser alguno de los 9, o la cara no se distingue)
  N = no_cara      (sin rostro visible: espalda, mano, cuerpo)

Ninguna etiqueta asigna nombre de los 9 registrados: con esta resolución no hay
identificación visual confiable. Las etiquetas son de BAJA confianza y deben
revisarse por una segunda persona antes de usarse como ground truth.
"""

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ARCHIVO = RAIZ / "experiments/unc/salon_muestra/muestra_anotacion.csv"

CODIGOS = {"D": "desconocido", "U": "dudoso", "N": "no_cara"}

CUADRICULAS = [
    "DDUDDD" "UDUUUU" "UDUDDN" "UUUUUU" "UDUNDU",
    "UUUDDN" "UUUDUU" "UDNUDU" "UUUUUD" "NDDUDN",
    "NDUDDU" "DDDDUU" "UUDUDU" "UUUUUD" "NDUUUN",
    "UUDDDU" "UUDUUU" "UUDUDU" "UDUUUU" "UDNDUU",
    "UDNDUU" "DUDDDD" "UUUUUU" "UUDUUU" "UDUUNU",
    "UDDDDU" "UUDDUU" "UUDUUD" "UDDUDU" "DDDUUD",
    "DUDDDU" "DUDDUU" "UUDUUN" "DDUUDU" "UDDDDU",
    "UUDDDU" "UDDUUU" "DUDDDU" "UUDUUU" "UDDUUD",
    "DUDUDU" "DDNUDU" "UUDUUU" "UUDUUU" "UUUDUD",
    "UDNDUD" "DUUDDD" "UDUUUN" "UUDUUU" "NUUUUU",
]


def main():
    with open(ARCHIVO, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    codigos = "".join(CUADRICULAS)
    if len(codigos) != len(filas):
        raise ValueError(f"Hay {len(codigos)} códigos para {len(filas)} recortes")

    for fila in filas:
        codigo = codigos[int(fila["orden"]) - 1]
        fila["etiqueta"] = CODIGOS[codigo]
        fila["notas"] = "primera pasada visual, baja confianza"

    with open(ARCHIVO, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    conteo = {}
    for fila in filas:
        conteo[fila["etiqueta"]] = conteo.get(fila["etiqueta"], 0) + 1
    print(f"Etiquetas aplicadas a {len(filas)} recortes: {conteo}")


if __name__ == "__main__":
    main()

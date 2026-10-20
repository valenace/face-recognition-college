"""
Fase 0 — Efecto del filtro de tamaño mínimo del motor sobre el reconocimiento.

Usa las decisiones ya calculadas (reconocimiento_salon.csv) y simula tres filtros:
  - 40 px (valor actual del motor)
  - 20 px
  - sin filtro (0 px)

Una cara solo cuenta como reconocida si su ancho es mayor que el filtro.
Si no pasa el filtro, el motor no la procesa y cuenta como no reconocida.

Salida:
  experiments/unc/salon_muestra/efecto_filtro_tamano.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DIR = RAIZ / "experiments/unc/salon_muestra"
RECONOCIMIENTO = SALIDA_DIR / "reconocimiento_salon.csv"
RECORTES = SALIDA_DIR / "recortes.csv"
FILTROS = [40, 20, 0]
PERSONAS = {"alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"}


def main():
    with open(RECORTES, encoding="utf-8") as f:
        ancho = {r["recorte_id"]: int(r["ancho"]) for r in csv.DictReader(f)}
    with open(RECONOCIMIENTO, encoding="utf-8") as f:
        rec = list(csv.DictReader(f))

    filas_salida = []
    for filtro in FILTROS:
        por_persona = defaultdict(lambda: {"total": 0, "reconocidos": 0})
        desconocidos = {"total": 0, "falsos_positivos": 0}
        for r in rec:
            procesado = ancho[r["recorte_id"]] > filtro
            if r["etiqueta"] in PERSONAS:
                por_persona[r["etiqueta"]]["total"] += 1
                if procesado and r["decision"] == r["etiqueta"]:
                    por_persona[r["etiqueta"]]["reconocidos"] += 1
            elif r["etiqueta"] == "desconocido":
                desconocidos["total"] += 1
                if procesado and r["decision"] != "Desconocido":
                    desconocidos["falsos_positivos"] += 1

        for persona in sorted(por_persona):
            d = por_persona[persona]
            filas_salida.append({
                "filtro_px": filtro,
                "persona": persona,
                "total": d["total"],
                "reconocidos": d["reconocidos"],
                "tasa": round(d["reconocidos"] / d["total"], 3),
            })
        filas_salida.append({
            "filtro_px": filtro,
            "persona": "desconocidos (falsos positivos)",
            "total": desconocidos["total"],
            "reconocidos": desconocidos["falsos_positivos"],
            "tasa": round(desconocidos["falsos_positivos"] / desconocidos["total"], 3),
        })

    with open(SALIDA_DIR / "efecto_filtro_tamano.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas_salida[0].keys()))
        writer.writeheader()
        writer.writerows(filas_salida)

    for filtro in FILTROS:
        print(f"Filtro {filtro} px:")
        for fila in filas_salida:
            if fila["filtro_px"] == filtro:
                print(f"  {fila['persona']:32s} reconocidos {fila['reconocidos']:3d}/{fila['total']:3d}  ({fila['tasa']:.1%})")


if __name__ == "__main__":
    main()

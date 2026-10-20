"""
Fase 6 — Evaluación final del sistema sobre el video 1 del salón.

Usa la corrida del motor real (carpeta de prueba) y calcula:
  1. Reconocimiento a nivel de recorte etiquetado (join por IoU en el mismo frame).
  2. Tiempo hasta la confirmación de cada persona (desde que aparece su pista
     hasta el evento de confirmación).
  3. Cobertura: cuántos de los 9 se confirman.

Salida (en la carpeta de la corrida):
  fase6_por_persona.csv
  fase6_tiempo_confirmacion.png
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RAIZ = Path(__file__).resolve().parents[2]
CORRIDA = RAIZ / "experiments/unc/demo/prueba_2010_0935_final"
MUESTRA = RAIZ / "experiments/unc/salon_muestra/muestra_anotacion.csv"
RECORTES = RAIZ / "experiments/unc/salon_muestra/recortes.csv"
PERSONAS = ["alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"]
UMBRAL_IOU = 0.5


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    with open(CORRIDA / "decisiones.csv", encoding="utf-8") as f:
        decisiones = list(csv.DictReader(f))
    with open(RECORTES, encoding="utf-8") as f:
        recortes = {r["recorte_id"]: r for r in csv.DictReader(f)}
    with open(MUESTRA, encoding="utf-8") as f:
        muestra = list(csv.DictReader(f))

    # --- 1. Tiempo hasta la confirmación
    pistas = defaultdict(list)
    for d in decisiones:
        pistas[d["track_id"]].append(d)
    tiempo_conf = {}
    for track_id, filas in pistas.items():
        filas.sort(key=lambda x: float(x["segundo"]))
        confirmadas = [x for x in filas if x["nombre"] in PERSONAS and x["estado"] == "CONFIRMADO"]
        if not confirmadas:
            continue
        persona = confirmadas[0]["nombre"]
        t_inicio = float(filas[0]["segundo"])
        t_conf = float(confirmadas[0]["segundo"])
        tiempo_conf.setdefault(persona, []).append((round(t_conf - t_inicio, 2), track_id, t_inicio, t_conf))

    # --- 2. Evaluación por recorte etiquetado
    por_frame = defaultdict(list)
    for d in decisiones:
        por_frame[int(d["frame_video"])].append(d)
    cuenta = defaultdict(lambda: {"total": 0, "confirmado": 0, "no_reconocido": 0, "sin_decision": 0, "error_id": 0})
    desconocidos = {"total": 0, "falso_confirmado": 0, "ok": 0, "sin_decision": 0}
    for fila in muestra:
        r = recortes[fila["recorte_id"]]
        etiqueta = fila["etiqueta"]
        frame = int(r["frame_idx"])
        if etiqueta not in PERSONAS and etiqueta != "desconocido":
            continue
        caja = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
        mejor, mejor_iou = None, 0.0
        for d in por_frame.get(frame, []):
            valor = iou(caja, (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])))
            if valor > mejor_iou:
                mejor, mejor_iou = d, valor
        if etiqueta in PERSONAS:
            c = cuenta[etiqueta]
            c["total"] += 1
            if mejor is None or mejor_iou < UMBRAL_IOU:
                c["sin_decision"] += 1
            elif mejor["nombre"] == etiqueta and mejor["estado"] in ("CONFIRMADO", "ANCLADO"):
                c["confirmado"] += 1
            elif mejor["nombre"] == etiqueta:
                c["no_reconocido"] += 1
            elif mejor["nombre"] in PERSONAS:
                c["error_id"] += 1
            else:
                c["no_reconocido"] += 1
        else:
            desconocidos["total"] += 1
            if mejor is None or mejor_iou < UMBRAL_IOU:
                desconocidos["sin_decision"] += 1
            elif mejor["nombre"] in PERSONAS and mejor["estado"] in ("CONFIRMADO", "ANCLADO"):
                desconocidos["falso_confirmado"] += 1
            else:
                desconocidos["ok"] += 1

    # --- 3. Salida por persona
    filas = []
    for persona in PERSONAS:
        c = cuenta[persona]
        confirmaciones = sorted(tiempo_conf.get(persona, []))
        filas.append({
            "persona": persona,
            "recortes_etiquetados": c["total"],
            "reconocidos_confirmados": c["confirmado"],
            "procesados_no_reconocidos": c["no_reconocido"],
            "sin_decision": c["sin_decision"],
            "error_identidad": c["error_id"],
            "recall": round(c["confirmado"] / c["total"], 3) if c["total"] else "",
            "pistas_confirmadas": len(confirmaciones),
            "primer_tiempo_confirmacion_s": confirmaciones[0][0] if confirmaciones else "",
        })

    with open(CORRIDA / "fase6_por_persona.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    print(f"{'persona':8s} {'recortes':>8s} {'confirm':>7s} {'no_rec':>6s} {'sin_dec':>7s} {'err_id':>6s} {'recall':>6s} {'pistas':>6s} {'t_confirm':>9s}")
    for f in filas:
        print(f"{f['persona']:8s} {f['recortes_etiquetados']:8d} {f['reconocidos_confirmados']:7d} "
              f"{f['procesados_no_reconocidos']:6d} {f['sin_decision']:7d} {f['error_identidad']:6d} "
              f"{str(f['recall']):>6s} {f['pistas_confirmadas']:6d} {str(f['primer_tiempo_confirmacion_s']):>9s}")

    total_nom = sum(c["total"] for c in cuenta.values())
    total_conf = sum(c["confirmado"] for c in cuenta.values())
    print(f"\nPersonas registradas en la muestra: {total_conf}/{total_nom} confirmadas ({100*total_conf/total_nom:.1f} %)")
    print(f"Desconocidos: {desconocidos['ok']}/{desconocidos['total']} correctos, "
          f"{desconocidos['falso_confirmado']} falsos confirmados, {desconocidos['sin_decision']} sin decisión")
    print(f"Cobertura: {sum(1 for p in PERSONAS if p in tiempo_conf)} de 9 personas confirmadas")
    for p in PERSONAS:
        if p in tiempo_conf:
            print(f"  {p}: primera confirmación {min(t for t,_,_,_ in tiempo_conf[p]):.2f} s tras aparecer su pista")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiempos = [min(t for t, _, _, _ in tiempo_conf[p]) for p in PERSONAS if p in tiempo_conf]
    nombres = [p for p in PERSONAS if p in tiempo_conf]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(nombres, tiempos)
    ax.set_ylabel("Segundos hasta la confirmación")
    ax.set_title("Fase 6: tiempo hasta confirmar a cada persona")
    fig.tight_layout()
    fig.savefig(CORRIDA / "fase6_tiempo_confirmacion.png", dpi=120)


if __name__ == "__main__":
    main()

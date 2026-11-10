"""Simulador offline de la regla de intrusos.

Reproduce `core/intrusos.py::SeguimientoIntrusos` con umbrales configurables a
partir de las pistas capturadas por `capturar_pistas.py`, para elegir el corte
con datos. No reprocesa videos ni carga modelos.

Uso:
    venv/bin/python experiments/unc/simular_intrusos.py
"""

import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ENTRADA = BASE / "experiments/unc/pistas_intrusos"


def cargar(nombre):
    ruta = ENTRADA / f"intrusos_{nombre}.json"
    if not ruta.exists():
        raise SystemExit(
            f"No existe {ruta}. Ejecuta primero `experiments/unc/capturar_pistas.py`."
        )
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def eventos(datos):
    """(frame, tipo, track_id, muestra) con tipo 0=confirmado, 1=desconocido."""
    ev = []
    for tid, info in datos["pistas"].items():
        tid = int(tid)
        for m in info["muestras"]:
            ev.append((m[0], 1, tid, m))
        for f in info.get("confirmaciones", []):
            ev.append((f, 0, tid, None))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def simular(datos, best_cut=None, avg_cut=0.24, frames_req=15, min_ancho=26,
            factor=2.5, olvido_esp=240):
    pistas = {}
    asientos = {}
    filas = {}
    siguiente = 1
    creados = reutilizados = descartados = 0

    ev = eventos(datos)
    frame_actual = -1
    vistos = set()

    for frame, tipo, tid, m in ev:
        if frame != frame_actual:
            frame_actual = frame
            vistos = set()
        vistos.add(tid)
        info = pistas.get(tid)
        if info is None:
            info = {"best": 0.0, "suma": 0.0, "n": 0, "rachas": 0, "descartada": False,
                    "registro_id": None, "creador": True, "bbox": None}
            pistas[tid] = info

        if tipo == 0:  # confirmada
            if info["registro_id"] and info["creador"]:
                descartados += 1
                filas.pop(info["registro_id"], None)
                asientos.pop(info["registro_id"], None)
            info["descartada"] = True
            info["registro_id"] = None
            continue

        _, sim, efectiva, x1, y1, x2, y2 = m
        info["best"] = max(info["best"], sim)
        info["rachas"] += 1
        info["suma"] += efectiva
        info["n"] += 1
        info["bbox"] = (x1, y1, x2, y2)

        promedio = info["suma"] / info["n"] if info["n"] else 0.0
        if (best_cut is not None and info["best"] >= best_cut) or promedio >= avg_cut:
            info["descartada"] = True
            if info["registro_id"] and info["creador"]:
                descartados += 1
                filas.pop(info["registro_id"], None)
                asientos.pop(info["registro_id"], None)
            info["registro_id"] = None
            continue

        if info["registro_id"] is not None:
            asiento = asientos.get(info["registro_id"])
            if asiento:
                asiento["ultimo"] = frame
                asiento["track_id"] = tid
                asiento["bbox"] = info["bbox"]
                ancho = x2 - x1
                if ancho > asiento.get("max_ancho", 0):
                    asiento["max_ancho"] = ancho

        if info["descartada"] or info["rachas"] < frames_req or info["registro_id"] is not None:
            continue

        # buscar asiento cercano reutilizable
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ancho = max(1.0, float(x2 - x1))
        reutilizar = None
        mejor_dist = float("inf")
        for rid, asiento in asientos.items():
            if asiento.get("track_id") in vistos:
                continue
            if frame - asiento.get("ultimo", -10 ** 9) > olvido_esp:
                continue
            sb = asiento.get("bbox")
            if not sb:
                continue
            scx, scy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
            sancho = max(1.0, float(sb[2] - sb[0]))
            dist = math.hypot(cx - scx, cy - scy)
            if dist <= max(ancho, sancho) * factor and dist < mejor_dist:
                mejor_dist = dist
                reutilizar = rid

        if reutilizar is not None:
            rid = reutilizar
            reutilizados += 1
        else:
            if ancho < min_ancho:
                continue  # demasiado pequeña: no se escribe fila
            rid = siguiente
            siguiente += 1
            creados += 1
            filas[rid] = {"best": info["best"], "promedio": promedio,
                          "ancho": ancho, "frames": info["rachas"]}
        info["registro_id"] = rid
        info["creador"] = reutilizar is None
        asiento = asientos.get(rid)
        if asiento is None:
            asientos[rid] = {"bbox": info["bbox"], "ultimo": frame, "track_id": tid,
                             "max_ancho": ancho}
        else:
            asiento["ultimo"] = frame
            asiento["track_id"] = tid
            asiento["bbox"] = info["bbox"]
            if ancho > asiento.get("max_ancho", 0):
                asiento["max_ancho"] = ancho

    return {"filas": len(filas), "creados": creados, "reutilizados": reutilizados,
            "descartados": descartados, "detalle": list(filas.values())}


def main():
    v1 = cargar("v1")
    v3 = cargar("v3")
    print(f"v1: frames={v1['frames']} pistas={len(v1['pistas'])}")
    print(f"v3: frames={v3['frames']} pistas={len(v3['pistas'])}")
    print()
    print(f"{'best_cut':>9} {'avg_cut':>8} {'ancho':>6} {'factor':>7} | "
          f"{'v1 filas':>9} {'v3 filas':>9} {'v3 creada':>10} {'v3 reus':>8}")
    print("-" * 78)
    for best_cut in (None, 0.22, 0.24, 0.26, 0.28, 0.30):
        for avg_cut in (0.16, 0.18, 0.20, 0.24):
            for min_ancho in (26, 32, 40):
                r1 = simular(v1, best_cut=best_cut, avg_cut=avg_cut, min_ancho=min_ancho)
                r3 = simular(v3, best_cut=best_cut, avg_cut=avg_cut, min_ancho=min_ancho)
                print(f"{str(best_cut):>9} {avg_cut:>8} {min_ancho:>6} {2.5:>7} | "
                      f"{r1['filas']:>9} {r3['filas']:>9} {r3['creados']:>10} {r3['reutilizados']:>8}")


if __name__ == "__main__":
    main()

"""Captura instrumentada de pistas desconocidas para calibrar la regla de intrusos.

Un solo paso por cada video; guarda por pista y por frame la similitud, la
similitud efectiva (con el candidato ya confirmado en otra pista) y la caja.
Después se simulan en offline varias reglas con `simular_intrusos.py`.

Salida: experiments/unc/pistas_intrusos/intrusos_v1.json y intrusos_v3.json.

Uso:
    venv/bin/python experiments/unc/capturar_pistas.py
    venv/bin/python experiments/unc/capturar_pistas.py --solo v1
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

import cv2

from scripts.asistencia_unificada_optimizada import MotorAsistencia

SALIDA = BASE / "experiments/unc/pistas_intrusos"
VIDEOS = {
    "v1": (BASE / "data/unc/salon/VID_20260907_102851.mp4", 36.0, 80.0),
    "v3": (BASE / "data/unc/salon/video5096333890499381162.mp4", 0.0, None),
}


def capturar(nombre, video, desde=0.0, hasta=None):
    motor = MotorAsistencia(registrar_csv=False)
    motor.tamano_minimo_rostro = 20

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if desde:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(desde * fps))
    limite = int((hasta - desde) * fps) if hasta else None

    pistas = {}
    frames = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if limite and frames >= limite:
            break
        frames += 1
        decisiones = motor.procesar_frame(frame)
        confirmados = {
            d["nombre"]
            for d in decisiones
            if d["estado"] in ("CONFIRMADO", "ANCLADO") and d.get("nombre")
        }
        for d in decisiones:
            track_id = d.get("track_id")
            if track_id is None:
                continue
            if d["estado"] not in ("DESCONOCIDO", "CONFIRMADO", "ANCLADO"):
                continue
            info = pistas.setdefault(
                track_id,
                {"best": 0.0, "rachas": 0, "confirmado": 0, "muestras": [],
                 "confirmaciones": []},
            )
            if d["estado"] in ("CONFIRMADO", "ANCLADO"):
                info["confirmado"] += 1
                info["confirmaciones"].append(frames)
                continue
            motivo = d.get("motivo") or ""
            sim = d.get("similitud")
            if motivo.startswith("conflicto") or sim is None:
                continue
            candidato = d.get("candidato")
            efectiva = 0.0 if (candidato and candidato in confirmados) else sim
            info["best"] = max(info["best"], sim)
            info["rachas"] += 1
            x1, y1, x2, y2 = d["bbox"]
            info["muestras"].append(
                [frames, round(sim, 4), round(efectiva, 4),
                 int(x1), int(y1), int(x2), int(y2)]
            )
        if frames % 300 == 0:
            print(f"  {nombre} frame {frames} pistas {len(pistas)}", flush=True)

    cap.release()
    salida = {
        "nombre": nombre,
        "frames": frames,
        "segundos": round(time.time() - t0, 1),
        "pistas": {str(k): v for k, v in pistas.items()},
    }
    destino = SALIDA / f"intrusos_{nombre}.json"
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(salida, f)
    print(
        f"GUARDADO {nombre}: frames={frames} pistas={len(pistas)} "
        f"tiempo={time.time()-t0:.0f}s -> {destino}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Captura pistas para la regla de intrusos")
    parser.add_argument("--solo", choices=sorted(VIDEOS), nargs="*", default=None,
                        help="Capturar solo estos videos (por defecto todos)")
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    nombres = args.solo or sorted(VIDEOS)
    for nombre in nombres:
        video, desde, hasta = VIDEOS[nombre]
        if not video.exists():
            print(f"[AVISO] No existe {video}; se omite {nombre}")
            continue
        capturar(nombre, video, desde, hasta)


if __name__ == "__main__":
    main()

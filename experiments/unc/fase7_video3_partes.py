"""
Fase 7: evaluación del motor sobre el video 3 dividido en 11 partes.

Escenario evaluado: cada parte se procesa como un archivo independiente, es decir,
el tracking y los votos arrancan de cero en cada parte, pero los modelos ONNX se
cargan una sola vez para no repetir el arranque 11 veces.

Salidas en experiments/unc/demo/prueba_<fecha>_video3/:
  decisiones_video3.csv    una fila por decisión, con columna `parte` y frame global
  resumen_por_parte.csv    frames, caras y confirmados de cada parte
  resumen_personas.csv     por persona: confirmaciones, similitud y primera aparición
  asistencia_video3.csv    asistencia de la sesión completa (la escribe el motor)
  estudiantes/<nombre>/    capturas de cada reconocimiento
  contacto_<nombre>.jpg    cuadrícula con las capturas de la persona
  desconocidos/            recortes de desconocidos con similitud >= 0.25 (casi pasan)
  desconocidos_resumen.jpg cuadrícula de esos recortes, ordenados por similitud

No guarda video anotado (ocupa ~480 MB y no aporta al análisis).

Uso:
  venv/bin/python experiments/unc/fase7_video3_partes.py [--filtro=20] [--partes=1,2,3]
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.asistencia_unificada_optimizada import MotorAsistencia

RAIZ = Path(__file__).resolve().parents[2]
PARTES_DIR = RAIZ / "data/unc/salon"
DEMO = RAIZ / "experiments/unc/demo"
PERSONAS = {"alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"}
BRILLO_MINIMO = 8      # por debajo de esto el frame es negro (relleno del corte)
SIMILITUD_DUDA = 0.25  # desconocidos por encima de esto se guardan para revisión
ANCHO_CAPTURA = 180
COLS_CONTACTO = 10
MAX_CONTACTO = 120
MAX_DUDAS = 1200


def reiniciar_pistas(motor):
    """Deja el motor listo para una parte nueva sin recargar los modelos."""
    motor.frame_idx = 0
    motor.identidades_ancladas = {}
    motor.votos_identidad = {}
    motor.vetos_identidad = {}
    motor.tracker = sv.ByteTrack()


def recorte(frame, caja, margen=0.3):
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = caja
    mx, my = int((x2 - x1) * margen), int((y2 - y1) * margen)
    return frame[max(0, y1 - my):min(alto, y2 + my), max(0, x1 - mx):min(ancho, x2 + mx)]


def guardar_contacto(carpeta, nombre, destino):
    capturas = sorted(carpeta.glob("*.jpg"))[:MAX_CONTACTO]
    if not capturas:
        return
    imgs = []
    for ruta in capturas:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        img = cv2.resize(img, (ANCHO_CAPTURA, ANCHO_CAPTURA))
        cv2.putText(img, ruta.stem, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
        imgs.append(img)
    while len(imgs) % COLS_CONTACTO:
        imgs.append(np.full((ANCHO_CAPTURA, ANCHO_CAPTURA, 3), 255, np.uint8))
    filas = [np.hstack(imgs[i:i + COLS_CONTACTO]) for i in range(0, len(imgs), COLS_CONTACTO)]
    cv2.imwrite(str(destino / f"contacto_{nombre}.jpg"), np.vstack(filas))


def guardar_resumen_dudas(dudas, destino, ancho=170, cols=10, limite=60):
    if not dudas:
        return
    imgs = []
    for sim, ruta in sorted(dudas, key=lambda x: -x[0])[:limite]:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        img = cv2.resize(img, (ancho, ancho))
        cv2.putText(img, f"sim={sim:.2f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 255), 1)
        imgs.append(img)
    while len(imgs) % cols:
        imgs.append(np.full((ancho, ancho, 3), 255, np.uint8))
    filas = [np.hstack(imgs[i:i + cols]) for i in range(0, len(imgs), cols)]
    cv2.imwrite(str(destino / "desconocidos_resumen.jpg"), np.vstack(filas))


def main():
    filtro = 20
    elegidas = None
    for a in sys.argv[1:]:
        if a.startswith("--filtro="):
            filtro = int(a.split("=", 1)[1])
        if a.startswith("--partes="):
            elegidas = {int(x) for x in a.split("=", 1)[1].split(",")}

    partes = sorted(PARTES_DIR.glob("lv_0_2026*.mp4"))
    if not partes:
        print(f"No encontré partes en {PARTES_DIR}")
        return
    if elegidas:
        partes = [p for i, p in enumerate(partes, 1) if i in elegidas]

    corrida = DEMO / f"prueba_{datetime.now():%d%m_%H%M}_video3"
    corrida.mkdir(parents=True, exist_ok=True)
    carpeta_est = corrida / "estudiantes"
    carpeta_dudas = corrida / "desconocidos"

    print(f"Corrida: {corrida}")
    print(f"Partes: {len(partes)} | filtro: {filtro} px")
    print("Cargando motor (una sola vez)...")
    motor = MotorAsistencia(
        db_path=str(RAIZ / "data/database_embeddings.pkl"),
        csv_path=str(corrida / "asistencia_video3.csv"),
    )
    motor.tamano_minimo_rostro = filtro

    columnas = ["parte", "frame_parte", "frame_global", "segundo_parte", "track_id",
                "nombre", "estado", "similitud", "sim_top2", "nombre2", "margen",
                "motivo", "revision", "conf_det", "x1", "y1", "x2", "y2"]
    archivo = open(corrida / "decisiones_video3.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(archivo)
    writer.writerow(columnas)

    resumen_partes = []
    capturas = set()
    dudas = []
    frame_global = 0
    global_personas = defaultdict(lambda: {"n": 0, "sims": [], "partes": set(), "primera_parte": None,
                                           "primer_frame": None})

    try:
        for i, parte in enumerate(partes, 1):
            reiniciar_pistas(motor)
            cap = cv2.VideoCapture(str(parte))
            if not cap.isOpened():
                print(f"  ✗ no se pudo abrir {parte.name}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_parte = 0
            negros = 0
            caras = 0
            estados = Counter()
            primera_confirmacion = {}

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_parte += 1
                frame_global += 1
                segundo = frame_parte / fps

                if frame.mean() < BRILLO_MINIMO:
                    negros += 1
                    continue

                decisiones = motor.procesar_frame(frame)
                caras += len(decisiones)

                for d in decisiones:
                    estado = d["estado"]
                    nombre = d["nombre"]
                    sim = d.get("similitud")
                    margen = d.get("margen")
                    x1, y1, x2, y2 = d["bbox"]
                    clave_estado = "VERIFICANDO" if estado.startswith("Verificando") else estado
                    estados[clave_estado] += 1

                    writer.writerow([
                        i, frame_parte, frame_global, round(segundo, 2), d["track_id"],
                        nombre, estado,
                        "" if sim is None else round(sim, 4),
                        d.get("sim_top2") if d.get("sim_top2") is not None else "",
                        d.get("nombre2") or "",
                        margen if margen is not None else "",
                        d.get("motivo", ""), d.get("revision", False),
                        d.get("confianza_det", ""), x1, y1, x2, y2,
                    ])

                    if clave_estado in ("CONFIRMADO", "ANCLADO") and nombre in PERSONAS:
                        info = global_personas[nombre]
                        info["n"] += 1
                        info["partes"].add(i)
                        if sim is not None:
                            info["sims"].append(sim)
                        if nombre not in primera_confirmacion:
                            primera_confirmacion[nombre] = (frame_parte, round(segundo, 2), sim)
                        if info["primera_parte"] is None:
                            info["primera_parte"] = i
                            info["primer_frame"] = (frame_parte, round(segundo, 2))

                        clave = (i, nombre, d["track_id"], int(segundo))
                        if clave not in capturas:
                            capturas.add(clave)
                            destino = carpeta_est / nombre
                            destino.mkdir(parents=True, exist_ok=True)
                            sufijo = f"sim{sim:.2f}" if clave_estado == "CONFIRMADO" and sim is not None else "anc"
                            cv2.imwrite(str(destino / f"p{i:02d}_s{segundo:06.1f}_t{d['track_id']}_{sufijo}.jpg"),
                                        recorte(frame, (x1, y1, x2, y2)))

                    if clave_estado == "DESCONOCIDO" and sim is not None and sim >= SIMILITUD_DUDA \
                            and len(dudas) < MAX_DUDAS:
                        ruta = carpeta_dudas / (f"p{i:02d}_s{segundo:06.1f}_t{d['track_id']}"
                                                f"_sim{sim:.3f}_top2-{d.get('nombre2') or 'na'}.jpg")
                        carpeta_dudas.mkdir(parents=True, exist_ok=True)
                        guardado = cv2.imwrite(str(ruta), recorte(frame, (x1, y1, x2, y2)))
                        if guardado:
                            dudas.append((sim, ruta))

            cap.release()
            resumen_partes.append({
                "parte": i, "archivo": parte.name, "frames": frame_parte, "negros": negros,
                "caras": caras, "confirmados": sorted(primera_confirmacion),
                "desconocidos": estados.get("DESCONOCIDO", 0),
                "verificando": estados.get("VERIFICANDO", 0),
                "centrate": estados.get("CENTRATE", 0),
                "fraude": estados.get("FRAUDE", 0),
            })
            detalle = " ".join(f"{n}({v[2]:.2f}@{v[1]:.1f}s)" for n, v in sorted(primera_confirmacion.items()))
            print(f"  parte {i:02d}/{len(partes)} {parte.name} | {frame_parte} frames "
                  f"({negros} negros) | {caras} caras | confirmados: {detalle or '—'} "
                  f"| desconocidos {estados.get('DESCONOCIDO', 0)}")
    finally:
        archivo.close()

    with open(corrida / "resumen_por_parte.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["parte", "archivo", "frames", "negros", "caras",
                                          "confirmados", "desconocidos", "verificando",
                                          "centrate", "fraude"])
        w.writeheader()
        for fila in resumen_partes:
            fila = dict(fila)
            fila["confirmados"] = " ".join(fila["confirmados"])
            w.writerow(fila)

    with open(corrida / "resumen_personas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "confirmaciones", "sim_media", "sim_max", "sim_min",
                    "partes", "primera_parte", "primer_frame", "primer_segundo"])
        for nombre in sorted(global_personas):
            info = global_personas[nombre]
            sims = info["sims"]
            w.writerow([nombre, info["n"],
                        round(float(np.mean(sims)), 4) if sims else "",
                        round(float(np.max(sims)), 4) if sims else "",
                        round(float(np.min(sims)), 4) if sims else "",
                        " ".join(map(str, sorted(info["partes"]))),
                        info["primera_parte"],
                        info["primer_frame"][0] if info["primer_frame"] else "",
                        info["primer_frame"][1] if info["primer_frame"] else ""])

    for nombre in sorted(PERSONAS):
        guardar_contacto(carpeta_est / nombre, nombre, corrida)
    guardar_resumen_dudas(dudas, corrida)

    print("\n=== Resumen global ===")
    print(f"{'persona':<10} {'confirm.':>8} {'sim media':>9} {'sim max':>7} {'partes':>12} {'1er frame':>10}")
    for nombre in sorted(global_personas):
        info = global_personas[nombre]
        sims = info["sims"]
        print(f"{nombre:<10} {info['n']:8d} {np.mean(sims) if sims else 0:9.3f} "
              f"{np.max(sims) if sims else 0:7.3f} {len(info['partes']):12d} "
              f"{info['primer_frame'][0] if info['primer_frame'] else 0:10d}")

    print("\n=== Confirmados por parte ===")
    for fila in resumen_partes:
        print(f"  parte {fila['parte']:02d}: {', '.join(fila['confirmados']) or '—'}")

    faltan = sorted(PERSONAS - set(global_personas))
    print(f"\nSin ninguna confirmación: {faltan or 'ninguno'}")
    print(f"Desconocidos guardados para revisión (sim >= {SIMILITUD_DUDA}): {len(dudas)}")
    print(f"Corrida guardada en: {corrida}")


if __name__ == "__main__":
    main()

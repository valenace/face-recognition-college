"""
Genera el video de demostración del motor con su panel lateral.

- Procesa el video del salón desde `--desde` hasta el final.
- Panel derecho con los estudiantes registrados de la sección.
- Al confirmar a cada uno: marca de verificación y similitud.
- Termina cuando todos están confirmados (o cuando acaba el video).

No escribe en la base: es una corrida de visualización.

Uso:
    venv/bin/python experiments/unc/grabar_demo_motor.py
    venv/bin/python experiments/unc/grabar_demo_motor.py --desde 36
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
sys.path.append(str(Path(__file__).resolve().parent))

from demo_util import (  # noqa: E402
    caja, check, crear_writer, formato_tiempo, franja, rellenar, sin_acentos, texto,
)

from core.db_config import PREFIJO_ALIAS, conectar  # noqa: E402

VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
SALIDA = RAIZ / "experiments/unc/demo_videos/demo_motor.mp4"

ANCHO, ALTO = 1280, 720
CABECERA = 56
ANCHO_VIDEO, ALTO_VIDEO = 960, 540
PANEL_X = ANCHO_VIDEO
ALTO_PANEL = ALTO - CABECERA

COLOR_FONDO = (28, 28, 30)
COLOR_PANEL = (36, 36, 40)
COLOR_LINEA = (70, 70, 76)
COLOR_TEXTO = (235, 235, 235)
COLOR_SUAVE = (150, 150, 156)
COLOR_OK = (80, 220, 120)
COLOR_REVISION = (0, 215, 255)  # amarillo
COLOR_DESCONOCIDO = (0, 0, 255)


def roster():
    """Estudiantes con vector facial, ordenados por nombre."""
    with conectar() as conexion:
        with conexion.cursor() as cur:
            cur.execute(
                """
                SELECT u.username, u.first_name, u.last_name
                FROM attendance_faceembedding e
                JOIN users_user u ON u.id = e.user_id
                ORDER BY u.first_name, u.last_name
                """
            )
            filas = cur.fetchall()

    personas = []
    for username, nombre, apellido in filas:
        alias = username[len(PREFIJO_ALIAS):] if username.startswith(PREFIJO_ALIAS) else username
        personas.append({
            "alias": alias,
            "nombre": sin_acentos(f"{nombre} {apellido}".strip()),
            "confirmado": False,
            "sim": None,
        })
    return personas


def dibujar_decisiones(frame, decisiones, escala_x, escala_y):
    for d in decisiones:
        x1 = int(d["bbox"][0] * escala_x)
        y1 = int(d["bbox"][1] * escala_y)
        x2 = int(d["bbox"][2] * escala_x)
        y2 = int(d["bbox"][3] * escala_y)
        if d["estado"] == "DESCONOCIDO":
            color = COLOR_DESCONOCIDO
            caja(frame, x1, y1, x2 - x1, y2 - y1, color, 2)
            continue
        color = COLOR_REVISION if d.get("revision") else (0, 220, 60)
        if d["estado"] in ("FRAUDE", "ERROR"):
            color = COLOR_DESCONOCIDO
        caja(frame, x1, y1, x2 - x1, y2 - y1, color, 2)
        etiqueta = d["nombre"]
        if d.get("revision"):
            etiqueta = "posible " + etiqueta
        if d.get("similitud") is not None:
            etiqueta += f" {d['similitud']:.2f}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        franja(frame, x1, max(0, y1 - th - 8), tw + 8, th + 10, color, 0.95)
        texto(frame, etiqueta, (x1 + 4, max(th, y1 - 3)), 0.55, (15, 15, 15), 2)


def dibujar_panel(panel, personas):
    rellenar(panel, 0, 0, panel.shape[1], panel.shape[0], COLOR_PANEL)

    texto(panel, "ESTUDIANTES", (18, 28), 0.65, COLOR_TEXTO, 2)
    texto(panel, f"{len(personas)} registrados", (18, 52), 0.45, COLOR_SUAVE, 1)

    y = 104
    alto_fila = 42
    for persona in personas:
        if persona["confirmado"]:
            franja(panel, 10, y - 26, panel.shape[1] - 20, alto_fila - 6, (30, 70, 45), 0.85)
            color = COLOR_OK
        else:
            color = COLOR_SUAVE

        caja(panel, 18, y - 18, 16, 16, color, 1)
        if persona["confirmado"]:
            check(panel, 17, y - 20, 18, COLOR_OK, 2)

        cadena = persona["nombre"][:22]
        if persona["sim"] is not None:
            cadena += f"   {persona['sim']:.2f}"
        texto(panel, cadena, (46, y), 0.55, color, 1)

        y += alto_fila

    # Resumen inferior
    y_resumen = y + 18
    caja(panel, 16, y_resumen - 14, panel.shape[1] - 32, 1, COLOR_LINEA, 1)
    return y_resumen + 18


def main():
    desde = 36.0
    for arg in sys.argv[1:]:
        if arg.startswith("--desde="):
            desde = float(arg.split("=", 1)[1])

    personas = roster()
    print(f"Registrados en el panel: {len(personas)}")
    for p in personas:
        print(f"  {p['nombre']}")

    from scripts.asistencia_unificada_optimizada import MotorAsistencia

    motor = MotorAsistencia(csv_path="/tmp/opencode/demo_motor_asistencia.csv")
    motor.tamano_minimo_rostro = 20

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"No se pudo abrir {VIDEO}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duracion_total = (total_frames - int(desde * fps)) / fps if total_frames else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(desde * fps))

    salida = crear_writer(SALIDA, fps, ANCHO, ALTO)
    ancho_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    alto_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    escala_x = ANCHO_VIDEO / ancho_original
    escala_y = ALTO_VIDEO / alto_original
    frames_escritos = 0
    confirmados = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            segundo = desde + frames_escritos / fps
            decisiones = motor.procesar_frame(frame)

            for d in decisiones:
                if d["estado"] not in ("CONFIRMADO", "ANCLADO"):
                    continue
                for persona in personas:
                    if persona["alias"] == d["nombre"]:
                        persona["confirmado"] = True
                        if d["similitud"] is not None:
                            persona["sim"] = max(persona["sim"] or 0.0, d["similitud"])
                        break
            confirmados = sum(1 for p in personas if p["confirmado"])

            # Lienzo: cabecera + video + panel
            lienzo = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
            lienzo[:] = COLOR_FONDO
            rellenar(lienzo, 0, 0, ANCHO, CABECERA, (20, 20, 22))
            texto(lienzo, "CV COLLEGE  |  CONTROL DE ASISTENCIA", (20, 37), 0.70, COLOR_TEXTO, 2)
            estado = "COMPLETADO" if confirmados == len(personas) else "RECONOCIENDO"
            color_estado = COLOR_OK if confirmados == len(personas) else COLOR_REVISION
            texto(lienzo, estado, (ANCHO - 300, 37), 0.55, color_estado, 1)
            tiempo = f"{formato_tiempo(segundo - desde)} / {formato_tiempo(duracion_total)}"
            texto(lienzo, tiempo, (ANCHO - 130, 37), 0.55, COLOR_SUAVE, 1)

            vista = cv2.resize(frame, (ANCHO_VIDEO, ALTO_VIDEO), interpolation=cv2.INTER_AREA)
            dibujar_decisiones(vista, decisiones, escala_x, escala_y)
            lienzo[CABECERA:CABECERA + ALTO_VIDEO, 0:ANCHO_VIDEO] = vista

            panel = lienzo[CABECERA:, PANEL_X:]
            y_resumen = dibujar_panel(panel, personas)
            texto(panel, f"Confirmados  {confirmados} / {len(personas)}", (18, y_resumen), 0.55, COLOR_TEXTO, 1)

            salida.write(lienzo)
            frames_escritos += 1

    finally:
        cap.release()
        salida.release()

    print(f"\nVideo: {SALIDA}")
    print(f"Frames: {frames_escritos} | duración: {frames_escritos / fps:.1f} s")
    print(f"Confirmados: {confirmados}/{len(personas)}")
    faltan = [p["nombre"] for p in personas if not p["confirmado"]]
    print(f"Sin confirmar: {faltan or 'ninguno'}")


if __name__ == "__main__":
    main()

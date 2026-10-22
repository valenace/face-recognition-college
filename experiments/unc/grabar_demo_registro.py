"""
Genera el video de demostración del registro biométrico de un estudiante.

- Reproduce el video de registro y marca el rostro detectado.
- Aplica los mismos filtros del registro real (alineación, nitidez, brillo y pose
  repetida) sobre el rostro ya alineado.
- Acumula a la derecha las fotos alineadas que se guardan.
- Al final muestra el centroide generado.

Uso:
    venv/bin/python experiments/unc/grabar_demo_registro.py --persona jesus
    venv/bin/python experiments/unc/grabar_demo_registro.py --persona alexa
    venv/bin/python experiments/unc/grabar_demo_registro.py --persona dani
"""

import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
sys.path.append(str(Path(__file__).resolve().parent))

from demo_util import caja, crear_writer, franja, rellenar, sin_acentos, texto  # noqa: E402

from core.db_config import PREFIJO_ALIAS, conectar  # noqa: E402

ANCHO, ALTO = 1280, 720
CABECERA = 56

VIDEO_X, VIDEO_Y, VIDEO_ANCHO, VIDEO_ALTO = 16, 76, 338, 600
GALERIA_X, GALERIA_Y = 380, 80
CELDA, HUECO, COLS, FILAS = 116, 8, 5, 4
STATS_X = 1010

MAX_FOTOS = 50
TAMANO_MINIMO = 60
BLUR_MINIMO = 60
BRILLO_MINIMO = 20
POSE_MAXIMA = 0.95

COLOR_FONDO = (28, 28, 30)
COLOR_PANEL = (36, 36, 40)
COLOR_LINEA = (70, 70, 76)
COLOR_TEXTO = (235, 235, 235)
COLOR_SUAVE = (150, 150, 156)
COLOR_OK = (80, 220, 120)
COLOR_RECHAZO = (0, 0, 255)
COLOR_AVISO = (0, 215, 255)


def nombre_de(persona):
    with conectar() as conexion:
        with conexion.cursor() as cur:
            cur.execute(
                "SELECT first_name, last_name FROM users_user WHERE username = %s",
                (f"{PREFIJO_ALIAS}{persona}",),
            )
            fila = cur.fetchone()
    return sin_acentos(f"{fila[0]} {fila[1]}".strip()) if fila else persona.capitalize()


def barra(frame, x, y, ancho, alto, fraccion, color=COLOR_OK):
    rellenar(frame, x, y, ancho, alto, (55, 55, 60))
    rellenar(frame, x, y, int(ancho * min(1.0, fraccion)), alto, color)


def main():
    persona = "jesus"
    for arg in sys.argv[1:]:
        if arg.startswith("--persona="):
            persona = arg.split("=", 1)[1].strip()

    carpeta = RAIZ / "data/unc/stds" / persona
    videos = sorted(carpeta.glob("*.mp4"))
    if not videos:
        print(f"No hay video de registro para {persona} en {carpeta}")
        return
    video = videos[0]
    nombre = nombre_de(persona)
    salida_ruta = RAIZ / f"experiments/unc/demo_videos/demo_registro_{persona}.mp4"

    from uniface import create_detector, create_recognizer
    from core.utils_facial import alinear_rostro, embedding_facial

    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    salida = crear_writer(salida_ruta, fps, ANCHO, ALTO)

    aceptadas = []
    vectores = []
    descartes = {"borrosa": 0, "pequena": 0, "oscura": 0, "pose": 0, "sin_rostro": 0}
    frames = 0
    ultimo_lienzo = None

    print(f"Registro de {nombre} | {video.name} | destino: {salida_ruta}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            rostros = detector.detect(frame)
            color = COLOR_SUAVE
            detalle = ""
            ancho_cara = 0
            nitidez = brillo = 0.0
            anotacion = None

            if not rostros:
                descartes["sin_rostro"] += 1
            else:
                rostro = max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
                x1, y1, x2, y2 = map(int, rostro.bbox)
                ancho_cara = x2 - x1

                motivo = None
                if ancho_cara < TAMANO_MINIMO:
                    motivo = "pequena"
                else:
                    alineado = alinear_rostro(frame, rostro.landmarks)
                    if alineado is None:
                        motivo = "pequena"
                    else:
                        gris = cv2.cvtColor(alineado, cv2.COLOR_BGR2GRAY)
                        nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
                        brillo = float(gris.mean())
                        if nitidez < BLUR_MINIMO:
                            motivo = "borrosa"
                        elif brillo < BRILLO_MINIMO:
                            motivo = "oscura"
                        else:
                            rostros_crop = detector.detect(alineado)
                            vector = None
                            if rostros_crop:
                                vector = embedding_facial(recognizer, alineado, rostros_crop[0].landmarks)
                                vector = vector / (np.linalg.norm(vector) + 1e-9)
                            if vector is None:
                                motivo = "borrosa"
                            else:
                                sim_pose = max((float(np.dot(vector, v)) for v in vectores), default=0.0)
                                if sim_pose > POSE_MAXIMA:
                                    motivo = "pose"
                                elif len(aceptadas) < MAX_FOTOS:
                                    aceptadas.append(alineado)
                                    vectores.append(vector)
                                    color = COLOR_OK
                                    detalle = f"{len(aceptadas)}/{MAX_FOTOS}"
                                else:
                                    detalle = f"{MAX_FOTOS}/{MAX_FOTOS}"

                if motivo is not None:
                    descartes[motivo] += 1
                    color = COLOR_AVISO if motivo == "pose" else COLOR_RECHAZO
                    detalle = motivo

                etiqueta = f"{ancho_cara}px | nitidez {nitidez:.0f} | brillo {brillo:.0f}"
                if detalle:
                    etiqueta += f" | {detalle}"
                anotacion = ((x1, y1, x2, y2), color, etiqueta)

            # Lienzo
            lienzo = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
            lienzo[:] = COLOR_FONDO
            rellenar(lienzo, 0, 0, ANCHO, CABECERA, (20, 20, 22))
            texto(lienzo, "REGISTRO BIOMETRICO", (20, 35), 0.62, COLOR_TEXTO, 2)
            texto(lienzo, nombre, (300, 35), 0.55, COLOR_OK, 1)
            texto(lienzo, f"{len(aceptadas)} / {MAX_FOTOS} fotos", (ANCHO - 190, 35), 0.55, COLOR_TEXTO, 1)

            alto_original, ancho_original = frame.shape[:2]
            vista = cv2.resize(frame, (VIDEO_ANCHO, VIDEO_ALTO), interpolation=cv2.INTER_AREA)
            if anotacion is not None:
                (bx1, by1, bx2, by2), color, etiqueta = anotacion
                ex = VIDEO_ANCHO / ancho_original
                ey = VIDEO_ALTO / alto_original
                vx1, vy1 = int(bx1 * ex), int(by1 * ey)
                vx2, vy2 = int(bx2 * ex), int(by2 * ey)
                caja(vista, vx1, vy1, vx2 - vx1, vy2 - vy1, color, 2)
                (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
                franja(vista, vx1, max(0, vy1 - th - 8), min(tw + 6, VIDEO_ANCHO - vx1), th + 8,
                       (10, 10, 10), 0.78)
                texto(vista, etiqueta, (vx1 + 3, max(th, vy1 - 4)), 0.34, color, 1)
            lienzo[VIDEO_Y:VIDEO_Y + VIDEO_ALTO, VIDEO_X:VIDEO_X + VIDEO_ANCHO] = vista
            caja(lienzo, VIDEO_X, VIDEO_Y, VIDEO_ANCHO, VIDEO_ALTO, COLOR_LINEA, 1)

            # Galeria con las ultimas 20 fotos
            rellenar(lienzo, GALERIA_X - 12, GALERIA_Y - 12,
                     COLS * (CELDA + HUECO) + 16, FILAS * (CELDA + HUECO) + 16, COLOR_PANEL)
            for indice, foto in enumerate(aceptadas[-(COLS * FILAS):]):
                col, fil = indice % COLS, indice // COLS
                x = GALERIA_X + col * (CELDA + HUECO)
                y = GALERIA_Y + fil * (CELDA + HUECO)
                lienzo[y:y + CELDA, x:x + CELDA] = cv2.resize(foto, (CELDA, CELDA))
                caja(lienzo, x, y, CELDA, CELDA, COLOR_LINEA, 1)

            # Estadisticas
            stats = [
                ("aceptadas", len(aceptadas), COLOR_OK),
                ("pose repetida", descartes["pose"], COLOR_AVISO),
                ("borrosas", descartes["borrosa"], COLOR_RECHAZO),
                ("pequenas", descartes["pequena"], COLOR_RECHAZO),
                ("oscuras", descartes["oscura"], COLOR_RECHAZO),
                ("sin rostro", descartes["sin_rostro"], COLOR_SUAVE),
            ]
            y = GALERIA_Y + 6
            for etiqueta, valor, color in stats:
                texto(lienzo, etiqueta, (STATS_X, y), 0.44, COLOR_SUAVE, 1)
                texto(lienzo, str(valor), (STATS_X + 150, y), 0.44, color, 1)
                y += 28

            # Progreso y filtros
            barra_y = ALTO - 96
            barra(lienzo, 380, barra_y, 884, 10, len(aceptadas) / MAX_FOTOS)
            texto(lienzo, f"filtros: tamano >= {TAMANO_MINIMO}px | nitidez >= {BLUR_MINIMO} | "
                          f"brillo >= {BRILLO_MINIMO} | pose <= {POSE_MAXIMA}",
                  (380, barra_y + 34), 0.40, COLOR_SUAVE, 1)

            salida.write(lienzo)
            ultimo_lienzo = lienzo
    finally:
        cap.release()

    # Cierre: pausa con el resumen del registro
    if ultimo_lienzo is not None:
        cierre = ultimo_lienzo.copy()
        franja(cierre, 0, ALTO // 2 - 60, ANCHO, 120, (20, 20, 22), 0.88)
        texto(cierre, "REGISTRO COMPLETADO", (ANCHO // 2 - 170, ALTO // 2 - 14), 0.75, COLOR_OK, 2)
        texto(cierre, f"{len(aceptadas)} fotos alineadas | centroide de 512 dimensiones",
              (ANCHO // 2 - 250, ALTO // 2 + 24), 0.55, COLOR_TEXTO, 1)
        for _ in range(int(fps * 2.5)):
            salida.write(cierre)

    salida.release()
    print(f"Frames: {frames} | fotos aceptadas: {len(aceptadas)}")
    print(f"Descartes: {descartes}")
    print(f"Video: {salida_ruta}")


if __name__ == "__main__":
    main()

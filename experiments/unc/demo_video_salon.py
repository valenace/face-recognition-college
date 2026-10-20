"""
Demo del motor de asistencia sobre un video, en dos modos:

1. Modo corrida (por defecto): guarda todo en una carpeta de la corrida.
     experiments/unc/demo/<carpeta>/
       demo_salon.mp4            video anotado
       decisiones.csv            una fila por decisión y frame
       asistencia.csv            confirmaciones de esta corrida
       estudiantes/<nombre>/*.jpg  capturas de cada reconocimiento (1 por segundo por pista)
       contacto_<nombre>.jpg     cuadrícula con las capturas de la persona
       dudosos/                  capturas con margen top-2 pequeño
2. Modo solo vista (--solo-vista): muestra el video anotado en una ventana y NO escribe
   nada en el proyecto. La asistencia se escribe en un archivo temporal del sistema
   que se borra al iniciar, solo porque el motor lo requiere internamente.

Uso:
  venv/bin/python experiments/unc/demo_video_salon.py [inicio] [duracion] [opciones]
Opciones:
  --video=1|2|<ruta>   video a procesar (1 y 2 son los del dataset UNC)
  --filtro=N           tamaño mínimo de rostro en px (por defecto 40)
  --mostrar            abre ventana en vivo
  --solo-vista         no guarda nada; implica ventana salvo --sin-ventana
  --sin-ventana        procesa sin abrir ventana
  --escala=X           escala de la ventana (por defecto 1.0)
  --carpeta=NNN        nombre de la carpeta de la corrida
  --margen=N           margen top-2 por debajo del cual se guarda una captura de duda

Controles con ventana:
  espacio = pausa/reanuda    n o → = avanzar 1 frame    b o ← = retroceder la vista 1 frame
  s = saltar 30 frames       d = imprimir detalles del frame    q = salir
  El retroceso es solo de visualización: el motor no retrocede su estado interno.
"""

import csv
import os
import sys
import tempfile
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.asistencia_unificada_optimizada import MotorAsistencia

RAIZ = Path(__file__).resolve().parents[2]
VIDEO = RAIZ / "data/unc/salon/VID_20260907_102851.mp4"
VIDEO_ALTERNO = RAIZ / "data/unc/salon/VID_20260907_105622.mp4"
DEMO = RAIZ / "experiments/unc/demo"
PERSONAS = {"alexa", "bob", "carlos", "dani", "fran", "jesus", "jose", "kate", "kevin"}
ANCHO_CAPTURA = 180
COLS_CONTACTO = 10
MAX_CONTACTO = 120
COLOR = {
    "CONFIRMADO": (0, 255, 0),
    "ANCLADO": (0, 255, 0),
    "VERIFICANDO": (0, 165, 255),
    "CENTRATE": (128, 128, 128),
    "ERROR": (0, 165, 255),
    "DESCONOCIDO": (0, 0, 255),
    "FRAUDE": (0, 0, 255),
}


def guardar_contacto(carpeta_estudiante, nombre, destino):
    capturas = sorted(carpeta_estudiante.glob("*.jpg"))[:MAX_CONTACTO]
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


def mostrar_detalles(decisiones):
    if not decisiones:
        print("  (sin caras en este frame)")
        return
    for d in decisiones:
        margen = d.get("margen")
        print(f"  #{d['track_id']:>3} {d['nombre']:<18} {str(d['estado']):<11} "
              f"sim={'—' if d['similitud'] is None else round(d['similitud'], 3)} "
              f"top2={d.get('nombre2') or '—'} "
              f"margen={'—' if margen is None else margen} "
              f"conf={d.get('confianza_det', '—')}")


def bucle_vista(frame, decisiones, estado, escala, fps, historial):
    """Dibuja el frame y maneja las teclas. estado: pausado, saltar, retroceso, salir."""
    ok, jpg = cv2.imencode(".jpg", frame)
    if ok:
        historial.append(jpg)
    estado["retroceso"] = 0

    vista = cv2.resize(frame, None, fx=escala, fy=escala) if escala != 1.0 else frame
    cv2.imshow("Demo asistencia", vista)

    if estado["saltar"] > 0:
        estado["saltar"] -= 1
        return

    while True:
        tecla = cv2.waitKey(0 if estado["pausado"] else max(1, int(1000 / fps))) & 0xFF
        if tecla == 0xFF and not estado["pausado"]:
            return
        if tecla == ord("q"):
            estado["salir"] = True
            return
        if tecla == ord(" "):
            estado["pausado"] = not estado["pausado"]
            return
        if tecla in (ord("n"), 13, 83, 3):  # avanzar un frame
            return
        if tecla == ord("s"):
            estado["saltar"] += 30
            return
        if tecla in (ord("b"), 81):  # retroceder la vista
            estado["retroceso"] = min(estado["retroceso"] + 1, len(historial) - 1)
            idx = len(historial) - 1 - estado["retroceso"]
            img = cv2.imdecode(historial[idx], cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imshow("Demo asistencia",
                           cv2.resize(img, None, fx=escala, fy=escala) if escala != 1.0 else img)
            continue
        if tecla == ord("d"):
            mostrar_detalles(decisiones)
            continue
        if not estado["pausado"]:
            return


def main():
    solo_vista = "--solo-vista" in sys.argv
    mostrar = "--mostrar" in sys.argv or (solo_vista and "--sin-ventana" not in sys.argv)
    filtro, escala, carpeta, margen_duda = 40, 1.0, None, 0.10
    video = VIDEO
    for a in sys.argv[1:]:
        if a.startswith("--filtro="):
            filtro = int(a.split("=", 1)[1])
        if a.startswith("--escala="):
            escala = float(a.split("=", 1)[1])
        if a.startswith("--carpeta="):
            carpeta = a.split("=", 1)[1]
        if a.startswith("--margen="):
            margen_duda = float(a.split("=", 1)[1])
        if a.startswith("--video="):
            elegido = a.split("=", 1)[1]
            video = VIDEO_ALTERNO if elegido == "2" else Path(elegido)
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    inicio = float(argumentos[0]) if len(argumentos) > 0 else 0.0
    duracion = float(argumentos[1]) if len(argumentos) > 1 else 30.0

    if solo_vista:
        carpeta_corrida = None
        ruta_asistencia = Path(tempfile.gettempdir()) / "cv_college_vista_temporal.csv"
        ruta_asistencia.unlink(missing_ok=True)
    else:
        if carpeta is None:
            carpeta = f"prueba_{datetime.now():%d%m_%H%M}"
        carpeta_corrida = DEMO / carpeta
        carpeta_corrida.mkdir(parents=True, exist_ok=True)
        ruta_asistencia = carpeta_corrida / "asistencia.csv"
    carpeta_estudiantes = (carpeta_corrida / "estudiantes") if carpeta_corrida else None
    print(f"Modo: {'solo vista (no se guarda nada)' if solo_vista else carpeta_corrida}")

    motor = MotorAsistencia(
        db_path=str(RAIZ / "data/database_embeddings.pkl"),
        csv_path=str(ruta_asistencia),
    )
    motor.tamano_minimo_rostro = filtro
    print(f"Filtro de tamaño mínimo: {filtro} px")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio * fps))
    total = int(duracion * fps)
    print(f"Video: {video.name} | {ancho}x{alto} | {fps:.2f} fps | desde {inicio:.1f} s por {duracion:.1f} s")

    salida = None
    if not solo_vista:
        salida = cv2.VideoWriter(str(carpeta_corrida / "demo_salon.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))

    capturas_guardadas = set()
    dudas = []
    conteo_estados = Counter()
    confirmados = Counter()

    # Controles de la ventana
    estado = {"pausado": False, "saltar": 0, "retroceso": 0, "salir": False}
    historial = deque(maxlen=300)  # frames anotados comprimidos, para retroceder la vista
    if mostrar:
        print("\nControles: espacio=pausa  n/flecha=avanzar  b=flecha izq=retroceder  "
              "s=saltar 30  d=detalles del frame  q=salir\n")

    archivo_decisiones = open(carpeta_corrida / "decisiones.csv", "w", newline="", encoding="utf-8") if not solo_vista else None
    writer = None
    if archivo_decisiones:
        writer = csv.writer(archivo_decisiones)
        writer.writerow(["frame_video", "segundo", "track_id", "nombre", "similitud", "estado",
                         "conf_det", "sim_top2", "nombre2", "margen", "x1", "y1", "x2", "y2"])

    try:
        for k in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            segundo = inicio + k / fps
            decisiones = motor.procesar_frame(frame)

            for d in decisiones:
                conteo_estados[d["estado"]] += 1
                if d["estado"] in ("CONFIRMADO", "ANCLADO"):
                    confirmados[d["nombre"]] += 1

                x1, y1, x2, y2 = d["bbox"]
                color = COLOR.get(d["estado"], (0, 0, 255))
                if d["estado"] == "DESCONOCIDO":
                    # Solo caja roja y confianza de detección, sin etiqueta de identidad
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    conf = d.get("confianza_det")
                    texto_conf = f"{conf:.2f}" if conf is not None else ""
                    if texto_conf:
                        cv2.putText(frame, texto_conf, (x1 + 2, max(12, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    texto = f"{d['nombre']} {d['similitud']:.2f} #{d['track_id']}" if d["similitud"] is not None else d["nombre"]
                    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, texto, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                if writer:
                    writer.writerow([k, round(segundo, 2), d["track_id"], d["nombre"],
                                     "" if d["similitud"] is None else round(d["similitud"], 4),
                                     d["estado"], d.get("confianza_det", ""),
                                     d.get("sim_top2", ""), d.get("nombre2", ""), d.get("margen", ""),
                                     x1, y1, x2, y2])

                if carpeta_estudiantes and d["nombre"] in PERSONAS and d["estado"] in ("CONFIRMADO", "ANCLADO"):
                    clave = (d["nombre"], d["track_id"], int(segundo))
                    if clave not in capturas_guardadas:
                        capturas_guardadas.add(clave)
                        mx, my = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
                        a, b = max(0, x1 - mx), max(0, y1 - my)
                        c, e = min(ancho, x2 + mx), min(alto, y2 + my)
                        carpeta_est = carpeta_estudiantes / d["nombre"]
                        carpeta_est.mkdir(parents=True, exist_ok=True)
                        sufijo = f"sim{d['similitud']:.2f}" if d["estado"] == "CONFIRMADO" else "anc"
                        cv2.imwrite(str(carpeta_est / f"s{segundo:06.1f}_p{d['track_id']}_{sufijo}.jpg"),
                                    frame[b:e, a:c])

                margen = d.get("margen")
                if carpeta_corrida and margen is not None and margen < margen_duda and d.get("nombre2"):
                    clave_duda = ("duda", d["track_id"], int(segundo))
                    if clave_duda not in capturas_guardadas:
                        capturas_guardadas.add(clave_duda)
                        mx, my = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
                        a, b = max(0, x1 - mx), max(0, y1 - my)
                        c, e = min(ancho, x2 + mx), min(alto, y2 + my)
                        ruta_duda = carpeta_corrida / "dudosos" / str(d["nombre"]) / f"s{segundo:06.1f}_p{d['track_id']}_m{margen:.2f}_top2-{d['nombre2']}.jpg"
                        ruta_duda.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(ruta_duda), frame[b:e, a:c])
                        dudas.append((margen, ruta_duda, d["nombre"], d["nombre2"]))

            hud = [f"segundo {segundo:6.2f}", f"caras en el frame: {len(decisiones)}",
                   f"asistencia hoy: {len(motor.asistencia_hoy)}"]
            for i, t in enumerate(hud):
                cv2.putText(frame, t, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                cv2.putText(frame, t, (12, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

            if salida:
                salida.write(frame)
            if mostrar:
                bucle_vista(frame, decisiones, estado, escala, fps, historial)
                if estado["salir"]:
                    print("Detenido por el usuario (q).")
                    break
            if k % 150 == 0:
                print(f"  procesado {k}/{total} frames (segundo {segundo:.1f})")
    finally:
        cap.release()
        if salida:
            salida.release()
        if archivo_decisiones:
            archivo_decisiones.close()
        if mostrar:
            cv2.destroyAllWindows()

    print("\nResumen de la corrida:")
    print(f"  decisiones: {sum(conteo_estados.values())}")
    print(f"  estados: {dict(conteo_estados)}")
    print(f"  personas confirmadas: {sorted(confirmados)}")
    print(f"  asistencia registrada: {sorted(motor.asistencia_hoy)}")

    if solo_vista:
        ruta_asistencia.unlink(missing_ok=True)
        print("  Modo solo vista: no se guardó nada en el proyecto.")
        return

    for nombre in sorted(PERSONAS):
        guardar_contacto(carpeta_estudiantes / nombre, nombre, carpeta_corrida)

    if dudas:
        pares = Counter(tuple(sorted((a, b))) for _, _, a, b in dudas)
        print("\nPares que más aparecen en las dudas:")
        for (a, b), n in pares.most_common(8):
            print(f"  {a} / {b}: {n}")
        dudas.sort(key=lambda x: x[0])
        imgs = []
        for margen, ruta, a, b in dudas[:40]:
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            img = cv2.resize(img, (170, 170))
            cv2.putText(img, f"{a}/{b} m={margen:.2f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
            imgs.append(img)
        while len(imgs) % 10:
            imgs.append(np.full((170, 170, 3), 255, np.uint8))
        filas = [np.hstack(imgs[i:i + 10]) for i in range(0, len(imgs), 10)]
        cv2.imwrite(str(carpeta_corrida / "dudosos_resumen.jpg"), np.vstack(filas))
        print(f"Capturas dudosas: {len(dudas)} en {carpeta_corrida / 'dudosos'}")

    print(f"\nCorrida guardada en: {carpeta_corrida}")


if __name__ == "__main__":
    main()

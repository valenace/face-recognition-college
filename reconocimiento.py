"""
reconocimiento.py — Sistema de Reconocimiento Facial y Asistencia (UniFace v2)
"""

import cv2
import numpy as np
import pickle
import time
import csv
import os
from datetime import datetime
from uniface import create_detector, create_recognizer
import supervision as sv
from utils_facial import aplicar_clahe


# ============================================================
# ⚙️ CONFIGURACIÓN
# ============================================================
DB_FILE = "database_embeddings.pkl"
ASISTENCIA_FILE = "asistencia.csv"

UMBRAL_SIMILITUD = 0.35
TAMANO_MINIMO_ROSTRO = 30


# ============================================================
# 🚀 INICIALIZACIÓN
# ============================================================
print("=" * 60)
print("  SISTEMA DE RECONOCIMIENTO Y ASISTENCIA — UniFace v2")
print("=" * 60)

if not os.path.exists(DB_FILE):
    print(f"\n❌ Error: No se encontró la base de datos '{DB_FILE}'.")
    print("   Ejecuta primero: python3 registro.py")
    exit()

with open(DB_FILE, "rb") as f:
    db = pickle.load(f)

print(f"\n[INFO] Base de datos cargada: {len(db)} personas registradas.")
for nombre in db:
    print(f"  → {nombre}")

print("\n[INFO] Cargando motores UniFace...")
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')
print("[OK] RetinaFace + ArcFace listos.")

tracker = sv.ByteTrack()
print("[OK] ByteTrack inicializado.")

identidades_ancladas = {}
asistencia_registrada = set()

if not os.path.exists(ASISTENCIA_FILE):
    with open(ASISTENCIA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre", "Fecha", "Hora", "Similitud"])
    print(f"[INFO] Archivo de asistencia creado: {ASISTENCIA_FILE}")
else:
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(ASISTENCIA_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[1] == fecha_hoy:
                    asistencia_registrada.add(row[0])
        if asistencia_registrada:
            print(f"[INFO] Ya registrados hoy: {', '.join(asistencia_registrada)}")
    except Exception:
        pass


def registrar_asistencia(nombre, similitud):
    if nombre in asistencia_registrada or nombre == "Desconocido":
        return

    ahora = datetime.now()
    with open(ASISTENCIA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            nombre,
            ahora.strftime("%Y-%m-%d"),
            ahora.strftime("%H:%M:%S"),
            f"{similitud:.4f}"
        ])

    asistencia_registrada.add(nombre)
    print(f"  📋 ASISTENCIA REGISTRADA: {nombre} a las {ahora.strftime('%H:%M:%S')}")


# ============================================================
# 🎥 BUCLE PRINCIPAL DE RECONOCIMIENTO
# ============================================================
WINDOW_NAME = "Reconocimiento y Asistencia - UniFace v2"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: No se pudo acceder a la cámara.")
    exit()

print(f"\n{'=' * 60}")
print("  🟢 SISTEMA ACTIVO — Presiona 'q' para salir")
print(f"{'=' * 60}\n")

fps = 0
tiempo_previo = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rostros = detector.detect(frame)

    detections_list = []
    rostros_data = []

    for rostro in rostros:
        x1, y1, x2, y2 = map(int, rostro.bbox)
        ancho = x2 - x1

        if ancho > TAMANO_MINIMO_ROSTRO and x2 > x1 and y2 > y1:
            detections_list.append([x1, y1, x2, y2, rostro.confidence])
            rostros_data.append(rostro)

    if len(detections_list) > 0:
        detections = sv.Detections(
            xyxy=np.array([d[:4] for d in detections_list]),
            confidence=np.array([d[4] for d in detections_list])
        )

        tracked_detections = tracker.update_with_detections(detections=detections)

        for track in tracked_detections:
            x1, y1, x2, y2 = map(int, track[0])
            tracker_id = track[4]
            nombre_mostrar = "Desconocido"
            similitud_mostrar = 0.0

            if tracker_id in identidades_ancladas:
                nombre_mostrar = identidades_ancladas[tracker_id]
            else:
                for rostro in rostros_data:
                    rx1, ry1, rx2, ry2 = map(int, rostro.bbox)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        frame_ecualizado = aplicar_clahe(frame)

                        vec_actual = recognizer.get_normalized_embedding(
                            frame_ecualizado, rostro.landmarks
                        )

                        max_similitud = -1.0
                        candidato = "Desconocido"

                        for nombre_db, vec_db in db.items():
                            similitud = np.dot(
                                vec_actual.flatten(), vec_db.flatten()
                            )
                            if similitud > max_similitud:
                                max_similitud = similitud
                                candidato = nombre_db

                        similitud_mostrar = max_similitud

                        print(
                            f"  👀 [ID:{tracker_id}] Se parece un "
                            f"{max_similitud:.2f} a {candidato}"
                        )

                        if max_similitud > UMBRAL_SIMILITUD:
                            nombre_mostrar = candidato
                            identidades_ancladas[tracker_id] = nombre_mostrar
                            registrar_asistencia(nombre_mostrar, max_similitud)

                        break

            color = (0, 255, 0) if nombre_mostrar != "Desconocido" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            etiqueta = f"{nombre_mostrar} (ID:{tracker_id})"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            color_texto = (0, 0, 0) if nombre_mostrar != "Desconocido" else (255, 255, 255)
            cv2.putText(frame, etiqueta, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2)

    tiempo_actual = time.time()
    if tiempo_actual - tiempo_previo > 0:
        fps = 1 / (tiempo_actual - tiempo_previo)
    tiempo_previo = tiempo_actual
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Asistencia: {len(asistencia_registrada)} registros",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 60}")
print(f"  SESIÓN FINALIZADA")
print(f"  Asistencias registradas: {len(asistencia_registrada)}")
for nombre in asistencia_registrada:
    print(f"    ✅ {nombre}")
print(f"  Archivo: {ASISTENCIA_FILE}")
print(f"{'=' * 60}")

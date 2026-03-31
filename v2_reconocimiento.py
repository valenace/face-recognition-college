"""
v2_reconocimiento.py — Sistema de Reconocimiento Facial y Asistencia (Arquitectura FAISS)
"""

import cv2
import numpy as np
import pickle
import time
import csv
import os
import faiss
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
VOTOS_REQUERIDOS = 3

# ============================================================
# 📐 IoU
# ============================================================
def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# ============================================================
# 🚀 INICIALIZACIÓN
# ============================================================
print("=" * 60)
print("  SISTEMA DE RECONOCIMIENTO Y ASISTENCIA — FAISS Engine")
print("=" * 60)

if not os.path.exists(DB_FILE):
    print(f"\n❌ Error: No se encontró '{DB_FILE}'. Ejecuta registro.py primero.")
    exit()

with open(DB_FILE, "rb") as f:
    db = pickle.load(f)

print(f"\n[INFO] Construyendo Índice Vectorial FAISS para {len(db)} personas...")
nombres_lista = list(db.keys())
vectores_lista = list(db.values())

dimension = 512 
matriz_vectores = np.array(vectores_lista).astype('float32')

indice_faiss = faiss.IndexFlatIP(dimension)
indice_faiss.add(matriz_vectores)
print("[OK] Índice FAISS en memoria y optimizado.")

print("\n[INFO] Cargando motores UniFace (ONNX) y ByteTrack...")
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')
tracker = sv.ByteTrack()
print("[OK] Motores listos.")

identidades_ancladas = {}
votos_identidad = {}
asistencia_registrada = set()

if not os.path.exists(ASISTENCIA_FILE):
    with open(ASISTENCIA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre", "Fecha", "Hora", "Similitud_Promedio"])
else:
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(ASISTENCIA_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[1] == fecha_hoy:
                    asistencia_registrada.add(row[0])
    except Exception:
        pass

def registrar_asistencia(nombre, similitud):
    if nombre in asistencia_registrada or nombre == "Desconocido":
        return
    ahora = datetime.now()
    with open(ASISTENCIA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([nombre, ahora.strftime("%Y-%m-%d"), ahora.strftime("%H:%M:%S"), f"{similitud:.4f}"])
    asistencia_registrada.add(nombre)
    print(f"  📋 ASISTENCIA REGISTRADA: {nombre} ({similitud:.2f})")

# ============================================================
# 🎥 BUCLE PRINCIPAL DE RECONOCIMIENTO
# ============================================================
WINDOW_NAME = "Reconocimiento FAISS - UniFace v2"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error de cámara.")
    exit()

tiempo_previo = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        rostros = detector.detect(frame)
        detections_list = []
        rostros_data = []

        for rostro in rostros:
            x1, y1, x2, y2 = map(int, rostro.bbox)
            if (x2 - x1) > TAMANO_MINIMO_ROSTRO and x2 > x1 and y2 > y1:
                detections_list.append([x1, y1, x2, y2, rostro.confidence])
                rostros_data.append(rostro)

        if len(detections_list) > 0:
            detections = sv.Detections(
                xyxy=np.array([d[:4] for d in detections_list]),
                confidence=np.array([d[4] for d in detections_list])
            )
            tracked_detections = tracker.update_with_detections(detections=detections)

            for track in tracked_detections:
                t_x1, t_y1, t_x2, t_y2 = map(int, track[0])
                tracker_id = track[4]
                
                nombre_mostrar = "Analizando..."
                color = (0, 165, 255)

                if tracker_id in identidades_ancladas:
                    nombre_mostrar = identidades_ancladas[tracker_id]
                    color = (0, 255, 0)
                else:
                    mejor_iou = 0
                    rostro_asociado = None
                    
                    bbox_track = [t_x1, t_y1, t_x2, t_y2]
                    for rostro in rostros_data:
                        r_x1, r_y1, r_x2, r_y2 = map(int, rostro.bbox)
                        iou_actual = calcular_iou(bbox_track, [r_x1, r_y1, r_x2, r_y2])
                        
                        if iou_actual > mejor_iou:
                            mejor_iou = iou_actual
                            rostro_asociado = rostro

                    if mejor_iou > 0.5 and rostro_asociado is not None:
                        frame_ecualizado = aplicar_clahe(frame)
                        vec_actual = recognizer.get_normalized_embedding(
                            frame_ecualizado, rostro_asociado.landmarks
                        )

                        vec_actual_np = np.array([vec_actual.flatten()]).astype('float32')
                        distancias, indices = indice_faiss.search(vec_actual_np, 1)
                        
                        max_similitud = distancias[0][0]
                        idx_ganador = indices[0][0]

                        if max_similitud > UMBRAL_SIMILITUD:
                            candidato = nombres_lista[idx_ganador]
                            
                            if tracker_id not in votos_identidad:
                                votos_identidad[tracker_id] = {}
                            
                            votos_identidad[tracker_id][candidato] = votos_identidad[tracker_id].get(candidato, 0) + 1
                            votos_actuales = votos_identidad[tracker_id][candidato]
                            
                            print(f"  [ID:{tracker_id}] Voto {votos_actuales}/{VOTOS_REQUERIDOS} para {candidato} (Similitud: {max_similitud:.2f})")

                            if votos_actuales >= VOTOS_REQUERIDOS:
                                identidades_ancladas[tracker_id] = candidato
                                registrar_asistencia(candidato, max_similitud)
                                nombre_mostrar = candidato
                                color = (0, 255, 0)
                        else:
                            nombre_mostrar = "Desconocido"
                            color = (0, 0, 255)

                cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), color, 2)
                etiqueta = f"{nombre_mostrar} (ID:{tracker_id})"
                (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (t_x1, t_y1 - th - 10), (t_x1 + tw, t_y1), color, -1)
                
                color_texto = (0, 0, 0) if nombre_mostrar not in ["Desconocido", "Analizando..."] else (255, 255, 255)
                cv2.putText(frame, etiqueta, (t_x1, t_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2)

        t_actual = time.time()
        fps = 1 / (t_actual - tiempo_previo) if (t_actual - tiempo_previo) > 0 else 0
        tiempo_previo = t_actual
        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Asistencia: {len(asistencia_registrada)}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'=' * 60}")
    print(f"  SESIÓN FINALIZADA")
    print(f"  Asistencias registradas: {len(asistencia_registrada)}")
    print(f"{'=' * 60}")
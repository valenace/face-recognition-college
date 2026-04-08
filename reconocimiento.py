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

DB_FILE = "database_embeddings.pkl"
ASISTENCIA_FILE = "asistencia.csv"

UMBRAL_SIMILITUD = 0.35
TAMANO_MINIMO_ROSTRO = 30
VOTOS_REQUERIDOS = 3
FRAMES_LIMITE_REGISTRO = 100


def calcular_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


if not os.path.exists(DB_FILE):
    print(f"Error: no se encontro '{DB_FILE}'. Ejecuta registro.py primero.")
    exit()

with open(DB_FILE, "rb") as f:
    db = pickle.load(f)

nombres_lista = list(db.keys())
vectores_lista = list(db.values())
indice_faiss = faiss.IndexFlatIP(512)
indice_faiss.add(np.array(vectores_lista).astype('float32'))

print("Cargando modelos...")
detector = create_detector('retinaface')
recognizer = create_recognizer('arcface')
tracker = sv.ByteTrack()

identidades_ancladas = {}
votos_identidad = {}
asistencia_registrada = set()
memoria_liveness = {}

if not os.path.exists(ASISTENCIA_FILE):
    with open(ASISTENCIA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre", "Fecha", "Hora", "Similitud"])
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
    print(f"Asistencia: {nombre} ({similitud:.2f})")


WINDOW_NAME = "Reconocimiento"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

tempo_previo = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

                nombre_mostrar = "Validando..."
                color = (0, 165, 255)

                bbox_track = [t_x1, t_y1, t_x2, t_y2]
                mejor_iou, rostro_asociado = 0, None
                for rostro in rostros_data:
                    r_x1, r_y1, r_x2, r_y2 = map(int, rostro.bbox)
                    iou_actual = calcular_iou(bbox_track, [r_x1, r_y1, r_x2, r_y2])
                    if iou_actual > mejor_iou:
                        mejor_iou, rostro_asociado = iou_actual, rostro

                if mejor_iou < 0.5 or rostro_asociado is None:
                    continue

                # liveness por paralaje (yaw ratio)
                if tracker_id not in memoria_liveness:
                    memoria_liveness[tracker_id] = {
                        'frames_active': 0,
                        'head_turned': False,
                        'validated': False
                    }

                liveness_data = memoria_liveness[tracker_id]
                liveness_data['frames_active'] += 1

                # Extraer coordenadas X de los landmarks centrales (Ojo Izq, Ojo Der, Nariz)

                if len(rostro_asociado.landmarks) >= 3:
                    ojo_izq_x = rostro_asociado.landmarks[0][0]
                    ojo_der_x = rostro_asociado.landmarks[1][0]
                    nariz_x = rostro_asociado.landmarks[2][0]

                    dist_izq = abs(nariz_x - ojo_izq_x)
                    dist_der = abs(ojo_der_x - nariz_x)
                    ratio_yaw = dist_izq / (dist_der + 1e-6)

                    if not liveness_data['validated']:
                        if ratio_yaw > 1.5 or ratio_yaw < 0.6:
                            liveness_data['head_turned'] = True

                        if liveness_data['head_turned'] and 0.7 < ratio_yaw < 1.3:
                            liveness_data['validated'] = True
                        elif liveness_data['frames_active'] > FRAMES_LIMITE_REGISTRO:
                            cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), (0, 0, 255), 2)
                            cv2.putText(frame, "SPOOF DETECTADO", (t_x1, t_y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            continue

                    if not liveness_data['validated']:
                        instruccion = "Gira izq/der" if not liveness_data['head_turned'] else "Mira al frente"
                        cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), color, 2)
                        cv2.putText(frame, instruccion, (t_x1, t_y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        progreso = int((liveness_data['frames_active'] / FRAMES_LIMITE_REGISTRO) * (t_x2 - t_x1))
                        cv2.line(frame, (t_x1, t_y2 + 10), (t_x1 + progreso, t_y2 + 10), color, 3)
                        continue
                else:
                    continue

                if tracker_id in identidades_ancladas:
                    nombre_mostrar, color = identidades_ancladas[tracker_id], (0, 255, 0)
                else:
                    frame_ecualizado = aplicar_clahe(frame)
                    vec_actual = recognizer.get_normalized_embedding(frame_ecualizado, rostro_asociado.landmarks)
                    vec_np = np.array([vec_actual.flatten()]).astype('float32')
                    distancias, indices = indice_faiss.search(vec_np, 1)

                    max_similitud, idx_ganador = distancias[0][0], indices[0][0]

                    if max_similitud > UMBRAL_SIMILITUD:
                        candidato = nombres_lista[idx_ganador]
                        if tracker_id not in votos_identidad:
                            votos_identidad[tracker_id] = {}
                        votos_identidad[tracker_id][candidato] = votos_identidad[tracker_id].get(candidato, 0) + 1

                        if votos_identidad[tracker_id][candidato] >= VOTOS_REQUERIDOS:
                            identidades_ancladas[tracker_id] = candidato
                            registrar_asistencia(candidato, max_similitud)
                            nombre_mostrar, color = candidato, (0, 255, 0)
                    else:
                        nombre_mostrar, color = "Desconocido", (0, 0, 255)

                cv2.rectangle(frame, (t_x1, t_y1), (t_x2, t_y2), color, 2)
                etiqueta = f"{nombre_mostrar} (ID:{tracker_id})"
                (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (t_x1, t_y1 - th - 10), (t_x1 + tw, t_y1), color, -1)
                cv2.putText(frame, etiqueta, (t_x1, t_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        t_actual = time.time()
        fps = 1 / (t_actual - tempo_previo) if (t_actual - tempo_previo) > 0 else 0
        tempo_previo = t_actual
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Asistencia: {len(asistencia_registrada)}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSesion finalizada. Asistencias: {len(asistencia_registrada)}")
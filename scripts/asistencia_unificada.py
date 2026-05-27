"""
asistencia_unificada.py — Motor de Inferencia Biométrico en Tiempo Real
================================================================================
Arquitectura:
  - Tracking: ByteTrack para mantener IDs consistentes en multitudes.
  - Liveness: Gatekeeper activo con factor de escala (anti-pantallas/papel).
  - Reconocimiento: FAISS IndexFlatIP para búsquedas O(log N).
  - Tolerancia a Fallos: Consenso temporal de 3 frames (evita parpadeos de identidad).
"""

import cv2
import numpy as np
import pickle
import time
import csv
import faiss
from pathlib import Path
from datetime import datetime
import sys
import os

# Inyección al PYTHONPATH para encontrar la carpeta 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer
import supervision as sv
from core.utils_facial import aplicar_clahe


class MotorAsistencia:
    def __init__(
        self, db_path="data/database_embeddings.pkl", csv_path="data/asistencia.csv"
    ):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)

        self.umbral_similitud = 0.35
        self.umbral_liveness = 0.90
        self.tamano_minimo_rostro = 40
        self.votos_requeridos = 3

        print("\n[INFO] Inicializando Motor de Asistencia...")

        # Carga de la Base de Datos
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"No se encontró la base de datos en {self.db_path}"
            )

        with open(self.db_path, "rb") as f:
            self.db = pickle.load(f)

        self.nombres_lista = list(self.db.keys())
        vectores_lista = list(self.db.values())

        # Inicialización de FAISS
        print(
            f"[INFO] Compilando Índice FAISS para {len(self.nombres_lista)} estudiantes..."
        )
        self.indice_faiss = faiss.IndexFlatIP(512)
        self.indice_faiss.add(np.array(vectores_lista).astype("float32"))

        # Carga de Motores Neuronales
        self.detector = create_detector("retinaface")
        self.recognizer = create_recognizer("arcface")
        self.spoofer = create_spoofer()
        self.tracker = sv.ByteTrack()

        # Memoria de Estado (RAM)
        self.identidades_ancladas = {}
        self.votos_identidad = {}
        self.asistencia_hoy = set()

        self._preparar_archivo_csv()
        print("[OK] Sistema de Inferencia Listo.\n")

    def _preparar_archivo_csv(self):
        """Prepara el archivo CSV y carga los registros del día actual para no duplicar."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Identificador", "Fecha", "Hora", "Similitud"])
        else:
            fecha_hoy = datetime.now().strftime("%Y-%m-%d")
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Saltar cabecera
                    for row in reader:
                        if len(row) >= 2 and row[1] == fecha_hoy:
                            self.asistencia_hoy.add(row[0])
            except Exception as e:
                print(f"Advertencia al leer CSV: {e}")

    def _registrar_asistencia(self, nombre, similitud):
        """Escribe en el CSV de forma atómica."""
        if nombre in self.asistencia_hoy or nombre == "Desconocido":
            return

        ahora = datetime.now()
        fecha = ahora.strftime("%Y-%m-%d")
        hora = ahora.strftime("%H:%M:%S")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([nombre, fecha, hora, f"{similitud:.4f}"])

        self.asistencia_hoy.add(nombre)
        print(
            f"  ✅ ASISTENCIA CONFIRMADA: {nombre} | Similitud: {similitud:.2f} | Hora: {hora}"
        )

    @staticmethod
    def _calcular_iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def iniciar_vigilancia(self, origen_video=0):
        cap = cv2.VideoCapture(origen_video)
        if not cap.isOpened():
            print("❌ Error: No se pudo acceder a la cámara o flujo de video.")
            return

        cv2.namedWindow("Control de Asistencia", cv2.WINDOW_NORMAL)
        tiempo_previo = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rostros_raw = self.detector.detect(frame)
                detections_list = []
                rostros_data = []

                # Filtrar rostros muy pequeños (basura de fondo)
                for r in rostros_raw:
                    x1, y1, x2, y2 = map(int, r.bbox)
                    if (x2 - x1) > self.tamano_minimo_rostro:
                        detections_list.append([x1, y1, x2, y2, r.confidence])
                        rostros_data.append(r)

                if len(detections_list) > 0:
                    detections = sv.Detections(
                        xyxy=np.array([d[:4] for d in detections_list]),
                        confidence=np.array([d[4] for d in detections_list]),
                    )
                    tracked = self.tracker.update_with_detections(detections=detections)

                    for t in tracked:
                        x1, y1, x2, y2 = map(int, t[0])
                        t_id = t[4]

                       
                        h_frame, w_frame = frame.shape[:2]
                        margen = 15
                        
                        if x1 < margen or y1 < margen or x2 > (w_frame - margen) or y2 > (h_frame - margen):
                            nombre_mostrar = "Céntrate"
                            color = (0, 165, 255) 
                            self._dibujar_ui(frame, x1, y1, x2, y2, nombre_mostrar, color, t_id)
                            continue

                        # Buscar el objeto de rostro correspondiente al ID del tracker
                        mejor_iou, r_asociado = 0, None
                        for r in rostros_data:
                            iou = self._calcular_iou(
                                [x1, y1, x2, y2], list(map(int, r.bbox))
                            )
                            if iou > mejor_iou:
                                mejor_iou, r_asociado = iou, r

                        if mejor_iou < 0.5 or r_asociado is None:
                            continue

                        nombre_mostrar = "Analizando..."
                        color = (0, 165, 255)  # Naranja procesando

                        if t_id in self.identidades_ancladas:
                            nombre_mostrar = self.identidades_ancladas[t_id]
                            color = (0, 255, 0)  # Verde reconocido
                        else:
                            # 1. Filtro Liveness (Gatekeeper)
                            try:
                                res_spoof = self.spoofer.predict(frame, r_asociado.bbox)
                                if (
                                    not res_spoof.is_real
                                    or res_spoof.confidence < self.umbral_liveness
                                ):
                                    color = (0, 0, 255)
                                    nombre_mostrar = "FRAUDE DETECTADO"
                                    self._dibujar_ui(
                                        frame,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        nombre_mostrar,
                                        color,
                                        t_id,
                                    )
                                    continue  # Salta el reconocimiento
                            except Exception:
                                continue

                            # 2. Reconocimiento ArcFace + FAISS
                            f_ecualizado = aplicar_clahe(frame)
                            vec = self.recognizer.get_normalized_embedding(
                                f_ecualizado, r_asociado.landmarks
                            )

                            dist, idx = self.indice_faiss.search(
                                np.array([vec.flatten()]).astype("float32"), 1
                            )
                            max_sim = dist[0][0]

                            if max_sim > self.umbral_similitud:
                                candidato = self.nombres_lista[idx[0][0]]

                                # Lógica de Votación Temporal
                                if t_id not in self.votos_identidad:
                                    self.votos_identidad[t_id] = {}
                                self.votos_identidad[t_id][candidato] = (
                                    self.votos_identidad[t_id].get(candidato, 0) + 1
                                )

                                if (
                                    self.votos_identidad[t_id][candidato]
                                    >= self.votos_requeridos
                                ):
                                    self.identidades_ancladas[t_id] = candidato
                                    nombre_mostrar = candidato
                                    color = (0, 255, 0)
                                    self._registrar_asistencia(candidato, max_sim)
                                else:
                                    nombre_mostrar = f"Verificando {self.votos_identidad[t_id][candidato]}/{self.votos_requeridos}"
                            else:
                                nombre_mostrar = "Desconocido"
                                color = (0, 0, 255)

                        self._dibujar_ui(
                            frame, x1, y1, x2, y2, nombre_mostrar, color, t_id
                        )

                # Cálculo de FPS
                t_actual = time.time()
                fps = (
                    1 / (t_actual - tiempo_previo)
                    if (t_actual - tiempo_previo) > 0
                    else 0
                )
                tiempo_previo = t_actual

                # UI Global
                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Asistencia Hoy: {len(self.asistencia_hoy)}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Control de Asistencia", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n{'=' * 50}")
            print(f" SESIÓN DE ASISTENCIA CERRADA")
            print(f" Estudiantes procesados hoy: {len(self.asistencia_hoy)}")
            print(f"{'=' * 50}")

    def _dibujar_ui(self, frame, x1, y1, x2, y2, nombre, color, t_id):
        """Dibuja las cajas y etiquetas de forma limpia en el frame."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        etiqueta = nombre if nombre == "FRAUDE DETECTADO" else f"{nombre} (ID:{t_id})"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        color_texto = (
            (0, 0, 0)
            if color == (0, 255, 0) or color == (0, 165, 255)
            else (255, 255, 255)
        )
        cv2.putText(
            frame, etiqueta, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2
        )


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    try:
        sistema = MotorAsistencia()
        # Puedes pasarle el índice de la webcam (0) o la ruta a un video RTSP/MP4
        sistema.iniciar_vigilancia(origen_video=0)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(
            "Asegúrate de ejecutar el módulo de Registro y crear la base de datos primero."
        )

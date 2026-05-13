import cv2
import pickle
import numpy as np
from pathlib import Path
import sys
import os

# Añadir raiz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uniface import create_detector, create_recognizer
from uniface.spoofing import create_spoofer
from core.utils_facial import aplicar_clahe, es_imagen_borrosa, alinear_rostro


class RegistroBiometrico:
    def __init__(self, db_path="data/database_embeddings.pkl", max_muestras=15):
        self.db_path = Path(db_path)
        self.max_muestras = max_muestras
        self.umbral_blur = 60
        self.umbral_brillo = 20
        self.umbral_liveness = 0.90

        self.detector = create_detector("retinaface")
        self.recognizer = create_recognizer("arcface")
        self.spoofer = create_spoofer()

        self.db_embeddings = self._cargar_db()

    def _cargar_db(self):
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _guardar_db(self):
        ruta_temporal = self.db_path.with_suffix(".pkl.tmp")
        try:
            with open(ruta_temporal, "wb") as f:
                pickle.dump(self.db_embeddings, f)
            ruta_temporal.replace(self.db_path)
        except Exception as e:
            print(f"Error al guardar DB: {e}")
            if ruta_temporal.exists():
                ruta_temporal.unlink()

    def _evaluar_calidad(self, frame, rostro):
        x1, y1, x2, y2 = map(int, rostro.bbox)

        # Tamano minimo
        if (x2 - x1) < 60 or (y2 - y1) < 60:
            return False, "MUY LEJOS", (0, 165, 255)

        # Landmarks
        if rostro.landmarks is None or len(rostro.landmarks) < 5:
            return False, "SIN LANDMARKS", (0, 165, 255)

        rostro_alineado = alinear_rostro(frame, rostro.landmarks)
        if rostro_alineado is None:
            return False, "FALLO ALINEACION", (0, 0, 255)

        # Borrosidad
        if es_imagen_borrosa(rostro_alineado, umbral=self.umbral_blur):
            return False, "BORROSO", (0, 0, 255)

        # Iluminacion
        gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < self.umbral_brillo:
            return False, "MUY OSCURO", (0, 0, 255)

        # Liveness
        try:
            resultado_spoof = self.spoofer.predict(frame, rostro.bbox)
            if (
                not resultado_spoof.is_real
                or resultado_spoof.confidence < self.umbral_liveness
            ):
                return False, f"SPOOF ({resultado_spoof.confidence:.2f})", (0, 0, 255)
        except Exception:
            return False, "ERROR LIVENESS", (0, 0, 255)

        return True, "OK", (0, 255, 0), rostro_alineado

    def enrolar_usuario(self, id_usuario, origen_video):
        if id_usuario in self.db_embeddings:
            resp = input(
                f"Usuario {id_usuario} ya existe. Sobrescribir? (s/n): "
            ).lower()
            if resp != "s":
                return

        cap = cv2.VideoCapture(origen_video)
        if not cap.isOpened():
            print("Error al abrir origen de video")
            return

        es_video_archivo = isinstance(origen_video, str)
        cv2.namedWindow(f"Enrolamiento: {id_usuario}", cv2.WINDOW_NORMAL)

        vectores_extraidos = []
        frames_procesados = 0

        try:
            while len(vectores_extraidos) < self.max_muestras:
                ret, frame = cap.read()
                if not ret:
                    break

                frames_procesados += 1

                # Muestreo espaciado para videos pregrabados
                if es_video_archivo and frames_procesados % 5 != 0:
                    continue

                display = frame.copy()
                rostros = self.detector.detect(frame)

                mensaje_estado = "Buscando..."
                color_mensaje = (0, 165, 255)

                if len(rostros) > 1:
                    mensaje_estado = "MULTIPLES ROSTROS"
                    color_mensaje = (0, 0, 255)
                elif len(rostros) == 1:
                    rostro = rostros[0]
                    x1, y1, x2, y2 = map(int, rostro.bbox)

                    es_valido, msj_qa, color_qa, *data_extra = self._evaluar_calidad(
                        frame, rostro
                    )
                    mensaje_estado, color_mensaje = msj_qa, color_qa

                    if es_valido:
                        rostro_alineado = data_extra[0]
                        h_c, w_c = rostro_alineado.shape[:2]
                        if 10 + h_c < display.shape[0] and 10 + w_c < display.shape[1]:
                            display[10 : 10 + h_c, 10 : 10 + w_c] = rostro_alineado
                            cv2.rectangle(
                                display,
                                (10, 10),
                                (10 + w_c, 10 + h_c),
                                color_mensaje,
                                2,
                            )

                        img_limpia = aplicar_clahe(rostro_alineado)
                        rostros_crop = self.detector.detect(img_limpia)
                        if rostros_crop:
                            vec = self.recognizer.get_normalized_embedding(
                                img_limpia, rostros_crop[0].landmarks
                            )
                            vectores_extraidos.append(vec)

                    cv2.rectangle(display, (x1, y1), (x2, y2), color_mensaje, 2)

                # UI rendering
                h_f, w_f = display.shape[:2]
                progreso = len(vectores_extraidos) / self.max_muestras
                cv2.rectangle(
                    display, (0, h_f - 20), (int(progreso * w_f), h_f), (0, 255, 0), -1
                )
                cv2.putText(
                    display,
                    f"{int(progreso*100)}%",
                    (20, h_f - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    mensaje_estado,
                    (w_f // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_mensaje,
                    2,
                )

                cv2.imshow(f"Enrolamiento: {id_usuario}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Vectorizacion final
        if len(vectores_extraidos) > 0:
            centroide = np.mean(vectores_extraidos, axis=0)
            centroide_normalizado = centroide / np.linalg.norm(centroide)
            self.db_embeddings[id_usuario] = centroide_normalizado
            self._guardar_db()
            print("Registro exitoso.")
        else:
            print("Fallo el registro.")


if __name__ == "__main__":
    motor = RegistroBiometrico()

    while True:
        opcion = input("\n1. Webcam\n2. Video MP4\n3. Salir\nOpcion: ").strip()
        if opcion == "3":
            break

        if opcion in ["1", "2"]:
            id_alumno = input("ID Alumno (Ej: 1234_Nombre): ").strip()
            if not id_alumno:
                continue

            if opcion == "1":
                motor.enrolar_usuario(id_alumno, origen_video=0)
            else:
                ruta = input("Ruta MP4: ").strip()
                if Path(ruta).exists():
                    motor.enrolar_usuario(id_alumno, origen_video=ruta)
                else:
                    print("El archivo no existe.")

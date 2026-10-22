"""
asistencia_unificada_optimizada.py — Motor de Inferencia Biométrico en Tiempo Real
================================================================================
Arquitectura:
  - Tracking: ByteTrack para mantener IDs consistentes en multitudes. Se actualiza
    también en frames sin detecciones para envejecer pistas perdidas.
  - Reconocimiento: centroide por estudiante + FAISS IndexFlatIP (similitud coseno).
  - Anti-spoofing: se evalúa en CADA decisión de reconocimiento, sin importar el tamaño
    del rostro. Error del spoofer != fraude (no suma votos ni confirma identidad).
  - Consenso temporal: 3 votos consecutivos (ventana de frames) del mismo candidato.
  - Ancla de identidad: se revalida cada REVALIDAR_CADA frames.
  - Margen top-2: se registra la segunda mejor coincidencia para análisis; la decisión
    todavía NO depende del margen.
  - Preprocesado compartido con el registro (core.utils_facial.embedding_facial).
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
from core.utils_facial import embedding_facial

COLORES = {
    "CONFIRMADO": (0, 255, 0),
    "ANCLADO": (0, 255, 0),
    "VERIFICANDO": (0, 165, 255),
    "CENTRATE": (0, 165, 255),
    "ERROR": (0, 165, 255),
    "DESCONOCIDO": (0, 0, 255),
    "FRAUDE": (0, 0, 255),
}


class MotorAsistencia:
    def __init__(
        self,
        db_path="data/database_embeddings.pkl",
        csv_path="data/asistencia.csv",
        usar_bd=True,
    ):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.usar_bd = usar_bd

        # Calibrado en Fase 4 con decisión de 3 frames: 0.30 da precisión 1.000
        # y 0 falsos positivos sobre la muestra etiquetada (docs/analisis/unc/04_resultados_curva_roc.md §7).
        self.umbral_similitud = 0.30
        self.umbral_liveness = 0.90
        self.tamano_minimo_rostro = 40
        # Confianza mínima del detector para procesar una cara. RetinaFace genera
        # falsos positivos en brazos o ropa con confianza de 0.55 a 0.73
        # (ver docs/analisis/unc/02_resultados_registro.md §24).
        self.confianza_minima_deteccion = 0.75
        # Por debajo de este ancho (px) no se evalúa el anti-spoofing: el modelo
        # es inestable en caras pequeñas y genera falsas alarmas de fraude.
        # Decisión del responsable, ver docs/analisis/unc/02_resultados_registro.md §20.
        self.tamano_minimo_liveness = 60
        self.votos_requeridos = 3
        self.ventana_votos = 5
        # Revalidar la identidad anclada cada 30 frames (~1 s). Cuesta ~2.4 % más de
        # cómputo y muestra "Verificando" durante 2–3 frames; no hay interfaz en el
        # producto final. Ver docs/analisis/unc/02_resultados_registro.md §27.
        self.revalidar_cada = 30
        # Dos pistas activas en el mismo frame no pueden ser la misma persona.
        # Si dos reclaman la misma identidad y la diferencia de similitud supera este
        # margen, la peor se degrada a Desconocido (ver
        # docs/analisis/unc/09_incidente_identidad_dani.md).
        self.margen_conflicto = 0.05
        # Frames que se mantiene vetada una identidad para una pista que perdió el
        # conflicto, mientras la pista ganadora siga anclada.
        self.frames_veto = 90
        self.vetos_identidad = {}
        # Por debajo de esta similitud (y por encima del umbral) la identidad se
        # muestra como "posible": el sistema no puede afirmarla con seguridad.
        # No cambia la decisión ni la asistencia, solo la presentación.
        self.umbral_revision = 0.40

        print("\n[INFO] Inicializando Motor de Asistencia Optimizado...")

        if usar_bd:
            # La base vectorial (PostgreSQL + pgvector) es la fuente de verdad.
            from core.db_config import cargar_centroides

            print("[INFO] Cargando centroides desde pgvector...")
            self.db = cargar_centroides()
            origen = "PostgreSQL/pgvector"
            if not self.db:
                raise ValueError(
                    "La base vectorial está vacía: registra al menos un estudiante "
                    "(python manage.py importar_centroides)."
                )
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"No se encontró la base de datos en {self.db_path}"
                )
            with open(self.db_path, "rb") as f:
                self.db = pickle.load(f)
            origen = str(self.db_path)
            if not self.db:
                raise ValueError(f"La base de datos {self.db_path} está vacía")

        self.nombres_lista = list(self.db.keys())
        vectores = np.array([np.asarray(v).reshape(-1) for v in self.db.values()], dtype="float32")
        if vectores.shape[1] != 512:
            raise ValueError(f"Dimensión de embedding inesperada: {vectores.shape[1]}")
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        if not np.allclose(normas, 1.0, atol=1e-3):
            print("[AVISO] Embeddings no normalizados en la DB; se normalizan al cargar.")
            vectores = vectores / normas

        print(
            f"[INFO] Compilando Índice FAISS para {len(self.nombres_lista)} estudiantes "
            f"desde {origen}..."
        )
        self.indice_faiss = faiss.IndexFlatIP(512)
        self.indice_faiss.add(vectores)

        self.detector = create_detector("retinaface")
        self.recognizer = create_recognizer("arcface")
        self.spoofer = create_spoofer()
        self.tracker = sv.ByteTrack()

        self.frame_idx = 0
        self.identidades_ancladas = {}
        self.votos_identidad = {}
        self.asistencia_hoy = set()
        self.fecha_actual = datetime.now().strftime("%Y-%m-%d")

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
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2 and row[1] == self.fecha_actual:
                            self.asistencia_hoy.add(row[0])
            except Exception as e:
                print(f"Advertencia al leer CSV: {e}")

    def _registrar_asistencia(self, nombre, similitud):
        """Escribe en el CSV. Reinicia el conjunto diario si cambió el día."""
        ahora = datetime.now()
        fecha = ahora.strftime("%Y-%m-%d")
        if fecha != self.fecha_actual:
            self.fecha_actual = fecha
            self.asistencia_hoy = set()

        if nombre in self.asistencia_hoy or nombre == "Desconocido":
            return

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

    def _contar_voto(self, t_id, candidato):
        voto = self.votos_identidad.get(t_id)
        if (
            voto is None
            or voto["candidato"] != candidato
            or self.frame_idx - voto["ultimo"] > self.ventana_votos
        ):
            voto = {"candidato": candidato, "conteo": 0, "ultimo": self.frame_idx}
            self.votos_identidad[t_id] = voto
        voto["conteo"] += 1
        voto["ultimo"] = self.frame_idx
        return voto["conteo"]

    def _evaluar_liveness(self, frame, bbox):
        ancho = bbox[2] - bbox[0]
        if ancho < self.tamano_minimo_liveness:
            return "omitido"
        try:
            res = self.spoofer.predict(frame, bbox)
        except Exception:
            return "error"
        if res.is_real and res.confidence >= self.umbral_liveness:
            return "real"
        return "fraude"

    def _decidir_identidad(self, frame, t_id, rostro):
        ancla = self.identidades_ancladas.get(t_id)
        if ancla is not None:
            if self.frame_idx - ancla["frame"] < self.revalidar_cada:
                return ancla["nombre"], ancla["similitud"], "ANCLADO", None, None, ancla["nombre"]
            del self.identidades_ancladas[t_id]

        vec = embedding_facial(self.recognizer, frame, rostro.landmarks)
        dist, idx = self.indice_faiss.search(vec.reshape(1, -1).astype("float32"), 2)
        sim = float(dist[0][0])
        sim2 = float(dist[0][1])
        nombre2 = self.nombres_lista[idx[0][1]]

        if sim <= self.umbral_similitud:
            self.votos_identidad.pop(t_id, None)
            return "Desconocido", sim, "DESCONOCIDO", sim2, nombre2, self.nombres_lista[idx[0][0]]

        candidato = self.nombres_lista[idx[0][0]]

        # Si esta pista perdió un conflicto de identidad, se mantiene como
        # Desconocido mientras la pista ganadora siga anclada.
        veto = self.vetos_identidad.get(t_id)
        if (
            veto
            and veto["nombre"] == candidato
            and self.frame_idx <= veto["hasta"]
            and veto["contra"] in self.identidades_ancladas
        ):
            self.votos_identidad.pop(t_id, None)
            return "Desconocido", sim, "DESCONOCIDO", sim2, nombre2, candidato

        liveness = self._evaluar_liveness(frame, rostro.bbox)
        if liveness == "fraude":  # "omitido" (cara pequeña) continúa como válido
            self.votos_identidad.pop(t_id, None)
            return "FRAUDE DETECTADO", sim, "FRAUDE", sim2, nombre2, candidato
        if liveness == "error":
            return "Error liveness", sim, "ERROR", sim2, nombre2, candidato

        conteo = self._contar_voto(t_id, candidato)
        if conteo >= self.votos_requeridos:
            self.identidades_ancladas[t_id] = {
                "nombre": candidato,
                "similitud": sim,
                "frame": self.frame_idx,
            }
            self.votos_identidad.pop(t_id, None)
            self._registrar_asistencia(candidato, sim)
            return candidato, sim, "CONFIRMADO", sim2, nombre2, candidato

        return f"Verificando {conteo}/{self.votos_requeridos}", sim, "VERIFICANDO", sim2, nombre2, candidato

    def _marcar_revision(self, decisiones):
        """Marca las identidades de baja confianza para revisión visual.

        No cambia la decisión ni la asistencia: solo indica que la similitud está
        entre el umbral y `umbral_revision`, donde el sistema no puede afirmar la
        identidad con seguridad (banda 0.30–0.40).
        """
        for decision in decisiones:
            confirmada = decision["estado"] in ("CONFIRMADO", "ANCLADO")
            sim = decision.get("similitud")
            decision["revision"] = bool(
                confirmada and sim is not None and sim < self.umbral_revision
            )
        return decisiones

    def _resolver_conflictos(self, decisiones):
        """Impide que dos pistas simultáneas compartan la misma identidad.

        Dos caras no pueden ser la misma persona en el mismo frame. Si dos pistas
        confirman el mismo nombre, se conserva la de mayor similitud y la otra vuelve
        a Desconocido. Si la diferencia es menor que `margen_conflicto`, se dejan ambas
        y se marca el motivo como ambiguo (puede ser un fragmento del tracker).
        """
        por_nombre = {}
        for decision in decisiones:
            if decision["estado"] in ("CONFIRMADO", "ANCLADO") and decision.get("similitud") is not None:
                por_nombre.setdefault(decision["nombre"], []).append(decision)

        for nombre, grupo in por_nombre.items():
            if len(grupo) < 2:
                continue
            grupo.sort(key=lambda d: d["similitud"], reverse=True)
            ganadora = grupo[0]
            for otra in grupo[1:]:
                diferencia = ganadora["similitud"] - otra["similitud"]
                if diferencia >= self.margen_conflicto:
                    self.identidades_ancladas.pop(otra["track_id"], None)
                    self.votos_identidad.pop(otra["track_id"], None)
                    self.vetos_identidad[otra["track_id"]] = {
                        "nombre": nombre,
                        "contra": ganadora["track_id"],
                        "hasta": self.frame_idx + self.frames_veto,
                    }
                    otra["estado"] = "DESCONOCIDO"
                    otra["nombre"] = "Desconocido"
                    otra["motivo"] = f"conflicto_con_pista_{ganadora['track_id']}"
                else:
                    otra["motivo"] = f"identidad_ambigua_con_pista_{ganadora['track_id']}"

        if self.frame_idx % 120 == 0:
            self.vetos_identidad = {
                track: veto for track, veto in self.vetos_identidad.items()
                if veto["hasta"] > self.frame_idx
            }
        return decisiones

    def procesar_frame(self, frame):
        """Procesa un frame y devuelve decisiones estructuradas (sin dibujar ni mostrar)."""
        self.frame_idx += 1
        h_frame, w_frame = frame.shape[:2]
        margen = 15

        rostros = [
            r for r in self.detector.detect(frame)
            if (r.bbox[2] - r.bbox[0]) > self.tamano_minimo_rostro
            and float(r.confidence) >= self.confianza_minima_deteccion
        ]

        if rostros:
            detections = sv.Detections(
                xyxy=np.array([r.bbox for r in rostros], dtype=float),
                confidence=np.array([r.confidence for r in rostros], dtype=float),
            )
        else:
            detections = sv.Detections.empty()
        tracked = self.tracker.update_with_detections(detections=detections)

        decisiones = []
        for t in tracked:
            x1, y1, x2, y2 = map(int, t[0])
            t_id = t[4]

            if x1 < margen or y1 < margen or x2 > (w_frame - margen) or y2 > (h_frame - margen):
                decisiones.append({"track_id": t_id, "bbox": (x1, y1, x2, y2),
                                   "nombre": "Céntrate", "similitud": None, "estado": "CENTRATE"})
                continue

            mejor_iou, r_asociado = 0, None
            for r in rostros:
                iou = self._calcular_iou([x1, y1, x2, y2], list(map(int, r.bbox)))
                if iou > mejor_iou:
                    mejor_iou, r_asociado = iou, r

            if mejor_iou < 0.5 or r_asociado is None:
                continue

            nombre, sim, estado, sim2, nombre2, candidato = self._decidir_identidad(frame, t_id, r_asociado)
            decisiones.append({
                "track_id": t_id, "bbox": (x1, y1, x2, y2),
                "nombre": nombre, "similitud": sim, "estado": estado,
                "confianza_det": round(float(r_asociado.confidence), 4),
                "sim_top2": None if sim2 is None else round(sim2, 4),
                "nombre2": nombre2,
                "candidato": candidato,
                "margen": None if sim2 is None else round(sim - sim2, 4),
            })

        self._marcar_revision(decisiones)
        return self._resolver_conflictos(decisiones)

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

                for d in self.procesar_frame(frame):
                    x1, y1, x2, y2 = d["bbox"]
                    self._dibujar_ui(frame, x1, y1, x2, y2, d["nombre"],
                                     COLORES.get(d["estado"], (0, 0, 255)), d["track_id"])

                t_actual = time.time()
                fps = 1 / (t_actual - tiempo_previo) if (t_actual - tiempo_previo) > 0 else 0
                tiempo_previo = t_actual

                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Asistencia Hoy: {len(self.asistencia_hoy)}",
                            (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Control de Asistencia", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n{'=' * 50}")
            print(" SESIÓN DE ASISTENCIA CERRADA")
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
        sistema.iniciar_vigilancia(origen_video=0)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(
            "Asegúrate de ejecutar el módulo de Registro y crear la base de datos primero."
        )

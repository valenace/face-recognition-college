"""
Servicio de asistencia en vivo.

Abre una cámara o un archivo de video, procesa cada frame con el motor calibrado
(`MotorAsistencia`), escribe la asistencia en la base y deja el último frame
anotado en memoria para servirlo por HTTP.

Usa exactamente el mismo motor y las mismas reglas que el procesamiento por
lotes y que los experimentos, de modo que "en vivo" y "grabado" se comportan
igual. Aceptar un archivo como fuente permite probar el flujo sin cámara.
"""

import threading
import time
import os

import cv2
from django.conf import settings
from django.db import close_old_connections

from core.db_config import alias_de
from core.registro_asistencia import aplicar_decisiones, mapa_estudiantes


def _dibujar(frame, decisiones):
    from scripts.asistencia_unificada_optimizada import COLORES

    for d in decisiones:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORES.get(d["estado"], (0, 0, 255))
        if d.get("revision"):
            color = (0, 215, 255)  # amarillo: identidad posible, revisar
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = f"posible {d['nombre']}" if d.get("revision") else d["nombre"]
        if d.get("similitud") is not None:
            texto += f" {d['similitud']:.2f}"
        cv2.putText(frame, texto, (x1 + 2, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


class ServicioEnVivo:
    def __init__(self):
        self._lock = threading.Lock()
        self._hilo = None
        self._activo = False
        self._jpeg = None
        self._estado = {
            "activo": False,
            "fuente": None,
            "sesion_id": None,
            "frames": 0,
            "fps": 0.0,
            "caras": 0,
            "confirmados": [],
            "error": None,
        }

    # ── API pública ────────────────────────────────────────────────
    def iniciar(self, fuente, sesion_id=None):
        with self._lock:
            if self._activo:
                return False, "Ya hay una captura en curso."
            self._activo = True
            self._jpeg = None
            self._estado = {
                "activo": True,
                "fuente": str(fuente),
                "sesion_id": sesion_id,
                "frames": 0,
                "fps": 0.0,
                "caras": 0,
                "confirmados": [],
                "error": None,
            }
        self._hilo = threading.Thread(
            target=self._bucle, args=(fuente, sesion_id), daemon=True
        )
        self._hilo.start()
        return True, "Captura iniciada."

    def detener(self):
        with self._lock:
            self._activo = False
        return True, "Captura detenida."

    def frame_jpeg(self):
        with self._lock:
            return self._jpeg

    def estado(self):
        with self._lock:
            return dict(self._estado)

    # ── Bucle de captura ───────────────────────────────────────────
    def _bucle(self, fuente, sesion_id):
        close_old_connections()
        sesion = estudiantes = profesor = alias_profesor = None
        motor = None
        try:
            from apps.attendance.models import SesionClase
            from scripts.asistencia_unificada_optimizada import MotorAsistencia

            if sesion_id:
                sesion = SesionClase.objects.select_related(
                    "asignacion_clase__profesor"
                ).get(id=sesion_id)
                asignacion = sesion.asignacion_clase
                estudiantes = mapa_estudiantes(asignacion)
                profesor = asignacion.profesor
                alias_profesor = alias_de(profesor.username, profesor.first_name)

            motor = MotorAsistencia(
                csv_path=os.path.join(settings.MEDIA_ROOT, "en_vivo_asistencia.csv")
            )
            motor.tamano_minimo_rostro = 20

            origen = int(fuente) if str(fuente).isdigit() else fuente
            captura = cv2.VideoCapture(origen)
            if not captura.isOpened():
                self._fallar(f"No se pudo abrir la fuente: {fuente}")
                return

            fps_video = captura.get(cv2.CAP_PROP_FPS) or 0
            espera = 1.0 / fps_video if fps_video > 0 else 0
            inicio = time.time()

            while self._activo:
                arranque = time.time()
                ok, frame = captura.read()
                if not ok:
                    break

                decisiones = motor.procesar_frame(frame)

                if sesion is not None:
                    resumen = aplicar_decisiones(
                        decisiones, frame, sesion, estudiantes,
                        alias_profesor=alias_profesor, profesor=profesor,
                    )
                else:
                    resumen = {"confirmados": []}

                anotado = _dibujar(frame, decisiones)
                ok, buffer = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buffer.tobytes()
                        self._estado["frames"] += 1
                        self._estado["caras"] = len(decisiones)
                        self._estado["fps"] = self._estado["frames"] / max(1e-6, time.time() - inicio)
                        confirmados = set(self._estado["confirmados"])
                        confirmados.update(resumen.get("confirmados", []))
                        self._estado["confirmados"] = sorted(confirmados)

                if espera:
                    time.sleep(max(0.0, espera - (time.time() - arranque)))

            captura.release()
        except Exception as e:
            self._fallar(str(e))
            return
        finally:
            close_old_connections()
            with self._lock:
                self._activo = False
                self._estado["activo"] = False
                if self._estado["error"] is None:
                    self._estado["mensaje"] = "Captura finalizada."

    def _fallar(self, mensaje):
        with self._lock:
            self._activo = False
            self._estado["activo"] = False
            self._estado["error"] = mensaje


servicio_en_vivo = ServicioEnVivo()

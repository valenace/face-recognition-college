"""
Procesamiento por lotes de imágenes para una sesión de clase.

Usa el motor calibrado (`MotorAsistencia`) sobre las imágenes en orden, como si
fueran frames de un video: mismo umbral (0.30), mismo filtro de tamaño (20 px),
misma confianza de detección (0.75), mismo anti-spoofing y la misma regla de
3 votos consecutivos que se validó en las fases 0–8.

Escribe la asistencia en la base (`RegistroAsistencia`). **No simula ni inventa
datos**: si no hay imágenes reales, deja el estado en `error` y termina.

Genera en `MEDIA_ROOT/batch_processed/<sesion_id>/`:
  status.json   progreso que consume el panel del profesor
  latest.jpg    último frame procesado y anotado
"""

import json
import os
import time

import cv2
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from apps.attendance.models import RegistroAsistencia, SesionClase
from core.db_config import alias_de
from core.registro_asistencia import (
    acumular_decisiones,
    mapa_estudiantes,
    nuevo_acumulado,
    registrar_lote,
    reiniciar_pistas,
)

MAX_IMAGENES = 60  # muestreo uniforme; el motor tarda ~130 ms por imagen


def _listar_imagenes(ruta_carpeta):
    if not ruta_carpeta or not os.path.isdir(ruta_carpeta):
        return []
    archivos = sorted(
        f for f in os.listdir(ruta_carpeta)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(archivos) > MAX_IMAGENES:
        paso = max(1, len(archivos) // MAX_IMAGENES)
        archivos = archivos[::paso][:MAX_IMAGENES]
    return [os.path.join(ruta_carpeta, f) for f in archivos]


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


def procesar_batch_imagenes(sesion_id, ruta_carpeta=None):
    """Procesa la carpeta y escribe la asistencia de la sesión en la base."""
    close_old_connections()
    print(f"[BATCH] Sesión {sesion_id} | carpeta: {ruta_carpeta}")

    try:
        sesion = SesionClase.objects.select_related(
            "asignacion_clase__profesor"
        ).get(id=sesion_id)
    except SesionClase.DoesNotExist:
        print(f"[BATCH] No existe la sesión {sesion_id}")
        return

    asignacion = sesion.asignacion_clase
    status_dir = os.path.join(settings.MEDIA_ROOT, "batch_processed", str(sesion_id))
    os.makedirs(status_dir, exist_ok=True)

    def update_status(current, total, state="processing", mensaje=""):
        with open(os.path.join(status_dir, "status.json"), "w", encoding="utf-8") as sf:
            json.dump({
                "current": current,
                "total": total,
                "percent": int((current / total) * 100) if total > 0 else 0,
                "state": state,
                "mensaje": mensaje,
                "timestamp": time.time(),
            }, sf)

    # Se empieza de cero en cada corrida
    RegistroAsistencia.objects.filter(sesion=sesion).delete()
    sesion.notas_profesor = ""
    sesion.save(update_fields=["notas_profesor"])

    imagenes = _listar_imagenes(ruta_carpeta)
    if not imagenes:
        mensaje = f"No se encontraron imágenes en '{ruta_carpeta or 'data/images/'}'."
        print(f"[BATCH] {mensaje}")
        update_status(0, 0, "error", mensaje)
        return

    total = len(imagenes)
    update_status(0, total, "processing", "Cargando modelos...")

    try:
        from scripts.asistencia_unificada_optimizada import MotorAsistencia

        motor = MotorAsistencia(registrar_csv=False)
        motor.tamano_minimo_rostro = 20
        # Cada imagen es una observación independiente: se confirma con el umbral
        # calibrado y luego se exige evidencia repetida en varias imágenes.
        motor.votos_requeridos = 1
    except Exception as e:
        mensaje = f"No se pudo iniciar el motor de reconocimiento: {e}"
        print(f"[BATCH] {mensaje}")
        update_status(0, total, "error", mensaje)
        return

    estudiantes_por_alias = mapa_estudiantes(asignacion)
    profesor = asignacion.profesor
    alias_profesor = alias_de(profesor.username, profesor.first_name)
    minimo_imagenes = min(3, total)
    acumulado = nuevo_acumulado()
    inicio = timezone.localtime()

    try:
        for indice, ruta in enumerate(imagenes, 1):
            frame = cv2.imread(ruta)
            if frame is None:
                update_status(indice, total, "processing", f"Ilegible: {os.path.basename(ruta)}")
                continue

            reiniciar_pistas(motor)
            decisiones = motor.procesar_frame(frame)
            acumular_decisiones(acumulado, decisiones, frame, indice)
            cv2.imwrite(os.path.join(status_dir, "latest.jpg"), _dibujar(frame, decisiones))
            update_status(indice, total, "processing",
                          f"{os.path.basename(ruta)} | caras: {len(decisiones)}")

        registrados = registrar_lote(
            acumulado, sesion, estudiantes_por_alias,
            alias_profesor=alias_profesor, profesor=profesor,
            minimo_imagenes=minimo_imagenes,
        )
        print(f"[BATCH] Confirmados con >= {minimo_imagenes} imágenes: {sorted(registrados)}")
    except Exception as e:
        mensaje = f"Error durante el procesamiento: {e}"
        print(f"[BATCH] {mensaje}")
        update_status(0, total, "error", mensaje)
        return
    finally:
        close_old_connections()

    update_status(total, total, "completed", "Procesamiento finalizado.")
    print(f"[BATCH] Terminado en {(timezone.localtime() - inicio).total_seconds():.1f} s")

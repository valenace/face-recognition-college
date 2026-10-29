import os
import re
import logging
import tempfile

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from apps.academic.models import Estudiante
from apps.attendance.models import FaceEmbedding

logger = logging.getLogger(__name__)
User = get_user_model()

# ──────────────────────────────────────────────────────────────
# Singleton del motor biométrico para evitar recargar los modelos
# de IA (RetinaFace, ArcFace, AntiSpoofing) en cada request.
# ──────────────────────────────────────────────────────────────
_motor_biometrico = None


def _get_motor():
    """Lazy-load del motor de IA. Solo se instancia una vez."""
    global _motor_biometrico
    if _motor_biometrico is None:
        import sys

        sys.path.insert(0, str(settings.BASE_DIR))
        from scripts.registro_unificado import RegistroBiometrico

        _motor_biometrico = RegistroBiometrico()
        logger.info("Motor biométrico (ArcFace + RetinaFace) cargado correctamente.")
    return _motor_biometrico


class RegistroBiometricoView(APIView):
    """
    POST /api/enrolar/
    Recibe: carnet, nombre_completo, video (.mp4)
    Flujo: Validar → Guardar temp → IA → Crear User + Estudiante + Embedding → Limpiar
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        if not (
            request.user.is_superuser
            or request.user.role in {"DEVELOPER", "COORDINATOR", "DIRECTOR"}
        ):
            return Response(
                {"exito": False, "mensaje": "Sin permiso para registrar estudiantes."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # ── 1. Extraer y validar campos ────────────────────────────
        carnet = request.data.get("carnet", "").strip()
        nombre = request.data.get("nombre_completo", "").strip()
        video = request.FILES.get("video")

        errores = {}
        if not carnet:
            errores["carnet"] = "El carnet es obligatorio."
        elif not re.match(r"^[a-zA-Z0-9_-]+$", carnet):
            errores["carnet"] = "El carnet solo puede contener letras, números, guiones y guiones bajos."

        if not nombre:
            errores["nombre_completo"] = "El nombre completo es obligatorio."

        if not video:
            errores["video"] = "Debes subir un archivo de video."
        elif not video.content_type.startswith("video/"):
            errores["video"] = "El archivo debe ser un video válido (.mp4, .avi, etc.)."

        if errores:
            return Response(
                {"exito": False, "errores": errores},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── 2. Verificar duplicado ─────────────────────────────────
        if User.objects.filter(username=carnet).exists():
            return Response(
                {
                    "exito": False,
                    "mensaje": f"Ya existe un estudiante registrado con el carnet '{carnet}'.",
                },
                status=status.HTTP_409_CONFLICT,
            )

        # ── 3. Guardar video en archivo temporal ───────────────────
        ruta_video_tmp = None
        try:
            tmp_dir = os.path.join(settings.MEDIA_ROOT, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            suffix = os.path.splitext(video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, suffix=suffix, delete=False
            ) as tmp_file:
                for chunk in video.chunks():
                    tmp_file.write(chunk)
                ruta_video_tmp = tmp_file.name

            logger.info(f"Video temporal guardado: {ruta_video_tmp}")

            # ── 4. Procesar con el motor de IA ─────────────────────
            # El alias del motor es el carnet (username), único por estudiante.
            motor = _get_motor()
            resultado = motor.enrolar_usuario_headless(
                id_usuario=carnet,
                ruta_video=ruta_video_tmp,
                sobrescribir=False,
            )

            if not resultado["exito"]:
                return Response(
                    {"exito": False, "mensaje": resultado["mensaje"]},
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY,
                )

            # ── 5. Crear usuario y guardar embedding en Django ─────
            partes_nombre = nombre.split(" ", 1)
            first_name = partes_nombre[0]
            last_name = partes_nombre[1] if len(partes_nombre) > 1 else ""

            with transaction.atomic():
                usuario = User.objects.create_user(                    username=carnet,
                    first_name=first_name,
                    last_name=last_name,
                    role=User.Role.STUDENT,
                    password=None,  # Sin login web, solo biometría
                )

                Estudiante.objects.create(user=usuario, matricula=carnet)

                FaceEmbedding.objects.create(
                    user=usuario,
                    embedding=resultado["embedding"],
                    modelo="arcface",
                    n_muestras=resultado.get("muestras", 0),
                    fuente=FaceEmbedding.Fuente.WEB,
                )

            logger.info(f"Estudiante '{carnet}' registrado correctamente.")

            return Response(
                {
                    "exito": True,
                    "mensaje": resultado["mensaje"],
                    "estudiante": {
                        "carnet": carnet,
                        "nombre": nombre,
                        "muestras_biometricas": resultado["muestras"],
                    },
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            logger.exception("Error inesperado durante el enrolamiento biométrico.")
            return Response(
                {
                    "exito": False,
                    "mensaje": f"Error interno del servidor: {str(e)}",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        finally:
            # ── 6. Limpieza estricta del archivo temporal ──────────
            if ruta_video_tmp and os.path.exists(ruta_video_tmp):
                try:
                    os.remove(ruta_video_tmp)
                    logger.info(f"Video temporal eliminado: {ruta_video_tmp}")
                except OSError as e:
                    logger.warning(f"No se pudo eliminar video temporal: {e}")


# ──────────────────────────────────────────────────────────────
# Asistencia en vivo (cámara o archivo de video)
# ──────────────────────────────────────────────────────────────
ROLES_EN_VIVO = {"PROFESSOR", "COORDINATOR", "DIRECTOR", "DEVELOPER"}


def _autorizado_en_vivo(user):
    return user.is_authenticated and (
        user.is_superuser or user.role in ROLES_EN_VIVO
    )


class EnVivoIniciarView(APIView):
    def post(self, request):
        if not _autorizado_en_vivo(request.user):
            return Response({"exito": False, "mensaje": "Sin permiso."}, status=403)

        from core.servicio_en_vivo import servicio_en_vivo

        fuente = request.data.get("fuente") or "0"
        sesion_id = request.data.get("sesion_id") or None
        exito, mensaje = servicio_en_vivo.iniciar(fuente, sesion_id)
        return Response({"exito": exito, "mensaje": mensaje},
                        status=200 if exito else 400)


class EnVivoDetenerView(APIView):
    def post(self, request):
        if not _autorizado_en_vivo(request.user):
            return Response({"exito": False, "mensaje": "Sin permiso."}, status=403)

        from core.servicio_en_vivo import servicio_en_vivo

        exito, mensaje = servicio_en_vivo.detener()
        return Response({"exito": exito, "mensaje": mensaje})


class EnVivoEstadoView(APIView):
    def get(self, request):
        if not _autorizado_en_vivo(request.user):
            return Response({"exito": False, "mensaje": "Sin permiso."}, status=403)

        from core.servicio_en_vivo import servicio_en_vivo

        return Response({"exito": True, "estado": servicio_en_vivo.estado()})


class EnVivoFrameView(APIView):
    def get(self, request):
        if not _autorizado_en_vivo(request.user):
            return Response({"exito": False, "mensaje": "Sin permiso."}, status=403)

        import cv2
        import numpy as np
        from django.http import HttpResponse
        from core.servicio_en_vivo import servicio_en_vivo

        jpeg = servicio_en_vivo.frame_jpeg()
        if jpeg is None:
            lienzo = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(lienzo, "Sin senal", (230, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            ok, buffer = cv2.imencode(".jpg", lienzo)
            jpeg = buffer.tobytes() if ok else b""

        respuesta = HttpResponse(jpeg, content_type="image/jpeg")
        respuesta["Cache-Control"] = "no-store"
        return respuesta

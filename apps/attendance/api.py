import os
import re
import logging
import tempfile

from django.conf import settings
from django.contrib.auth import get_user_model

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

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
    Flujo: Validar → Guardar temp → IA → Crear User + Embedding → Limpiar
    """

    def post(self, request):
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
            nombre_limpio = re.sub(r"\s+", "_", nombre)
            id_usuario = f"{carnet}_{nombre_limpio}"

            motor = _get_motor()
            resultado = motor.enrolar_usuario_headless(
                id_usuario=id_usuario,
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

            usuario = User.objects.create_user(
                username=carnet,
                first_name=first_name,
                last_name=last_name,
                role=User.Role.STUDENT,
                password=None,  # Sin login web, solo biometría
            )

            FaceEmbedding.objects.create(
                user=usuario,
                embedding=resultado["embedding"],
            )

            logger.info(f"Estudiante '{id_usuario}' registrado correctamente.")

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

from django.urls import path
from apps.attendance.api import (
    RegistroBiometricoView,
    EnVivoIniciarView,
    EnVivoDetenerView,
    EnVivoEstadoView,
    EnVivoFrameView,
    EnVivoStreamView,
)
from apps.attendance.views import (
    live_attendance,
    directorio,
    crear_sesion_demo_view,
    sesiones,
    sesion_detalle,
    descartar_registro,
    eliminar_sesion,
    eliminar_todas_sesiones,
    marcar_presente,
    resolver_revision,
    revocar_biometria,
)

urlpatterns = [
    # ── Páginas HTML ───────────────────────────────────────────
    path("en-vivo/", live_attendance, name="live_attendance"),
    path("en-vivo/sesion-demo/", crear_sesion_demo_view, name="crear_sesion_demo"),
    path("directorio/", directorio, name="directorio"),
    path("directorio/<int:user_id>/revocar/", revocar_biometria, name="revocar_biometria"),
    path("sesiones/", sesiones, name="sesiones"),
    path("sesiones/eliminar-todas/", eliminar_todas_sesiones, name="eliminar_todas_sesiones"),
    path("sesiones/<int:sesion_id>/", sesion_detalle, name="sesion_detalle"),
    path("sesiones/<int:sesion_id>/eliminar/", eliminar_sesion, name="eliminar_sesion"),
    path(
        "sesiones/<int:sesion_id>/presente/<int:estudiante_id>/",
        marcar_presente,
        name="marcar_presente",
    ),
    path(
        "sesiones/<int:sesion_id>/revision/<int:registro_id>/<str:accion>/",
        resolver_revision,
        name="resolver_revision",
    ),
    path(
        "sesiones/<int:sesion_id>/registros/<int:registro_id>/descartar/",
        descartar_registro,
        name="descartar_registro",
    ),

    # ── API Endpoints ──────────────────────────────────────────
    path("api/enrolar/", RegistroBiometricoView.as_view(), name="api_enrolar"),

    # Asistencia en vivo
    path("api/en-vivo/iniciar/", EnVivoIniciarView.as_view(), name="api_en_vivo_iniciar"),
    path("api/en-vivo/detener/", EnVivoDetenerView.as_view(), name="api_en_vivo_detener"),
    path("api/en-vivo/estado/", EnVivoEstadoView.as_view(), name="api_en_vivo_estado"),
    path("api/en-vivo/frame/", EnVivoFrameView.as_view(), name="api_en_vivo_frame"),
    path("api/en-vivo/stream/", EnVivoStreamView.as_view(), name="api_en_vivo_stream"),
]

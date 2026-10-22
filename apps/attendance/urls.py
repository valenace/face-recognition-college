from django.urls import path
from apps.attendance.views import live_attendance, directorio
from apps.attendance.api import (
    RegistroBiometricoView,
    EnVivoIniciarView,
    EnVivoDetenerView,
    EnVivoEstadoView,
    EnVivoFrameView,
)

urlpatterns = [
    # ── Páginas HTML ───────────────────────────────────────────
    path("en-vivo/", live_attendance, name="live_attendance"),
    path("directorio/", directorio, name="directorio"),

    # ── API Endpoints ──────────────────────────────────────────
    path("api/enrolar/", RegistroBiometricoView.as_view(), name="api_enrolar"),

    # Asistencia en vivo
    path("api/en-vivo/iniciar/", EnVivoIniciarView.as_view(), name="api_en_vivo_iniciar"),
    path("api/en-vivo/detener/", EnVivoDetenerView.as_view(), name="api_en_vivo_detener"),
    path("api/en-vivo/estado/", EnVivoEstadoView.as_view(), name="api_en_vivo_estado"),
    path("api/en-vivo/frame/", EnVivoFrameView.as_view(), name="api_en_vivo_frame"),
]

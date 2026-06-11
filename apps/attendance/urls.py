from django.urls import path
from . import views
from apps.attendance.views import live_attendance, directorio
from apps.attendance.api import RegistroBiometricoView

urlpatterns = [
    # ── Páginas HTML ───────────────────────────────────────────
    path("en-vivo/", live_attendance, name="live_attendance"),
    path("clase-en-vivo/<int:session_id>/", views.ClaseEnVivoView.as_view(), name="clase-en-vivo"),
    path("directorio/", directorio, name="directorio"),

    # ── API Endpoints ──────────────────────────────────────────
    path("api/enrolar/", RegistroBiometricoView.as_view(), name="api_enrolar"),
    path("api/resolver-alerta/", views.ResolverAlertaAPI.as_view(), name="api-resolver-alerta"),

    # Módulo de Dirección (Seguridad)
    path('direccion/auditoria/', views.AuditoriaSeguridadView.as_view(), name='auditoria-seguridad'),
    path('coordinacion/reportes/', views.ReportesAsistenciaView.as_view(), name='reportes-asistencia'),
]

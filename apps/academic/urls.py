from django.urls import path
from .views import (
    PanelProfesorView, DetalleAsignacionView, IniciarSesionView,
    MetricasAlumnoView, HistorialClasesView,
    PanelEnrolamientoView, VisorAcademicoView, EnrolarRostroAPI,
    MiHorarioView, MateriasProfesorView
)
from apps.attendance.views import (
    ClaseEnVivoView, ResolverAlertaAPI,
    ReportesAsistenciaView, AuditoriaSeguridadView
)

app_name = "academic"

urlpatterns = [
    # ── PROFESOR ──────────────────────────────────────────────
    path("profesor/panel/", PanelProfesorView.as_view(), name="panel-profesor"),
    path("profesor/horario/", MiHorarioView.as_view(), name="mi-horario"),
    path("profesor/materias/", MateriasProfesorView.as_view(), name="materias-profesor"),
    path("profesor/asignacion/<int:pk>/", DetalleAsignacionView.as_view(), name="detalle-asignacion"),
    path("profesor/iniciar-sesion/<int:asignacion_id>/", IniciarSesionView.as_view(), name="iniciar-sesion"),
    path("profesor/monitor-clase/<int:session_id>/", ClaseEnVivoView.as_view(), name="monitor-clase"),
    path("profesor/metricas-alumno/<int:student_id>/<int:class_id>/", MetricasAlumnoView.as_view(), name="metricas-alumno"),
    path("profesor/historial/", HistorialClasesView.as_view(), name="historial-clases"),
    path("profesor/api/resolver-alerta/", ResolverAlertaAPI.as_view(), name="api-resolver-alerta"),

    # ── COORDINADOR ───────────────────────────────────────────
    path("coordinador/enrolamiento/", PanelEnrolamientoView.as_view(), name="panel-enrolamiento"),
    path("coordinador/visor-academico/", VisorAcademicoView.as_view(), name="visor-academico"),
    path("coordinador/api/enrolar-rostro/", EnrolarRostroAPI.as_view(), name="api-enrolar-rostro"),
    path("coordinador/reportes/", ReportesAsistenciaView.as_view(), name="reportes-asistencia"),

    # ── DIRECTOR ──────────────────────────────────────────────
    path("director/auditoria/", AuditoriaSeguridadView.as_view(), name="auditoria-seguridad"),
]

from django.urls import path
from .views import DashboardProfesorView, MonitorClaseView, HistorialClasesView, MetricasAlumnoView
from . import views

app_name = "academic"

urlpatterns = [
    # dashboard principal del docente
    path("dashboard-profesor/", DashboardProfesorView.as_view(), name="dashboard-profesor"),
    # monitor de sesion de clase por id
    path("monitor-clase/<int:session_id>/", MonitorClaseView.as_view(), name="monitor-clase"),
    # historial de clases finalizadas
    path("historial-clases/", HistorialClasesView.as_view(), name="historial-clases"),
    # métricas individuales de alumno por asignación
    path("metricas-alumno/<int:student_id>/<int:class_id>/", MetricasAlumnoView.as_view(), name="metricas-alumno"),
    # Módulo de Coordinación
    path('coordinacion/enrolamiento/', views.PanelEnrolamientoView.as_view(), name='panel-enrolamiento'),
    path('api/enrolar-rostro/', views.EnrolarRostroAPI.as_view(), name='api-enrolar-rostro'),
    path('coordinacion/visor-academico/', views.VisorAcademicoView.as_view(), name='visor-academico'),
]

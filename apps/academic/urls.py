from django.urls import path
from .views import DashboardProfesorView, MonitorClaseView

app_name = "academic"

urlpatterns = [
    # dashboard principal del docente
    path("dashboard-profesor/", DashboardProfesorView.as_view(), name="dashboard-profesor"),
    # monitor de sesion de clase por id
    path("monitor-clase/<int:session_id>/", MonitorClaseView.as_view(), name="monitor-clase"),
]

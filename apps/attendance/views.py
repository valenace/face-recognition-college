from django.shortcuts import render
from django.contrib.auth import get_user_model
from django.views.generic import ListView
from apps.users.permissions import RoleRequiredMixin
from apps.attendance.models import RegistroAsistencia


User = get_user_model()


def registration(request):
    return render(request, "attendance/registration.html")


def live_attendance(request):
    return render(request, "attendance/live.html")


def directorio(request):
    """Vista del directorio de estudiantes con enrolamiento biométrico."""
    estudiantes = (
        User.objects.filter(role=User.Role.STUDENT)
        .select_related("face_data")
        .order_by("-date_joined")
    )
    return render(request, "attendance/directorio.html", {
        "estudiantes": estudiantes,
    })

class AuditoriaSeguridadView(RoleRequiredMixin, ListView):
    allowed_roles = ['COORDINATOR', 'DIRECTOR'] 
    template_name = 'coordinacion/auditoria_seguridad.html'
    context_object_name = 'alertas_seguridad'

    def get_queryset(self):
        # Filtramos solo los registros marcados como fraude
        return RegistroAsistencia.objects.filter(
            es_fraude=True
        ).select_related(
            'sesion__asignacion_clase__asignatura', 
            'estudiante__user'
        ).order_by('-sesion__fecha', '-hora_entrada')
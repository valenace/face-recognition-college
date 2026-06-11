from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import get_user_model
from django.views import View
from django.views.generic import ListView
from django.db.models import Q
from django.core.exceptions import PermissionDenied
from apps.users.permissions import RoleRequiredMixin
from apps.attendance.models import RegistroAsistencia, SesionClase


User = get_user_model()





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


class ReportesAsistenciaView(RoleRequiredMixin, ListView):
    model = RegistroAsistencia
    allowed_roles = ['COORDINATOR', 'DIRECTOR']
    template_name = 'coordinacion/reportes_asistencia.html'
    context_object_name = 'reportes'

    def get_queryset(self):
        queryset = RegistroAsistencia.objects.select_related(
            'sesion__asignacion_clase__asignatura', 
            'estudiante__user'
        ).all().order_by('-sesion__fecha', '-hora_entrada')
        
        q = self.request.GET.get('q')
        fecha = self.request.GET.get('fecha')
        
        if q:
            queryset = queryset.filter(
                Q(estudiante__user__first_name__icontains=q) |
                Q(estudiante__user__last_name__icontains=q) |
                Q(estudiante__matricula__icontains=q) |
                Q(sesion__asignacion_clase__asignatura__nombre__icontains=q)
            )
        if fecha:
            queryset = queryset.filter(sesion__fecha=fecha)
            
        return queryset


class ClaseEnVivoView(RoleRequiredMixin, View):
    allowed_roles = [User.Role.PROFESSOR]

    def get(self, request, session_id):
        # Redirect to the existing MonitorClaseView
        return redirect('academic:monitor-clase', session_id=session_id)

    def post(self, request, session_id):
        sesion = get_object_or_404(SesionClase, pk=session_id)
        # Check permissions: only the assigned professor (or a superuser) can finalize the session
        if not request.user.is_superuser and sesion.asignacion_clase.profesor != request.user:
            raise PermissionDenied("no tienes permiso para modificar esta sesión de clase.")
        
        notas = request.POST.get('notas_profesor', '')
        sesion.notas_profesor = notas
        sesion.estado = SesionClase.Estado.FINALIZADA
        sesion.save()

        # Iterate over attendance records and copy last seen time to exit time
        for ast in RegistroAsistencia.objects.filter(sesion=sesion):
            if ast.ultima_vez_visto:
                ast.hora_salida = ast.ultima_vez_visto
                ast.save()
        
        return redirect('academic:dashboard-profesor')
from django.views.generic import ListView, DetailView
from django.core.exceptions import PermissionDenied
from apps.users.permissions import RoleRequiredMixin
from apps.users.models import User
from .models import AsignacionClase, Inscripcion
from apps.attendance.models import SesionClase, RegistroAsistencia

class DashboardProfesorView(RoleRequiredMixin, ListView):
    """
    vista tipo dashboard para que el profesor vea todas sus clases asignadas.
    """
    model = AsignacionClase
    template_name = "academic/dashboard_profesor.html"
    context_object_name = "clases"
    allowed_roles = [User.Role.PROFESSOR]

    def get_queryset(self):
        # retornar únicamente las asignaciones vinculadas al profesor actual
        return AsignacionClase.objects.filter(profesor=self.request.user)


class MonitorClaseView(RoleRequiredMixin, DetailView):
    """
    vista para monitorear una sesión de clase en tiempo real.
    """
    model = SesionClase
    template_name = "academic/monitor_clase.html"
    context_object_name = "sesion"
    allowed_roles = [User.Role.PROFESSOR]
    pk_url_kwarg = "session_id"

    def get_object(self, queryset=None):
        sesion = super().get_object(queryset)
        # validar regla de negocio: que la sesión pertenezca al profesor actual
        if sesion.asignacion_clase.profesor != self.request.user:
            raise PermissionDenied("no tienes permiso para visualizar esta sesión de clase.")
        return sesion

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        sesion = self.object
        
        # obtener asistencias indexadas por el id del estudiante
        asistencias = RegistroAsistencia.objects.filter(sesion=sesion).select_related("estudiante__user")
        asistencias_map = {ast.estudiante_id: ast for ast in asistencias}
        
        # obtener estudiantes inscritos
        inscripciones = Inscripcion.objects.filter(
            asignacion_clase=sesion.asignacion_clase
        ).select_related("estudiante__user")
        
        # pre-procesar estudiantes con su asistencia correspondiente para el template
        estudiantes_monitoreo = []
        for insc in inscripciones:
            estudiantes_monitoreo.append({
                "estudiante": insc.estudiante,
                "asistencia": asistencias_map.get(insc.estudiante_id)
            })
            
        context["estudiantes_monitoreo"] = estudiantes_monitoreo
        return context

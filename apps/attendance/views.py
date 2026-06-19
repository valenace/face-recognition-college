from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import get_user_model
from django.views import View
from django.views.generic import ListView, DetailView
from django.db.models import Q
from django.core.exceptions import PermissionDenied
from django.http import JsonResponse
import json
from apps.users.permissions import RoleRequiredMixin
from apps.attendance.models import RegistroAsistencia, SesionClase
from apps.academic.models import Inscripcion


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


class ClaseEnVivoView(RoleRequiredMixin, DetailView):
    model = SesionClase
    template_name = "academic/monitor_clase.html"
    context_object_name = "sesion"
    allowed_roles = [User.Role.PROFESSOR]
    pk_url_kwarg = "session_id"

    def get_object(self, queryset=None):
        sesion = super().get_object(queryset)
        if not self.request.user.is_superuser and sesion.asignacion_clase.profesor != self.request.user:
            raise PermissionDenied("no tienes permiso para visualizar esta sesión de clase.")
        return sesion

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        sesion = self.object
        
        # obtener asistencias
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

        # estudiantes_ausentes: filtra los estudiantes inscritos en la sección que NO tienen un RegistroAsistencia para la sesión actual
        asistencia_student_ids = asistencias.values_list("estudiante_id", flat=True)
        context["estudiantes_ausentes"] = [
            insc.estudiante for insc in inscripciones if insc.estudiante_id not in asistencia_student_ids
        ]
        context["registros_asistencia"] = asistencias
        
        return context

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
        
        return redirect('academic:panel-profesor')


class ResolverAlertaAPI(RoleRequiredMixin, View):
    allowed_roles = [User.Role.PROFESSOR]

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            registro_id = data.get("registro_id")
            accion = data.get("accion")
        except json.JSONDecodeError:
            return JsonResponse({"status": "error", "message": "JSON inválido."}, status=400)

        if not registro_id or not accion:
            return JsonResponse({"status": "error", "message": "Parámetros incompletos."}, status=400)

        registro = get_object_or_404(RegistroAsistencia, pk=registro_id)

        # Check permissions: only the assigned professor of the session (or superuser) can resolve
        if not request.user.is_superuser and registro.sesion.asignacion_clase.profesor != request.user:
            return JsonResponse({"status": "error", "message": "no tienes permiso para modificar este registro."}, status=403)

        if accion == "falsa_alarma":
            registro.es_fraude = False
            registro.alerta_revisada = True
            registro.notas_auditoria = "Marcado como falsa alarma por el profesor"
            registro.save()
        elif accion == "confirmar_fraude":
            registro.es_fraude = True
            registro.alerta_revisada = True
            registro.notas_auditoria = "Fraude confirmado por el profesor"
            registro.save()
        else:
            return JsonResponse({"status": "error", "message": "Acción no válida."}, status=400)

        return JsonResponse({"status": "success"})
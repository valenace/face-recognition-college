from django.views.generic import ListView, DetailView, TemplateView
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404
from django.utils import timezone
from apps.users.permissions import RoleRequiredMixin
from apps.users.models import User
from .models import AsignacionClase, Inscripcion, Estudiante
from apps.attendance.models import SesionClase, RegistroAsistencia, FaceEmbedding
from django.views import View
from django.db.models import Count
from django.http import JsonResponse
import json
from apps.academic.models import Estudiante, AsignacionClase, Salon

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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        clases = self.get_queryset()
        
        # Total de clases asignadas
        context["total_clases"] = clases.count()
        
        # Total de estudiantes (únicos) registrados en las clases del profesor
        context["total_estudiantes"] = Inscripcion.objects.filter(
            asignacion_clase__in=clases
        ).values('estudiante').distinct().count()
        
        # Total de alertas de fraude en las sesiones del profesor
        context["total_alertas"] = RegistroAsistencia.objects.filter(
            sesion__asignacion_clase__in=clases,
            es_fraude=True
        ).count()

        # Próxima clase
        ahora = timezone.localtime().time()
        proxima = clases.filter(horario_inicio__gt=ahora).order_by('horario_inicio').first()
        if not proxima:
            proxima = clases.order_by('horario_inicio').first()
        context["proxima_clase"] = proxima
        
        return context


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


class HistorialClasesView(RoleRequiredMixin, ListView):
    """
    Vista protegida que lista las sesiones de clase pasadas (FINALIZADA)
    del profesor actual.
    """
    model = SesionClase
    template_name = "academic/historial_clases.html"
    context_object_name = "sesiones"
    allowed_roles = [User.Role.PROFESSOR]

    def get_queryset(self):
        return SesionClase.objects.filter(
            asignacion_clase__profesor=self.request.user,
            estado=SesionClase.Estado.FINALIZADA
        ).select_related(
            "asignacion_clase__asignatura",
            "asignacion_clase__seccion",
            "asignacion_clase__salon"
        ).order_by("-fecha", "-id")


class MetricasAlumnoView(RoleRequiredMixin, TemplateView):
    """
    Vista protegida que calcula y muestra las métricas de asistencia,
    faltas, salidas tempranas e intentos de fraude de un estudiante en una clase.
    """
    template_name = "academic/metricas_alumno.html"
    allowed_roles = [User.Role.PROFESSOR]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        student_id = self.kwargs.get("student_id")
        class_id = self.kwargs.get("class_id")

        estudiante = get_object_or_404(Estudiante, pk=student_id)
        asignacion = get_object_or_404(AsignacionClase, pk=class_id)

        # Validar que la clase pertenezca al profesor autenticado
        if asignacion.profesor != self.request.user:
            raise PermissionDenied("No tienes permiso para ver métricas de esta clase.")

        # 1. Total de clases dictadas hasta la fecha (sesiones de esta asignacion finalizadas)
        sesiones_finalizadas = SesionClase.objects.filter(
            asignacion_clase=asignacion,
            estado=SesionClase.Estado.FINALIZADA
        )
        total_dictadas = sesiones_finalizadas.count()

        # 2. Total de faltas (clases dictadas donde el alumno NO tiene RegistroAsistencia)
        sesiones_con_asistencia = RegistroAsistencia.objects.filter(
            estudiante=estudiante,
            sesion__in=sesiones_finalizadas
        ).values_list("sesion_id", flat=True)
        total_faltas = sesiones_finalizadas.exclude(id__in=sesiones_con_asistencia).count()

        # 3. Salidas tempranas sin justificación (donde hora_salida < horario_fin)
        salidas_tempranas = RegistroAsistencia.objects.filter(
            estudiante=estudiante,
            sesion__asignacion_clase=asignacion,
            hora_salida__lt=asignacion.horario_fin
        ).count()

        # 4. Intentos de fraude históricos (en toda la trayectoria del alumno en el sistema)
        fraudes_historicos = RegistroAsistencia.objects.filter(
            estudiante=estudiante,
            es_fraude=True
        ).count()

        context.update({
            "estudiante": estudiante,
            "asignacion": asignacion,
            "total_dictadas": total_dictadas,
            "total_faltas": total_faltas,
            "salidas_tempranas": salidas_tempranas,
            "fraudes_historicos": fraudes_historicos,
        })
        return context

class PanelEnrolamientoView(RoleRequiredMixin, ListView):
    allowed_roles = ['COORDINATOR', 'DIRECTOR']
    template_name = 'coordinacion/panel_enrolamiento.html'
    context_object_name = 'estudiantes'

    def get_queryset(self):
        # Cuenta los rostros registrados a través de la relación OneToOne con User
        return Estudiante.objects.select_related('user').annotate(
            tiene_rostro=Count('user__face_data')
        ).order_by('tiene_rostro', 'user__first_name')

class EnrolarRostroAPI(RoleRequiredMixin, View):
    allowed_roles = ['COORDINATOR', 'DIRECTOR']

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            user_id = data.get('user_id')
            imagen_base64 = data.get('imagen')

            # Dummy de ejemplo mientras conectamos tu script IA:
            vector_simulado = [0.12, 0.45, 0.78]

            FaceEmbedding.objects.update_or_create(
                user_id=user_id,
                defaults={'embedding': vector_simulado}
            )
            return JsonResponse({'status': 'success', 'message': 'Biometría registrada exitosamente.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

class VisorAcademicoView(RoleRequiredMixin, TemplateView):
    allowed_roles = ['COORDINATOR', 'DIRECTOR']
    template_name = 'coordinacion/visor_academico.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['total_salones'] = Salon.objects.count()
        context['total_estudiantes'] = Estudiante.objects.count()
        context['asignaciones'] = AsignacionClase.objects.select_related(
            'asignatura', 'profesor', 'salon', 'seccion'
        ).all().order_by('asignatura__nombre')
        return context

class ReportesAsistenciaView(RoleRequiredMixin, TemplateView):
    allowed_roles = ['COORDINATOR', 'DIRECTOR']
    template_name = 'coordinacion/reportes_asistencia.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Filtros de búsqueda
        fecha_inicio = self.request.GET.get('fecha_inicio')
        fecha_fin = self.request.GET.get('fecha_fin')
        asignatura_id = self.request.GET.get('asignatura')
        matricula = self.request.GET.get('matricula')
        
        queryset = RegistroAsistencia.objects.select_related(
            'estudiante__user',
            'sesion__asignacion_clase__asignatura'
        ).all()
        
        if fecha_inicio:
            queryset = queryset.filter(sesion__fecha__gte=fecha_inicio)
        if fecha_fin:
            queryset = queryset.filter(sesion__fecha__lte=fecha_fin)
        if asignatura_id:
            queryset = queryset.filter(sesion__asignacion_clase__asignatura_id=asignatura_id)
        if matricula:
            queryset = queryset.filter(estudiante__matricula__icontains=matricula)
            
        context['reportes'] = queryset.order_by('-sesion__fecha', '-hora_entrada')[:50]
        context['asignaturas'] = Asignatura.objects.all()
        return context



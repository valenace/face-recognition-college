from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.views import View
from django.views.generic import ListView, DetailView
from django.db.models import Count, Q
from django.core.exceptions import PermissionDenied
from django.http import JsonResponse
from django.utils import timezone
import json
from apps.users.permissions import RoleRequiredMixin, role_required
from apps.attendance.models import RegistroAsistencia, SesionClase
from apps.academic.models import Estudiante, Inscripcion

# Similitud por debajo de la cual una asistencia necesita revisión humana.
UMBRAL_REVISION = 0.40


User = get_user_model()





@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def live_attendance(request):
    sesiones = (
        SesionClase.objects.filter(estado=SesionClase.Estado.EN_CURSO)
        .select_related(
            "asignacion_clase__asignatura",
            "asignacion_clase__seccion",
            "asignacion_clase__profesor",
        )
        .order_by("-fecha", "-id")
    )
    return render(request, "attendance/live.html", {"sesiones": sesiones})


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def crear_sesion_demo_view(request):
    """Crea una nueva sesión desde el control en vivo o desde el listado de sesiones."""
    if request.method != "POST":
        return redirect("live_attendance")

    from core.sesion_demo import crear_sesion_demo

    crear_sesion_demo()
    siguiente = request.POST.get("next") or "live_attendance"
    if siguiente not in {"live_attendance", "sesiones"}:
        siguiente = "live_attendance"
    return redirect(siguiente)


def _borrar_sesion_y_archivos(sesion):
    """Borra las capturas en disco de la sesión y elimina la sesión (y andamiaje demo si aplica)."""
    from core.sesion_demo import limpiar_sesion_demo

    for reg in sesion.asistencias.all():
        if reg.captura_fraude:
            reg.captura_fraude.delete(save=False)

    if sesion.asignacion_clase.asignatura.codigo == "DEMO-FLUJO":
        limpiar_sesion_demo(sesion)
    else:
        sesion.delete()


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def eliminar_sesion(request, sesion_id):
    """Elimina una sesión específica y sus registros."""
    if request.method != "POST":
        return redirect("sesiones")

    sesion = get_object_or_404(
        SesionClase.objects.select_related("asignacion_clase__asignatura"),
        pk=sesion_id,
    )
    _borrar_sesion_y_archivos(sesion)
    return redirect("sesiones")


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def eliminar_todas_sesiones(request):
    """Elimina todas las sesiones registradas para empezar desde cero."""
    if request.method != "POST":
        return redirect("sesiones")

    for sesion in SesionClase.objects.select_related("asignacion_clase__asignatura").all():
        _borrar_sesion_y_archivos(sesion)
    return redirect("sesiones")


@login_required
def directorio(request):
    """Vista del directorio de estudiantes con enrolamiento biométrico."""
    estudiantes = list(
        User.objects.filter(role=User.Role.STUDENT)
        .select_related("face_data")
        .order_by("-date_joined")
    )
    con_biometria = sum(1 for estudiante in estudiantes if hasattr(estudiante, "face_data"))
    return render(request, "attendance/directorio.html", {
        "estudiantes": estudiantes,
        "con_biometria": con_biometria,
        "sin_biometria": len(estudiantes) - con_biometria,
    })


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def sesiones(request):
    """Listado de sesiones con el resumen de su asistencia."""
    sesiones_qs = (
        SesionClase.objects.select_related(
            "asignacion_clase__asignatura",
            "asignacion_clase__seccion",
            "asignacion_clase__profesor",
        )
        .annotate(
            confirmados=Count(
                "asistencias",
                filter=Q(
                    asistencias__tipo_evento=RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL
                ),
                distinct=True,
            ),
            intrusos=Count(
                "asistencias",
                filter=Q(asistencias__tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO),
                distinct=True,
            ),
            spoofing=Count(
                "asistencias",
                filter=Q(asistencias__tipo_evento=RegistroAsistencia.TipoEvento.SPOOFING),
                distinct=True,
            ),
        )
        .order_by("-fecha", "-id")
    )
    return render(request, "attendance/sesiones.html", {"sesiones": sesiones_qs})


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def sesion_detalle(request, sesion_id):
    """Detalle de una sesión: resumen y filas de asistencia."""
    sesion = get_object_or_404(
        SesionClase.objects.select_related(
            "asignacion_clase__asignatura",
            "asignacion_clase__seccion",
            "asignacion_clase__profesor",
        ),
        pk=sesion_id,
    )
    registros = sesion.asistencias.select_related("estudiante__user").order_by(
        "tipo_evento", "hora_entrada", "id"
    )

    resumen = {"confirmados": 0, "intrusos": 0, "spoofing": 0, "otra_seccion": 0,
               "rechazados": 0, "revision": 0, "ausentes": 0}
    for registro in registros:
        if registro.tipo_evento == RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL:
            resumen["confirmados"] += 1
        elif registro.tipo_evento == RegistroAsistencia.TipoEvento.INTRUSO:
            resumen["intrusos"] += 1
        elif registro.tipo_evento == RegistroAsistencia.TipoEvento.SPOOFING:
            resumen["spoofing"] += 1
        elif registro.tipo_evento == RegistroAsistencia.TipoEvento.RECHAZADO:
            resumen["rechazados"] += 1
        else:
            resumen["otra_seccion"] += 1

    # Nómina de la clase: quién está, quién falta y quién está en la banda de
    # revisión (similitud 0.30–0.40), que el docente confirma o rechaza.
    inscritos = list(
        Inscripcion.objects.filter(asignacion_clase=sesion.asignacion_clase)
        .select_related("estudiante__user")
        .order_by("estudiante__user__last_name", "estudiante__user__first_name")
    )
    normales = {
        registro.estudiante_id: registro
        for registro in registros
        if registro.tipo_evento == RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL
        and registro.estudiante_id
    }
    revision = [
        registro
        for registro in normales.values()
        if registro.similitud_ia is not None
        and registro.similitud_ia < UMBRAL_REVISION
        and not registro.alerta_revisada
    ]
    revision.sort(key=lambda r: r.similitud_ia)
    ausentes = [
        inscripcion.estudiante
        for inscripcion in inscritos
        if inscripcion.estudiante_id not in normales
    ]
    resumen["revision"] = len(revision)
    resumen["ausentes"] = len(ausentes)

    return render(
        request,
        "attendance/sesion_detalle.html",
        {
            "sesion": sesion,
            "registros": registros,
            "resumen": resumen,
            "presentes": sorted(
                normales.values(),
                key=lambda r: (r.estudiante.user.last_name, r.estudiante.user.first_name)
                if r.estudiante else "",
            ),
            "revision": revision,
            "ausentes": ausentes,
        },
    )


def _puede_gestionar(user, sesion):
    """El profesor dueño de la sesión o un directivo puede gestionarla."""
    if user.is_superuser or user.role in (
        User.Role.DEVELOPER, User.Role.DIRECTOR, User.Role.COORDINATOR
    ):
        return True
    return (
        user.role == User.Role.PROFESSOR
        and sesion.asignacion_clase.profesor_id == user.id
    )


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def marcar_presente(request, sesion_id, estudiante_id):
    """El docente marca manualmente la asistencia de un estudiante."""
    if request.method != "POST":
        return redirect("sesion_detalle", sesion_id=sesion_id)

    sesion = get_object_or_404(
        SesionClase.objects.select_related("asignacion_clase"), pk=sesion_id
    )
    if not _puede_gestionar(request.user, sesion):
        return redirect("sesion_detalle", sesion_id=sesion_id)

    estudiante = get_object_or_404(Estudiante, pk=estudiante_id)
    ahora = timezone.localtime().time()
    RegistroAsistencia.objects.update_or_create(
        sesion=sesion,
        estudiante=estudiante,
        defaults={
            "hora_entrada": ahora,
            "ultima_vez_visto": ahora,
            "tipo_evento": RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL,
            "alerta_revisada": True,
            "notas_auditoria": "Marcado manualmente por el profesor.",
        },
    )
    return redirect("sesion_detalle", sesion_id=sesion_id)


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def resolver_revision(request, sesion_id, registro_id, accion):
    """Acepta o rechaza una asistencia de baja confianza (banda 0.30–0.40)."""
    if request.method != "POST":
        return redirect("sesion_detalle", sesion_id=sesion_id)

    sesion = get_object_or_404(
        SesionClase.objects.select_related("asignacion_clase"), pk=sesion_id
    )
    if not _puede_gestionar(request.user, sesion):
        return redirect("sesion_detalle", sesion_id=sesion_id)

    registro = get_object_or_404(RegistroAsistencia, pk=registro_id, sesion=sesion)
    if accion == "aceptar":
        registro.alerta_revisada = True
        registro.notas_auditoria = "Similitud limite confirmada por el profesor."
        registro.save(update_fields=["alerta_revisada", "notas_auditoria"])
    elif accion == "rechazar":
        registro.tipo_evento = RegistroAsistencia.TipoEvento.RECHAZADO
        registro.alerta_revisada = True
        registro.notas_auditoria = "Rechazada por el profesor: falso positivo de identidad."
        registro.save(update_fields=["tipo_evento", "alerta_revisada", "notas_auditoria"])
    return redirect("sesion_detalle", sesion_id=sesion_id)


@role_required("PROFESSOR", "COORDINATOR", "DIRECTOR")
def descartar_registro(request, sesion_id, registro_id):
    """Marca una alerta como revisada (el profesor ya identificó a la persona)."""
    if request.method != "POST":
        return redirect("sesion_detalle", sesion_id=sesion_id)

    RegistroAsistencia.objects.filter(
        pk=registro_id, sesion_id=sesion_id
    ).update(alerta_revisada=True)
    return redirect("sesion_detalle", sesion_id=sesion_id)


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
        
        # Partition assistance records to separate registered students from unregistered/unknown faces
        context["registros_asistencia"] = asistencias
        context["presentes_seccion"] = [ast for ast in asistencias if ast.estudiante is not None]
        context["desconocidos_detectados"] = [ast for ast in asistencias if ast.estudiante is None]
        
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


import threading
from core.procesador_batch import procesar_batch_imagenes

class ProcesarDatasetView(RoleRequiredMixin, View):
    allowed_roles = [User.Role.PROFESSOR, User.Role.COORDINATOR, User.Role.DIRECTOR]

    def post(self, request, session_id, *args, **kwargs):
        sesion = get_object_or_404(SesionClase, pk=session_id)
        
        # Check permissions: only the assigned professor, coordinators, or directors can run this
        if not request.user.is_superuser and request.user.role not in ['COORDINATOR', 'DIRECTOR'] and sesion.asignacion_clase.profesor != request.user:
            return JsonResponse({"status": "error", "message": "No tienes permiso para ejecutar el procesamiento en esta sesión de clase."}, status=403)

        try:
            data = json.loads(request.body) if request.body else {}
        except json.JSONDecodeError:
            data = {}

        ruta_carpeta = data.get('ruta_carpeta', 'data/images/')

        # Synchronously clear previous database records and notes for a clean 0-state
        RegistroAsistencia.objects.filter(sesion=sesion).delete()
        sesion.notas_profesor = ""
        sesion.save()

        # Synchronously delete the media/batch_processed/<session_id> folder
        import os
        import shutil
        from django.conf import settings
        status_dir = os.path.join(settings.MEDIA_ROOT, 'batch_processed', str(session_id))
        if os.path.exists(status_dir):
            try:
                shutil.rmtree(status_dir)
            except Exception:
                pass

        # Start the batch processing task in a background thread
        thread = threading.Thread(
            target=procesar_batch_imagenes,
            args=(session_id, ruta_carpeta)
        )
        thread.start()

        return JsonResponse({
            "status": "success",
            "message": "Procesamiento batch de imágenes iniciado en segundo plano."
        })


class LimpiarSesionView(RoleRequiredMixin, View):
    allowed_roles = [User.Role.PROFESSOR, User.Role.COORDINATOR, User.Role.DIRECTOR]

    def post(self, request, session_id, *args, **kwargs):
        sesion = get_object_or_404(SesionClase, pk=session_id)
        
        # Check permissions
        if not request.user.is_superuser and request.user.role not in ['COORDINATOR', 'DIRECTOR'] and sesion.asignacion_clase.profesor != request.user:
            return JsonResponse({"status": "error", "message": "No tienes permiso para vaciar la asistencia en esta sesión de clase."}, status=403)

        # Clear database records
        RegistroAsistencia.objects.filter(sesion=sesion).delete()
        sesion.notas_profesor = ""
        sesion.save()

        # Delete batch_processed status files on disk
        import os
        import shutil
        from django.conf import settings
        status_dir = os.path.join(settings.MEDIA_ROOT, 'batch_processed', str(session_id))
        if os.path.exists(status_dir):
            try:
                shutil.rmtree(status_dir)
            except Exception:
                pass

        return JsonResponse({
            "status": "success",
            "message": "Registros de asistencia y archivos de procesamiento limpiados exitosamente."
        })
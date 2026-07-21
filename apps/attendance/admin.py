from django.contrib import admin
from .models import FaceEmbedding, SesionClase, RegistroAsistencia

@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "updated_at")
    search_fields = ("user__username", "user__email")
    ordering = ("-created_at",)


@admin.register(SesionClase)
class SesionClaseAdmin(admin.ModelAdmin):
    list_display = ("asignacion_clase", "fecha", "estado")
    list_filter = ("estado", "fecha")
    search_fields = ("asignacion_clase__asignatura__nombre", "asignacion_clase__seccion__codigo")
    ordering = ("-fecha",)


@admin.register(RegistroAsistencia)
class RegistroAsistenciaAdmin(admin.ModelAdmin):
    list_display = ("estudiante", "sesion", "hora_entrada", "hora_salida", "similitud_ia", "es_fraude")
    # filtro facil para identificar fraudes de liveness
    list_filter = ("es_fraude", "sesion__fecha", "sesion__asignacion_clase__asignatura")
    search_fields = (
        "estudiante__user__first_name",
        "estudiante__user__last_name",
        "estudiante__matricula",
        "sesion__asignacion_clase__asignatura__nombre"
    )
    ordering = ("-sesion__fecha", "hora_entrada")

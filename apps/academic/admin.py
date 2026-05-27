from django.contrib import admin
from .models import Asignatura, Salon, Seccion, Estudiante, AsignacionClase, Inscripcion

# inline para inscripciones en la vista de asignacionclase
class InscripcionInline(admin.TabularInline):
    model = Inscripcion
    extra = 1
    verbose_name = "Estudiante Inscrito"
    verbose_name_plural = "Estudiantes Inscritos"


@admin.register(Asignatura)
class AsignaturaAdmin(admin.ModelAdmin):
    list_display = ("nombre", "codigo", "descripcion")
    search_fields = ("nombre", "codigo")
    ordering = ("nombre",)


@admin.register(Salon)
class SalonAdmin(admin.ModelAdmin):
    list_display = ("nombre", "capacidad", "ubicacion")
    search_fields = ("nombre", "ubicacion")
    list_filter = ("ubicacion",)
    ordering = ("nombre",)


@admin.register(Seccion)
class SeccionAdmin(admin.ModelAdmin):
    list_display = ("codigo",)
    search_fields = ("codigo",)
    ordering = ("codigo",)


@admin.register(Estudiante)
class EstudianteAdmin(admin.ModelAdmin):
    list_display = ("get_full_name", "matricula", "activo", "get_email")
    search_fields = ("user__first_name", "user__last_name", "matricula", "user__email")
    list_filter = ("activo",)
    ordering = ("user__last_name", "user__first_name")

    def get_full_name(self, obj):
        return obj.user.get_full_name()
    get_full_name.short_description = "Nombre Completo"
    get_full_name.admin_order_field = "user__last_name"

    def get_email(self, obj):
        return obj.user.email
    get_email.short_description = "Correo Electrónico"


@admin.register(AsignacionClase)
class AsignacionClaseAdmin(admin.ModelAdmin):
    list_display = ("asignatura", "seccion", "profesor_name", "salon", "horario_inicio", "horario_fin")
    search_fields = ("asignatura__nombre", "seccion__codigo", "profesor__first_name", "profesor__last_name", "salon__nombre")
    list_filter = ("salon", "seccion", "horario_inicio")
    inlines = [InscripcionInline]
    ordering = ("horario_inicio", "asignatura")

    def profesor_name(self, obj):
        return obj.profesor.get_full_name()
    profesor_name.short_description = "Profesor"
    profesor_name.admin_order_field = "profesor__last_name"


@admin.register(Inscripcion)
class InscripcionAdmin(admin.ModelAdmin):
    list_display = ("estudiante", "asignacion_clase", "fecha_inscripcion")
    search_fields = ("estudiante__user__first_name", "estudiante__user__last_name", "estudiante__matricula", "asignacion_clase__asignatura__nombre")
    list_filter = ("fecha_inscripcion", "asignacion_clase__asignatura")
    ordering = ("-fecha_inscripcion",)

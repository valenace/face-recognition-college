from django.db import models
from django.conf import settings

class Asignatura(models.Model):
    nombre = models.CharField(max_length=100, verbose_name="Nombre de la Asignatura")
    codigo = models.CharField(max_length=20, unique=True, verbose_name="Código de Asignatura")
    descripcion = models.TextField(blank=True, verbose_name="Descripción")

    class Meta:
        verbose_name = "Asignatura"
        verbose_name_plural = "Asignaturas"

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"


class Salon(models.Model):
    nombre = models.CharField(max_length=50, verbose_name="Nombre del Salón")
    capacidad = models.IntegerField(verbose_name="Capacidad de Alumnos")
    ubicacion = models.CharField(max_length=100, blank=True, verbose_name="Ubicación/Pabellón")

    class Meta:
        verbose_name = "Salón"
        verbose_name_plural = "Salones"

    def __str__(self):
        return f"{self.nombre} - Cap: {self.capacidad}"


class Seccion(models.Model):
    codigo = models.CharField(max_length=20, unique=True, verbose_name="Código/Grupo de Sección")

    class Meta:
        verbose_name = "Sección"
        verbose_name_plural = "Secciones"

    def __str__(self):
        return f"Sección {self.codigo}"


class Estudiante(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        limit_choices_to={"role": "STUDENT"},
        related_name="estudiante_perfil",
        verbose_name="Usuario Estudiante"
    )
    matricula = models.CharField(max_length=30, unique=True, verbose_name="Matrícula/Carnet")
    activo = models.BooleanField(default=True, verbose_name="¿Está Activo?")

    class Meta:
        verbose_name = "Estudiante"
        verbose_name_plural = "Estudiantes"

    def __str__(self):
        return f"{self.user.get_full_name()} ({self.matricula})"


class AsignacionClase(models.Model):
    profesor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        limit_choices_to={"role": "PROFESSOR"},
        related_name="asignaciones_profesor",
        verbose_name="Profesor Asignado"
    )
    asignatura = models.ForeignKey(
        Asignatura,
        on_delete=models.CASCADE,
        related_name="asignaciones_materia",
        verbose_name="Asignatura"
    )
    seccion = models.ForeignKey(
        Seccion,
        on_delete=models.CASCADE,
        related_name="asignaciones_grupo",
        verbose_name="Sección"
    )
    salon = models.ForeignKey(
        Salon,
        on_delete=models.CASCADE,
        related_name="asignaciones_salon",
        verbose_name="Salón"
    )
    horario_inicio = models.TimeField(verbose_name="Horario de Inicio")
    horario_fin = models.TimeField(verbose_name="Horario de Fin")

    class Meta:
        verbose_name = "Asignación de Clase"
        verbose_name_plural = "Asignaciones de Clases"
        # evitar traslape de la misma seccion a la misma hora en el mismo salon
        unique_together = ("seccion", "salon", "horario_inicio")

    def __str__(self):
        return f"{self.asignatura.nombre} - Sec: {self.seccion.codigo} ({self.horario_inicio.strftime('%H:%M')} - {self.horario_fin.strftime('%H:%M')})"


class Inscripcion(models.Model):
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.CASCADE,
        related_name="inscripciones_alumno",
        verbose_name="Estudiante"
    )
    asignacion_clase = models.ForeignKey(
        AsignacionClase,
        on_delete=models.CASCADE,
        related_name="inscripciones_clase",
        verbose_name="Clase Asignada"
    )
    fecha_inscripcion = models.DateField(auto_now_add=True, verbose_name="Fecha de Inscripción")

    class Meta:
        verbose_name = "Inscripción"
        verbose_name_plural = "Inscripciones"
        unique_together = ("estudiante", "asignacion_clase")

    def __str__(self):
        return f"{self.estudiante.user.username} inscrito en {self.asignacion_clase}"

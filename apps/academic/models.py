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


from django.core.exceptions import ValidationError

class AsignacionClase(models.Model):
    class DiaSemana(models.TextChoices):
        LUNES = "LUNES", "Lunes"
        MARTES = "MARTES", "Martes"
        MIERCOLES = "MIERCOLES", "Miércoles"
        JUEVES = "JUEVES", "Jueves"
        VIERNES = "VIERNES", "Viernes"
        SABADO = "SABADO", "Sábado"
        DOMINGO = "DOMINGO", "Domingo"

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
    dia_semana = models.CharField(
        max_length=15,
        choices=DiaSemana.choices,
        default=DiaSemana.LUNES,
        verbose_name="Día de la Semana"
    )
    horario_inicio = models.TimeField(verbose_name="Horario de Inicio")
    horario_fin = models.TimeField(verbose_name="Horario de Fin")

    class Meta:
        verbose_name = "Asignación de Clase"
        verbose_name_plural = "Asignaciones de Clases"
        # evitar traslape de la misma seccion el mismo dia y a la misma hora de inicio
        unique_together = ("seccion", "dia_semana", "horario_inicio")

    def clean(self):
        super().clean()
        if self.horario_inicio and self.horario_fin and self.horario_inicio >= self.horario_fin:
            raise ValidationError("El horario de inicio debe ser anterior al horario de fin.")

        # Evitar traslape del profesor el mismo día a la misma hora
        if self.profesor and self.dia_semana and self.horario_inicio and self.horario_fin:
            overlapping_professor = AsignacionClase.objects.filter(
                profesor=self.profesor,
                dia_semana=self.dia_semana
            ).exclude(pk=self.pk)
            for cls in overlapping_professor:
                if (self.horario_inicio < cls.horario_fin) and (self.horario_fin > cls.horario_inicio):
                    raise ValidationError(
                        f"El profesor {self.profesor} ya tiene asignada la clase {cls.asignatura.nombre} en el horario {cls.horario_inicio} - {cls.horario_fin} el día {self.dia_semana.lower()}."
                    )

        # Evitar traslape de la sección el mismo día a la misma hora (cobertura total de horario)
        if self.seccion and self.dia_semana and self.horario_inicio and self.horario_fin:
            overlapping_section = AsignacionClase.objects.filter(
                seccion=self.seccion,
                dia_semana=self.dia_semana
            ).exclude(pk=self.pk)
            for cls in overlapping_section:
                if (self.horario_inicio < cls.horario_fin) and (self.horario_fin > cls.horario_inicio):
                    raise ValidationError(
                        f"La sección {self.seccion} ya tiene asignada la clase {cls.asignatura.nombre} en el horario {cls.horario_inicio} - {cls.horario_fin} el día {self.dia_semana.lower()}."
                    )

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

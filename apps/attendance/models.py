from django.db import models
from django.conf import settings
from apps.academic.models import AsignacionClase, Estudiante

class FaceEmbedding(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="face_data")
    embedding = models.JSONField()  # guardar lista de floats
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for {self.user.username}"


class SesionClase(models.Model):
    class Estado(models.TextChoices):
        EN_CURSO = "EN_CURSO", "en curso"
        FINALIZADA = "FINALIZADA", "finalizada"

    asignacion_clase = models.ForeignKey(
        AsignacionClase,
        on_delete=models.CASCADE,
        related_name="sesiones",
        verbose_name="asignación de clase"
    )
    fecha = models.DateField(auto_now_add=True, verbose_name="fecha")
    estado = models.CharField(
        max_length=20,
        choices=Estado.choices,
        default=Estado.EN_CURSO,
        verbose_name="estado"
    )

    class Meta:
        verbose_name = "sesión de clase"
        verbose_name_plural = "sesiones de clases"

    def __str__(self):
        return f"{self.asignacion_clase.asignatura.nombre} - {self.fecha} ({self.get_estado_display()})"


class RegistroAsistencia(models.Model):
    sesion = models.ForeignKey(SesionClase, on_delete=models.CASCADE, related_name="asistencias", verbose_name="sesión")
    estudiante = models.ForeignKey(Estudiante, on_delete=models.CASCADE, related_name="asistencias", verbose_name="estudiante")
    hora_entrada = models.TimeField(null=True, blank=True, verbose_name="hora de entrada")
    hora_salida = models.TimeField(null=True, blank=True, verbose_name="hora de salida")
    similitud_ia = models.FloatField(null=True, blank=True, verbose_name="similitud ia")
    es_fraude = models.BooleanField(default=False, verbose_name="¿es fraude?")
    captura_fraude = models.ImageField(upload_to="fraudes/", null=True, blank=True, verbose_name="captura de fraude")

    class Meta:
        verbose_name = "registro de asistencia"
        verbose_name_plural = "registros de asistencia"
        unique_together = ("sesion", "estudiante")

    def __str__(self):
        return f"{self.estudiante.user.username} - {self.sesion.asignacion_clase.asignatura.nombre} ({self.sesion.fecha})"

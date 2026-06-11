from django.db import models
from django.conf import settings
from apps.academic.models import AsignacionClase, Estudiante
from datetime import datetime
from django.utils import timezone

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
    notas_profesor = models.TextField(blank=True, default="", verbose_name="notas del profesor")

    class Meta:
        verbose_name = "sesión de clase"
        verbose_name_plural = "sesiones de clases"

    def __str__(self):
        return f"{self.asignacion_clase.asignatura.nombre} - {self.fecha} ({self.get_estado_display()})"


class RegistroAsistencia(models.Model):
    class TipoEvento(models.TextChoices):
        ASISTENCIA_NORMAL = "ASISTENCIA_NORMAL", "Asistencia Normal"
        INTRUSO = "INTRUSO", "Intruso"
        OTRA_SECCION = "OTRA_SECCION", "Otra Sección"
        SPOOFING = "SPOOFING", "Spoofing"

    sesion = models.ForeignKey(SesionClase, on_delete=models.CASCADE, related_name="asistencias", verbose_name="sesión")
    estudiante = models.ForeignKey(Estudiante, on_delete=models.CASCADE, null=True, blank=True, related_name="asistencias", verbose_name="estudiante")
    hora_entrada = models.TimeField(null=True, blank=True, verbose_name="hora de entrada")
    hora_salida = models.TimeField(null=True, blank=True, verbose_name="hora de salida")
    ultima_vez_visto = models.TimeField(null=True, blank=True, verbose_name="última vez visto")
    similitud_ia = models.FloatField(null=True, blank=True, verbose_name="similitud ia")
    es_fraude = models.BooleanField(default=False, verbose_name="¿es fraude?")
    captura_fraude = models.ImageField(upload_to="fraudes/", null=True, blank=True, verbose_name="captura de fraude")
    tipo_evento = models.CharField(
        max_length=20,
        choices=TipoEvento.choices,
        default=TipoEvento.ASISTENCIA_NORMAL,
        verbose_name="tipo de evento"
    )

    class Meta:
        verbose_name = "registro de asistencia"
        verbose_name_plural = "registros de asistencia"
        unique_together = ("sesion", "estudiante")

    @property
    def posible_fuga(self):
        if not self.hora_entrada or not self.ultima_vez_visto:
            return False
        today = datetime.today()
        dt_entrada = datetime.combine(today, self.hora_entrada)
        dt_visto = datetime.combine(today, self.ultima_vez_visto)
        diff_seconds = abs((dt_visto - dt_entrada).total_seconds())
        return diff_seconds < 900

    @property
    def porcentaje_permanencia(self):
        if not self.hora_entrada:
            return 0
        
        end_time = self.ultima_vez_visto
        if not end_time:
            end_time = self.hora_salida
        if not end_time:
            if self.sesion.estado == "EN_CURSO":
                end_time = timezone.localtime().time()
            else:
                return 0
                
        today = datetime.today()
        dt_entrada = datetime.combine(today, self.hora_entrada)
        dt_fin = datetime.combine(today, end_time)
        
        diff_minutes = (dt_fin - dt_entrada).total_seconds() / 60.0
        if diff_minutes < 0:
            diff_minutes = abs(diff_minutes)
            
        horario_inicio = self.sesion.asignacion_clase.horario_inicio
        horario_fin = self.sesion.asignacion_clase.horario_fin
        
        dt_class_start = datetime.combine(today, horario_inicio)
        dt_class_end = datetime.combine(today, horario_fin)
        class_duration_minutes = (dt_class_end - dt_class_start).total_seconds() / 60.0
        
        if class_duration_minutes <= 0:
            return 0
            
        porcentaje = (diff_minutes / class_duration_minutes) * 100
        return min(100, max(0, int(round(porcentaje))))

    def __str__(self):
        estudiante_name = self.estudiante.user.username if self.estudiante else "Desconocido"
        return f"{estudiante_name} - {self.sesion.asignacion_clase.asignatura.nombre} ({self.sesion.fecha})"

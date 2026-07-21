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
    alerta_revisada = models.BooleanField(default=False, verbose_name="¿alerta revisada?")
    notas_auditoria = models.TextField(blank=True, default="", verbose_name="notas de auditoría")
    historial_intervalos = models.JSONField(default=list, blank=True, verbose_name="historial de intervalos")

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

    def registrar_ping(self, hora_actual):
        from datetime import datetime
        if isinstance(hora_actual, str):
            try:
                t_val = datetime.strptime(hora_actual, "%H:%M:%S").time()
            except ValueError:
                t_val = datetime.strptime(hora_actual, "%H:%M").time()
        else:
            t_val = hora_actual

        hora_str = t_val.strftime("%H:%M")

        if not self.historial_intervalos:
            self.historial_intervalos = [{"inicio": hora_str, "fin": hora_str}]
        else:
            ultimo = self.historial_intervalos[-1]
            try:
                t_fin = datetime.strptime(ultimo["fin"], "%H:%M").time()
            except ValueError:
                t_fin = datetime.strptime(ultimo["fin"], "%H:%M:%S").time()

            today = datetime.today()
            dt_fin = datetime.combine(today, t_fin)
            dt_actual = datetime.combine(today, t_val)

            diff_minutes = (dt_actual - dt_fin).total_seconds() / 60.0
            if diff_minutes < 0:
                diff_minutes = abs(diff_minutes)

            if diff_minutes <= 3.0:
                ultimo["fin"] = hora_str
            else:
                self.historial_intervalos.append({"inicio": hora_str, "fin": hora_str})

        self.historial_intervalos = list(self.historial_intervalos)
        self.ultima_vez_visto = t_val
        self.save()

    @property
    def porcentaje_permanencia(self):
        if self.notas_auditoria == "Detección por Imagen":
            return 100
        horario_inicio = self.sesion.asignacion_clase.horario_inicio
        horario_fin = self.sesion.asignacion_clase.horario_fin
        
        from datetime import datetime
        if isinstance(horario_inicio, str):
            try:
                horario_inicio = datetime.strptime(horario_inicio, "%H:%M:%S").time()
            except ValueError:
                horario_inicio = datetime.strptime(horario_inicio, "%H:%M").time()
                
        if isinstance(horario_fin, str):
            try:
                horario_fin = datetime.strptime(horario_fin, "%H:%M:%S").time()
            except ValueError:
                horario_fin = datetime.strptime(horario_fin, "%H:%M").time()

        today = datetime.today()
        dt_class_start = datetime.combine(today, horario_inicio)
        dt_class_end = datetime.combine(today, horario_fin)
        class_duration_minutes = (dt_class_end - dt_class_start).total_seconds() / 60.0
        
        if class_duration_minutes <= 0:
            return 0

        if self.historial_intervalos:
            total_minutos_permanencia = 0.0
            for block in self.historial_intervalos:
                inicio_str = block["inicio"]
                fin_str = block["fin"]
                dt_inicio = datetime.strptime(inicio_str, "%H:%M")
                dt_fin = datetime.strptime(fin_str, "%H:%M")
                diff = (dt_fin - dt_inicio).total_seconds() / 60.0
                if diff < 0:
                    diff = abs(diff)
                total_minutos_permanencia += diff
        else:
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
            
            dt_entrada = datetime.combine(today, self.hora_entrada)
            dt_fin = datetime.combine(today, end_time)
            
            total_minutos_permanencia = (dt_fin - dt_entrada).total_seconds() / 60.0
            if total_minutos_permanencia < 0:
                total_minutos_permanencia = abs(total_minutos_permanencia)

        porcentaje = (total_minutos_permanencia / class_duration_minutes) * 100
        return min(100, max(0, int(round(porcentaje))))

    def __str__(self):
        estudiante_name = self.estudiante.user.username if self.estudiante else "Desconocido"
        return f"{estudiante_name} - {self.sesion.asignacion_clase.asignatura.nombre} ({self.sesion.fecha})"

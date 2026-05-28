from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from apps.academic.models import Asignatura, Salon, Seccion, Estudiante, AsignacionClase, Inscripcion
from apps.attendance.models import SesionClase, RegistroAsistencia
from datetime import time

User = get_user_model()

class Command(BaseCommand):
    help = "Genera datos de prueba para ensayar la interfaz del profesor (dashboard y monitor de clases)"

    def handle(self, *args, **options):
        self.stdout.write("generando datos de prueba...")

        # 1. crear profesor
        prof, created = User.objects.get_or_create(
            username="profesor1",
            defaults={
                "email": "profesor1@college.edu",
                "first_name": "Roberto",
                "last_name": "Gómez",
                "role": User.Role.PROFESSOR
            }
        )
        if created:
            prof.set_password("admin123")
            prof.save()
            self.stdout.write("profesor 'profesor1' creado exitosamente.")
        else:
            self.stdout.write("el profesor 'profesor1' ya existe.")

        # 2. crear asignatura
        materia, _ = Asignatura.objects.get_or_create(
            codigo="IA-101",
            defaults={
                "nombre": "Inteligencia Artificial",
                "descripcion": "introducción a redes neuronales y visión artificial."
            }
        )

        # 3. crear salon
        salon, _ = Salon.objects.get_or_create(
            nombre="Laboratorio de Cómputo 3",
            defaults={
                "capacidad": 30,
                "ubicacion": "pabellón b, segundo piso"
            }
        )

        # 4. crear seccion
        seccion, _ = Seccion.objects.get_or_create(codigo="IA-01")

        # 5. crear asignación de clase
        asignacion, _ = AsignacionClase.objects.get_or_create(
            seccion=seccion,
            salon=salon,
            horario_inicio=time(8, 0),
            defaults={
                "profesor": prof,
                "asignatura": materia,
                "horario_fin": time(10, 0)
            }
        )

        # 6. crear sesión de clase
        sesion, _ = SesionClase.objects.get_or_create(
            asignacion_clase=asignacion,
            estado=SesionClase.Estado.EN_CURSO
        )

        # 7. crear alumnos de prueba
        alumnos_data = [
            ("estudiante1", "Laura", "Pérez", "20261001"),
            ("estudiante2", "Carlos", "Sosa", "20261002"),
            ("estudiante3", "Sofía", "Castro", "20261003"),
        ]

        for username, first_name, last_name, matricula in alumnos_data:
            user_est, user_created = User.objects.get_or_create(
                username=username,
                defaults={
                    "email": f"{username}@college.edu",
                    "first_name": first_name,
                    "last_name": last_name,
                    "role": User.Role.STUDENT
                }
            )
            if user_created:
                user_est.set_password("admin123")
                user_est.save()

            estudiante_profile, _ = Estudiante.objects.get_or_create(
                user=user_est,
                defaults={
                    "matricula": matricula,
                    "activo": True
                }
            )

            # inscribir alumno en la clase
            Inscripcion.objects.get_or_create(
                estudiante=estudiante_profile,
                asignacion_clase=asignacion
            )

        # 8. crear algunos registros de asistencia (uno normal y uno fraudulento)
        est1 = Estudiante.objects.get(matricula="20261001")
        est2 = Estudiante.objects.get(matricula="20261002")

        # laura entra bien
        RegistroAsistencia.objects.get_or_create(
            sesion=sesion,
            estudiante=est1,
            defaults={
                "hora_entrada": time(8, 5, 12),
                "similitud_ia": 0.94,
                "es_fraude": False
            }
        )

        # carlos intenta entrar con suplantación (fraude)
        RegistroAsistencia.objects.get_or_create(
            sesion=sesion,
            estudiante=est2,
            defaults={
                "hora_entrada": time(8, 14, 45),
                "similitud_ia": 0.38,
                "es_fraude": True
            }
        )

        self.stdout.write("generación de datos de prueba completada exitosamente.")

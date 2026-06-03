import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import transaction
from django.conf import settings

from apps.academic.models import Asignatura, Salon, Seccion, Estudiante, AsignacionClase, Inscripcion
from apps.attendance.models import SesionClase, RegistroAsistencia

User = get_user_model()

class Command(BaseCommand):
    help = "Populates the database from the mock_universidad.json API mock file"

    def handle(self, *args, **options):
        json_path = settings.BASE_DIR / "mock_universidad.json"
        if not json_path.exists():
            self.stderr.write(self.style.ERROR(f"File not found: {json_path}"))
            return

        self.stdout.write("Reading mock_universidad.json...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            with transaction.atomic():
                # 1. Cargar usuarios administrativos
                self.stdout.write("Loading administrative users...")
                for admin_data in data.get("usuarios_administrativos", []):
                    username = admin_data["username"]
                    user, created = User.objects.get_or_create(
                        username=username,
                        defaults={
                            "first_name": admin_data["first_name"],
                            "last_name": admin_data["last_name"],
                            "email": admin_data["email"],
                            "role": admin_data["role"]
                        }
                    )
                    if created:
                        user.set_password(admin_data["password"])
                        user.save()
                        self.stdout.write(f"Created admin user: {username}")
                    else:
                        self.stdout.write(f"Admin user {username} already exists.")

                # 2. Cargar profesores
                self.stdout.write("Loading professors...")
                for prof_data in data.get("profesores", []):
                    username = prof_data["username"]
                    user, created = User.objects.get_or_create(
                        username=username,
                        defaults={
                            "first_name": prof_data["first_name"],
                            "last_name": prof_data["last_name"],
                            "email": prof_data["email"],
                            "role": User.Role.PROFESSOR
                        }
                    )
                    if created:
                        user.set_password(prof_data["password"])
                        user.save()
                        self.stdout.write(f"Created professor: {username}")
                    else:
                        self.stdout.write(f"Professor {username} already exists.")

                # 3. Cargar estudiantes
                self.stdout.write("Loading students...")
                for est_data in data.get("estudiantes", []):
                    username = est_data["username"]
                    user, created = User.objects.get_or_create(
                        username=username,
                        defaults={
                            "first_name": est_data["first_name"],
                            "last_name": est_data["last_name"],
                            "email": est_data["email"],
                            "role": User.Role.STUDENT
                        }
                    )
                    if created:
                        user.set_password(est_data["password"])
                        user.save()
                    
                    estudiante, est_created = Estudiante.objects.get_or_create(
                        user=user,
                        defaults={
                            "matricula": est_data["matricula"],
                            "activo": est_data.get("activo", True)
                        }
                    )
                    if est_created:
                        self.stdout.write(f"Created student: {username} ({est_data['matricula']})")
                    else:
                        self.stdout.write(f"Student profile for {username} already exists.")

                # 4. Cargar Infraestructura
                infra = data.get("infraestructura", {})
                
                self.stdout.write("Loading subjects...")
                for sub_data in infra.get("asignaturas", []):
                    sub, sub_created = Asignatura.objects.get_or_create(
                        codigo=sub_data["codigo"],
                        defaults={
                            "nombre": sub_data["nombre"],
                            "descripcion": sub_data.get("descripcion", "")
                        }
                    )
                    if sub_created:
                        self.stdout.write(f"Created subject: {sub.nombre} ({sub.codigo})")

                self.stdout.write("Loading class rooms...")
                for salon_data in infra.get("salones", []):
                    salon, salon_created = Salon.objects.get_or_create(
                        nombre=salon_data["nombre"],
                        defaults={
                            "capacidad": salon_data["capacidad"],
                            "ubicacion": salon_data.get("ubicacion", "")
                        }
                    )
                    if salon_created:
                        self.stdout.write(f"Created classroom: {salon.nombre}")

                self.stdout.write("Loading sections...")
                for sec_data in infra.get("secciones", []):
                    sec, sec_created = Seccion.objects.get_or_create(
                        codigo=sec_data["codigo"]
                    )
                    if sec_created:
                        self.stdout.write(f"Created section: {sec.codigo}")

                # 5. Planificación Académica
                self.stdout.write("Loading academic planning and enrollments...")
                for plan in data.get("planificacion_academica", []):
                    profesor = User.objects.get(username=plan["profesor_username"])
                    asignatura = Asignatura.objects.get(codigo=plan["asignatura_codigo"])
                    seccion = Seccion.objects.get(codigo=plan["seccion_codigo"])
                    salon = Salon.objects.get(nombre=plan["salon_nombre"])

                    asignacion, plan_created = AsignacionClase.objects.get_or_create(
                        seccion=seccion,
                        salon=salon,
                        horario_inicio=plan["horario_inicio"],
                        defaults={
                            "profesor": profesor,
                            "asignatura": asignatura,
                            "horario_fin": plan["horario_fin"]
                        }
                    )
                    if plan_created:
                        self.stdout.write(f"Created class assignment: {asignatura.nombre} - Sec: {seccion.codigo}")
                    
                    # Inscribir estudiantes
                    for est_username in plan.get("inscritos", []):
                        est_user = User.objects.get(username=est_username)
                        estudiante = Estudiante.objects.get(user=est_user)
                        insc, insc_created = Inscripcion.objects.get_or_create(
                            estudiante=estudiante,
                            asignacion_clase=asignacion
                        )
                        if insc_created:
                            self.stdout.write(f"Enrolled {est_username} in {asignatura.nombre}")

                # 6. Historial de Sesiones Pasadas
                self.stdout.write("Loading past class sessions and attendance history...")
                for hist in data.get("historial_sesiones_pasadas", []):
                    asignatura = Asignatura.objects.get(codigo=hist["asignatura_codigo"])
                    seccion = Seccion.objects.get(codigo=hist["seccion_codigo"])
                    
                    # Buscar la asignación de clase correspondiente
                    asignacion = AsignacionClase.objects.filter(
                        asignatura=asignatura,
                        seccion=seccion
                    ).first()
                    
                    if not asignacion:
                        self.stderr.write(self.style.WARNING(f"Assignment not found for subject {asignatura.codigo} and section {seccion.codigo}"))
                        continue
                    
                    # Crear o buscar la sesión de clase
                    sesion, sesion_created = SesionClase.objects.get_or_create(
                        asignacion_clase=asignacion,
                        fecha=hist["fecha"],
                        defaults={
                            "estado": SesionClase.Estado.FINALIZADA
                        }
                    )
                    if sesion_created:
                        self.stdout.write(f"Created past class session for {asignatura.nombre} on {hist['fecha']}")
                    
                    # Cargar los registros de asistencia de la sesión
                    for reg in hist.get("registros", []):
                        est_user = User.objects.get(username=reg["username"])
                        estudiante = Estudiante.objects.get(user=est_user)
                        
                        registro, reg_created = RegistroAsistencia.objects.update_or_create(
                            sesion=sesion,
                            estudiante=estudiante,
                            defaults={
                                "hora_entrada": reg["hora_entrada"],
                                "hora_salida": reg["hora_salida"],
                                "similitud_ia": reg["similitud_ia"],
                                "es_fraude": reg["es_fraude"]
                            }
                        )
                        if reg_created:
                            self.stdout.write(f"Created attendance record for {reg['username']}")

            self.stdout.write(self.style.SUCCESS("Database seeding completed successfully!"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Transaction failed, seeding aborted: {e}"))
            raise e

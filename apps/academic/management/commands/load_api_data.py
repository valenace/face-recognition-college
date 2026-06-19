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
                self.stdout.write("Clearing existing attendance sessions and records...")
                RegistroAsistencia.objects.all().delete()
                SesionClase.objects.all().delete()

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
                        dia_semana=plan.get("dia_semana", "LUNES"),
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
                    
                    # Crear o buscar la sesión de clase (bypassing auto_now_add using update)
                    sesion = SesionClase.objects.filter(
                        asignacion_clase=asignacion,
                        fecha=hist["fecha"]
                    ).first()
                    if not sesion:
                        sesion = SesionClase.objects.create(
                            asignacion_clase=asignacion,
                            estado=SesionClase.Estado.FINALIZADA
                        )
                        SesionClase.objects.filter(pk=sesion.pk).update(fecha=hist["fecha"])
                        sesion.refresh_from_db()
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

                # Programmatic fallback: ensure all assignments have past finalized sessions, live sessions, and future sessions
                self.stdout.write("Ensuring all assignments have past finalized sessions, live sessions, and future sessions...")
                from datetime import date, timedelta
                for asignacion in AsignacionClase.objects.all():
                    # 1. Create 3 past finalized sessions
                    for days_ago in [21, 14, 7]:
                        past_date = date.today() - timedelta(days=days_ago)
                        sesion = SesionClase.objects.filter(
                            asignacion_clase=asignacion,
                            fecha=past_date
                        ).first()
                        if not sesion:
                            sesion = SesionClase.objects.create(
                                asignacion_clase=asignacion,
                                estado=SesionClase.Estado.FINALIZADA,
                                notas_profesor=f"Clase dictada el {past_date}. Todo en orden."
                            )
                            SesionClase.objects.filter(pk=sesion.pk).update(fecha=past_date)
                            sesion.refresh_from_db()
                            self.stdout.write(f"Created programmatic past session for {asignacion} on {past_date}")
                            
                            # Add present/absent students
                            inscritos = Inscripcion.objects.filter(asignacion_clase=asignacion)
                            for i, insc in enumerate(inscritos):
                                if i % 2 == 0:
                                    # Present
                                    RegistroAsistencia.objects.create(
                                        sesion=sesion,
                                        estudiante=insc.estudiante,
                                        hora_entrada="09:05:00",
                                        hora_salida="10:55:00",
                                        ultima_vez_visto="10:55:00",
                                        similitud_ia=0.92,
                                        es_fraude=False,
                                        historial_intervalos=[
                                            {"inicio": "09:05", "fin": "09:40"},
                                            {"inicio": "09:45", "fin": "10:55"}
                                        ]
                                    )

                    # 2. Create a live session (today)
                    today_date = date.today()
                    sesion_en_curso = SesionClase.objects.filter(
                        asignacion_clase=asignacion,
                        fecha=today_date
                    ).first()
                    if not sesion_en_curso:
                        sesion_en_curso = SesionClase.objects.create(
                            asignacion_clase=asignacion,
                            estado=SesionClase.Estado.EN_CURSO,
                            notas_profesor=""
                        )
                        # today_date doesn't need to be updated with update() but we can still do it to be safe
                        SesionClase.objects.filter(pk=sesion_en_curso.pk).update(fecha=today_date)
                        sesion_en_curso.refresh_from_db()
                        self.stdout.write(f"Created active live session for {asignacion} on {today_date}")
                        
                        # Add some active registrations
                        inscritos = Inscripcion.objects.filter(asignacion_clase=asignacion)
                        for i, insc in enumerate(inscritos):
                            if i == 0:
                                # Present
                                RegistroAsistencia.objects.create(
                                    sesion=sesion_en_curso,
                                    estudiante=insc.estudiante,
                                    hora_entrada="09:02:00",
                                    ultima_vez_visto="09:15:00",
                                    similitud_ia=0.94,
                                    es_fraude=False,
                                    historial_intervalos=[
                                        {"inicio": "09:02", "fin": "09:15"}
                                    ]
                                )
                            elif i == 1:
                                # Fraud / spoofing alert active
                                RegistroAsistencia.objects.create(
                                    sesion=sesion_en_curso,
                                    estudiante=insc.estudiante,
                                    hora_entrada="09:05:00",
                                    ultima_vez_visto="09:05:00",
                                    similitud_ia=0.38,
                                    es_fraude=True,
                                    tipo_evento=RegistroAsistencia.TipoEvento.SPOOFING,
                                    historial_intervalos=[
                                        {"inicio": "09:05", "fin": "09:05"}
                                    ]
                                )

                    # 3. Create a future session (e.g. in 2 days)
                    future_date = date.today() + timedelta(days=2)
                    sesion_futura = SesionClase.objects.filter(
                        asignacion_clase=asignacion,
                        fecha=future_date
                    ).first()
                    if not sesion_futura:
                        sesion_futura = SesionClase.objects.create(
                            asignacion_clase=asignacion,
                            estado=SesionClase.Estado.EN_CURSO,
                            notas_profesor="Sesión programada a futuro."
                        )
                        SesionClase.objects.filter(pk=sesion_futura.pk).update(fecha=future_date)
                        sesion_futura.refresh_from_db()
                        self.stdout.write(f"Created future scheduled session for {asignacion} on {future_date}")

            self.stdout.write(self.style.SUCCESS("Database seeding completed successfully!"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Transaction failed, seeding aborted: {e}"))
            raise e

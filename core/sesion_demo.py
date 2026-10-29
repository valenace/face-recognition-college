"""
Sesión de demostración para probar los flujos de asistencia sin datos académicos.

La usan `manage.py ver_flujo --crear-sesion` y el motor de consola
(`scripts/asistencia_unificada_optimizada.py --crear-sesion`). Crea el profesor,
la asignatura, el salón, la sección, la asignación y las inscripciones de todos
los estudiantes, y devuelve la `SesionClase` abierta.

No importa Django al cargar el módulo: los scripts de `scripts/` deben poder
importarlo desde la terminal antes de configurar el entorno.
"""

import os


def configurar_django():
    """Configura Django una sola vez, sin importarlo al cargar el módulo."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cv_college.settings")
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


def crear_sesion_demo():
    """Crea una sesión de demo nueva y devuelve la `SesionClase`."""
    configurar_django()

    from django.contrib.auth import get_user_model

    from apps.academic.models import (
        AsignacionClase,
        Asignatura,
        Estudiante,
        Inscripcion,
        Salon,
        Seccion,
    )
    from apps.attendance.models import SesionClase

    User = get_user_model()
    profesor, _ = User.objects.get_or_create(
        username="prof_flujo",
        defaults={"first_name": "Profesor", "last_name": "Flujo",
                  "role": User.Role.PROFESSOR},
    )
    asignatura, _ = Asignatura.objects.get_or_create(
        codigo="DEMO-FLUJO", defaults={"nombre": "Demo de flujo"}
    )
    salon, _ = Salon.objects.get_or_create(
        nombre="Salón Demo", defaults={"capacidad": 30}
    )
    seccion, _ = Seccion.objects.get_or_create(codigo="DEMO-S")
    asignacion, _ = AsignacionClase.objects.get_or_create(
        profesor=profesor, asignatura=asignatura, seccion=seccion, salon=salon,
        defaults={"dia_semana": AsignacionClase.DiaSemana.LUNES,
                  "horario_inicio": "08:00", "horario_fin": "10:00"},
    )
    for estudiante in Estudiante.objects.all():
        Inscripcion.objects.get_or_create(
            estudiante=estudiante, asignacion_clase=asignacion
        )

    # Cada llamada crea una sesión nueva: no se reutiliza ni se borran datos de
    # una captura anterior.
    return SesionClase.objects.create(
        asignacion_clase=asignacion, estado=SesionClase.Estado.EN_CURSO
    )


def limpiar_sesion_demo(sesion):
    """Borra la sesión de demo y, si no quedan otras, su andamiaje académico."""
    configurar_django()

    from django.contrib.auth import get_user_model

    from apps.academic.models import (
        Asignatura,
        Inscripcion,
        Salon,
        Seccion,
    )
    from apps.attendance.models import SesionClase

    asignacion = sesion.asignacion_clase
    sesion.delete()

    # Si quedan otras sesiones de la asignación de demo, se conservan.
    if SesionClase.objects.filter(asignacion_clase=asignacion).exists():
        return

    Inscripcion.objects.filter(asignacion_clase=asignacion).delete()
    asignacion.delete()
    Asignatura.objects.filter(codigo="DEMO-FLUJO").delete()
    Salon.objects.filter(nombre="Salón Demo").delete()
    Seccion.objects.filter(codigo="DEMO-S").delete()
    get_user_model().objects.filter(username="prof_flujo").delete()

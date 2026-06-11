from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from apps.academic.models import Estudiante, AsignacionClase, Asignatura, Salon, Seccion
from apps.attendance.models import SesionClase, RegistroAsistencia

User = get_user_model()

class AttendanceModelsAndViewsTestCase(TestCase):
    def setUp(self):
        # Create professors
        self.professor_owner = User.objects.create_user(
            username="professor_owner",
            password="password123",
            role=User.Role.PROFESSOR
        )
        self.professor_other = User.objects.create_user(
            username="professor_other",
            password="password123",
            role=User.Role.PROFESSOR
        )
        self.student_user = User.objects.create_user(
            username="student_user",
            password="password123",
            role=User.Role.STUDENT
        )

        # Create academic entities
        self.estudiante = Estudiante.objects.create(
            user=self.student_user,
            matricula="MAT-888"
        )
        self.asignatura = Asignatura.objects.create(
            nombre="Fisica",
            codigo="FIS101"
        )
        self.salon = Salon.objects.create(
            nombre="Laboratorio 1",
            capacidad=25
        )
        self.seccion = Seccion.objects.create(
            codigo="B"
        )
        self.asignacion = AsignacionClase.objects.create(
            profesor=self.professor_owner,
            asignatura=self.asignatura,
            seccion=self.seccion,
            salon=self.salon,
            horario_inicio="09:00:00",
            horario_fin="11:00:00"
        )
        self.sesion = SesionClase.objects.create(
            asignacion_clase=self.asignacion,
            estado=SesionClase.Estado.EN_CURSO
        )

    def test_registro_asistencia_nullable_estudiante(self):
        # Intruso (estudiante = None)
        registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=None,
            tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO,
            es_fraude=True
        )
        self.assertIsNone(registro.estudiante)
        self.assertEqual(registro.tipo_evento, RegistroAsistencia.TipoEvento.INTRUSO)
        self.assertIn("Desconocido", str(registro))
        self.assertIn("Fisica", str(registro))

    def test_clase_en_vivo_view_get_redirects(self):
        self.client.login(username="professor_owner", password="password123")
        url = reverse('clase-en-vivo', kwargs={'session_id': self.sesion.id})
        response = self.client.get(url)
        self.assertRedirects(response, reverse('academic:monitor-clase', kwargs={'session_id': self.sesion.id}))

    def test_clase_en_vivo_view_post_finalizes_session(self):
        from datetime import time
        # Create assistance record with entry time and last seen time
        registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=self.estudiante,
            hora_entrada=time(9, 5),
            ultima_vez_visto=time(9, 12),
            es_fraude=False
        )
        self.client.login(username="professor_owner", password="password123")
        url = reverse('clase-en-vivo', kwargs={'session_id': self.sesion.id})
        post_data = {
            'notas_profesor': 'Todo en orden con la clase de hoy.'
        }
        response = self.client.post(url, data=post_data)
        
        # Verify redirect to professor dashboard
        self.assertRedirects(response, reverse('academic:dashboard-profesor'))
        
        # Verify db changes
        self.sesion.refresh_from_db()
        self.assertEqual(self.sesion.estado, SesionClase.Estado.FINALIZADA)
        self.assertEqual(self.sesion.notas_profesor, 'Todo en orden con la clase de hoy.')

        # Verify exit time sealing
        registro.refresh_from_db()
        self.assertEqual(registro.hora_salida, time(9, 12))
        
        # Verify posible_fuga property (9:05 to 9:12 is 7 minutes, which is < 15 minutes)
        self.assertTrue(registro.posible_fuga)

        # Verify porcentaje_permanencia calculation
        # 7 minutes of 120 minutes total duration: (7 / 120) * 100 = 5.83%, which rounds to 6%
        self.assertEqual(registro.porcentaje_permanencia, 6)

    def test_clase_en_vivo_view_unauthorized_professor(self):
        self.client.login(username="professor_other", password="password123")
        url = reverse('clase-en-vivo', kwargs={'session_id': self.sesion.id})
        post_data = {
            'notas_profesor': 'Intento de modificar notas.'
        }
        # RoleRequiredMixin redirects unauthorized/unpermitted role/owner to 'dashboard'
        response = self.client.post(url, data=post_data)
        self.assertRedirects(response, reverse('dashboard'), fetch_redirect_response=False)
        
        # Verify db has not changed
        self.sesion.refresh_from_db()
        self.assertEqual(self.sesion.estado, SesionClase.Estado.EN_CURSO)
        self.assertEqual(self.sesion.notas_profesor, '')

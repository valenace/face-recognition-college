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

    def test_clase_en_vivo_view_get_renders_directly(self):
        self.client.login(username="professor_owner", password="password123")
        url = reverse('academic:monitor-clase', kwargs={'session_id': self.sesion.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

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
        url = reverse('academic:monitor-clase', kwargs={'session_id': self.sesion.id})
        post_data = {
            'notes_profesor': 'Todo en orden con la clase de hoy.'
        }
        # Wait, the field in the view is notas_profesor, let's keep 'notas_profesor'
        post_data = {
            'notas_profesor': 'Todo en orden con la clase de hoy.'
        }
        response = self.client.post(url, data=post_data)
        
        # Verify redirect to professor dashboard
        self.assertRedirects(response, reverse('academic:panel-profesor'))
        
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
        url = reverse('academic:monitor-clase', kwargs={'session_id': self.sesion.id})
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

    def test_resolver_alerta_api_success(self):
        import json
        # Create assistance record marked as fraud
        registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=self.estudiante,
            es_fraude=True
        )
        self.client.login(username="professor_owner", password="password123")
        url = reverse('academic:api-resolver-alerta')
        
        # Test falsa_alarma
        data = {
            "registro_id": registro.id,
            "accion": "falsa_alarma"
        }
        response = self.client.post(url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success"})
        
        registro.refresh_from_db()
        self.assertFalse(registro.es_fraude)
        self.assertTrue(registro.alerta_revisada)
        self.assertEqual(registro.notas_auditoria, "Marcado como falsa alarma por el profesor")
        
        # Test confirmar_fraude
        data = {
            "registro_id": registro.id,
            "accion": "confirmar_fraude"
        }
        response = self.client.post(url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success"})
        
        registro.refresh_from_db()
        self.assertTrue(registro.es_fraude)
        self.assertTrue(registro.alerta_revisada)
        self.assertEqual(registro.notas_auditoria, "Fraude confirmado por el profesor")

    def test_resolver_alerta_api_unauthorized(self):
        import json
        registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=self.estudiante,
            es_fraude=True
        )
        self.client.login(username="professor_other", password="password123")
        url = reverse('academic:api-resolver-alerta')
        
        data = {
            "registro_id": registro.id,
            "accion": "falsa_alarma"
        }
        response = self.client.post(url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 403)
        
        # Verify db has not changed
        registro.refresh_from_db()
        self.assertTrue(registro.es_fraude)
        self.assertFalse(registro.alerta_revisada)
        self.assertEqual(registro.notas_auditoria, "")

    def test_registrar_ping_and_porcentaje_permanencia_intervals(self):
        from datetime import time
        registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=self.estudiante,
            hora_entrada=time(9, 0)
        )
        # 1. Ping on empty history
        registro.registrar_ping(time(9, 5))
        self.assertEqual(registro.historial_intervalos, [{"inicio": "09:05", "fin": "09:05"}])
        self.assertEqual(registro.ultima_vez_visto, time(9, 5))

        # 2. Ping <= 3 minutes (9:07 - 9:05 = 2 minutes)
        registro.registrar_ping("09:07")
        self.assertEqual(registro.historial_intervalos, [{"inicio": "09:05", "fin": "09:07"}])

        # 3. Ping > 3 minutes (9:12 - 9:07 = 5 minutes)
        registro.registrar_ping("09:12")
        self.assertEqual(registro.historial_intervalos, [
            {"inicio": "09:05", "fin": "09:07"},
            {"inicio": "09:12", "fin": "09:12"}
        ])

        # 4. Another ping <= 3 minutes to extend the new block (9:15 - 9:12 = 3 minutes)
        registro.registrar_ping(time(9, 15))
        self.assertEqual(registro.historial_intervalos, [
            {"inicio": "09:05", "fin": "09:07"},
            {"inicio": "09:12", "fin": "09:15"}
        ])

        # 5. Check porcentaje_permanencia calculation:
        # Interval 1: 09:05 to 09:07 -> 2 minutes
        # Interval 2: 09:12 to 09:15 -> 3 minutes
        # Total active minutes: 5 minutes
        # Total class duration: 9:00 to 11:00 -> 120 minutes
        # (5 / 120) * 100 = 4.16% -> rounded to 4%
        self.assertEqual(registro.porcentaje_permanencia, 4)

    from unittest.mock import patch

    @patch('apps.attendance.views.threading.Thread')
    def test_procesar_dataset_view_trigger(self, mock_thread):
        self.client.login(username="professor_owner", password="password123")
        url = reverse('academic:api-procesar-dataset', kwargs={'session_id': self.sesion.id})
        response = self.client.post(
            url,
            data='{"ruta_carpeta": "data/images/"}',
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn("Procesamiento batch de imágenes", data['message'])
        self.assertTrue(mock_thread.called)
        self.assertTrue(mock_thread.return_value.start.called)

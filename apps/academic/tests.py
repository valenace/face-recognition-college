from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
from apps.academic.models import Estudiante, AsignacionClase, Asignatura, Salon, Seccion
from apps.attendance.models import FaceEmbedding, SesionClase, RegistroAsistencia
import json

User = get_user_model()

class CoordinationViewsTestCase(TestCase):
    def setUp(self):
        # Crear usuarios para cada rol
        self.director = User.objects.create_user(
            username="director",
            password="password123",
            role=User.Role.DIRECTOR,
            first_name="Director",
            last_name="Test"
        )
        self.coordinator = User.objects.create_user(
            username="coordinator",
            password="password123",
            role=User.Role.COORDINATOR,
            first_name="Coordinator",
            last_name="Test"
        )
        self.professor = User.objects.create_user(
            username="professor",
            password="password123",
            role=User.Role.PROFESSOR
        )
        self.student_user = User.objects.create_user(
            username="student1",
            password="password123",
            role=User.Role.STUDENT,
            first_name="Student",
            last_name="One"
        )
        
        # Crear datos de la academia
        self.estudiante = Estudiante.objects.create(
            user=self.student_user,
            matricula="MAT-001"
        )
        
        self.asignatura = Asignatura.objects.create(
            nombre="Matematicas",
            codigo="MAT101"
        )
        
        self.salon = Salon.objects.create(
            nombre="Aula 101",
            capacidad=30
        )
        
        self.seccion = Seccion.objects.create(
            codigo="A"
        )
        
        self.asignacion = AsignacionClase.objects.create(
            profesor=self.professor,
            asignatura=self.asignatura,
            seccion=self.seccion,
            salon=self.salon,
            horario_inicio="08:00:00",
            horario_fin="10:00:00"
        )
        
        self.sesion = SesionClase.objects.create(
            asignacion_clase=self.asignacion,
            estado=SesionClase.Estado.EN_CURSO
        )
        
        self.registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=self.estudiante,
            es_fraude=True
        )

    def test_panel_enrolamiento_permissions(self):
        url = reverse('academic:panel-enrolamiento')
        
        # Anónimo -> Redirecciona
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        
        # Estudiante -> 403 Forbidden
        self.client.login(username="student1", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)
        self.client.logout()
        
        # Profesor -> 403 Forbidden
        self.client.login(username="professor", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 403)
        self.client.logout()
        
        # Coordinador -> 200 OK
        self.client.login(username="coordinator", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.client.logout()
        
        # Director -> 200 OK
        self.client.login(username="director", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.client.logout()

    def test_panel_enrolamiento_query(self):
        self.client.login(username="coordinator", password="password123")
        url = reverse('academic:panel-enrolamiento')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
        estudiantes = response.context['estudiantes']
        self.assertEqual(len(estudiantes), 1)
        # El estudiante no tiene rostro enrolado
        self.assertEqual(estudiantes[0].tiene_rostro, 0)
        
        # Crear face embedding
        FaceEmbedding.objects.create(user=self.student_user, embedding=[0.1, 0.2, 0.3])
        
        response = self.client.get(url)
        estudiantes = response.context['estudiantes']
        self.assertEqual(estudiantes[0].tiene_rostro, 1)

    def test_enrolar_rostro_api(self):
        url = reverse('academic:api-enrolar-rostro')
        
        # Probar POST con coordinador
        self.client.login(username="coordinator", password="password123")
        data = {
            'user_id': self.student_user.id,
            'imagen': 'base64_string_placeholder'
        }
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertEqual(resp_json['status'], 'success')
        
        # Verificar que se creó
        self.assertTrue(FaceEmbedding.objects.filter(user=self.student_user).exists())

    def test_visor_academico(self):
        url = reverse('academic:visor-academico')
        self.client.login(username="coordinator", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_salones'], 1)
        self.assertEqual(response.context['total_estudiantes'], 1)
        self.assertEqual(len(response.context['salones']), 1)

    def test_auditoria_seguridad(self):
        url = reverse('auditoria-seguridad')
        
        # Director -> 200 OK
        self.client.login(username="director", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['alertas_seguridad']), 1)
        self.client.logout()

        # Coordinador -> 200 OK
        self.client.login(username="coordinator", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['alertas_seguridad']), 1)
        self.client.logout()

    def test_reportes_asistencia(self):
        url = reverse('reportes-asistencia')
        
        # Anonimo -> Redirecciona
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        
        # Coordinador -> 200 OK
        self.client.login(username="coordinator", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['reportes']), 1)
        
        # Probar filtros
        response = self.client.get(url, {'q': 'MAT-001'})
        self.assertEqual(len(response.context['reportes']), 1)
        
        response = self.client.get(url, {'q': 'NON-EXISTENT'})
        self.assertEqual(len(response.context['reportes']), 0)
        self.client.logout()


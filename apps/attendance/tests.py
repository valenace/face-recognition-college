import tempfile
from unittest.mock import patch

import numpy as np
from django.test import TestCase, override_settings
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.academic.models import Estudiante, AsignacionClase, Asignatura, Salon, Seccion
from apps.attendance.models import SesionClase, RegistroAsistencia
from core.registro_asistencia import (
    acumular_decisiones,
    aplicar_decisiones,
    nuevo_acumulado,
    registrar_lote,
)

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

    def test_sesiones_list_renders(self):
        self.client.login(username="professor_owner", password="password123")
        response = self.client.get(reverse("sesiones"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Fisica")

    def test_sesion_detalle_renders(self):
        self.client.login(username="professor_owner", password="password123")
        url = reverse("sesion_detalle", kwargs={"sesion_id": self.sesion.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Fisica")

    def test_descartar_registro_marca_revisada(self):
        registro = RegistroAsistencia.objects.create(
            sesion=self.sesion,
            estudiante=None,
            tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO,
            es_fraude=False,
        )
        self.client.login(username="professor_owner", password="password123")
        url = reverse(
            "descartar_registro",
            kwargs={"sesion_id": self.sesion.id, "registro_id": registro.id},
        )
        response = self.client.post(url)
        self.assertRedirects(
            response,
            reverse("sesion_detalle", kwargs={"sesion_id": self.sesion.id}),
        )
        registro.refresh_from_db()
        self.assertTrue(registro.alerta_revisada)

    def test_directorio_muestra_contadores(self):
        self.client.login(username="professor_owner", password="password123")
        response = self.client.get(reverse("directorio"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["con_biometria"], 0)
        self.assertEqual(response.context["sin_biometria"], 1)

    def test_api_enrolar_requiere_autenticacion(self):
        response = self.client.post("/api/enrolar/", {})
        self.assertEqual(response.status_code, 403)

    def test_api_enrolar_rechaza_profesor(self):
        self.client.login(username="professor_owner", password="password123")
        response = self.client.post("/api/enrolar/", {})
        self.assertEqual(response.status_code, 403)

    @patch("apps.attendance.api._get_motor")
    def test_api_enrolar_crea_estudiante_y_embedding(self, mock_get_motor):
        from apps.attendance.models import FaceEmbedding

        UserModel = get_user_model()
        UserModel.objects.create_user(
            username="developer_test",
            password="password123",
            role=UserModel.Role.DEVELOPER,
        )
        self.client.login(username="developer_test", password="password123")

        mock_get_motor.return_value.enrolar_usuario_headless.return_value = {
            "exito": True,
            "muestras": 15,
            "mensaje": "Registro exitoso",
            "embedding": [0.0] * 512,
        }

        video = SimpleUploadedFile("test.mp4", b"fake-video", content_type="video/mp4")
        response = self.client.post(
            "/api/enrolar/",
            {
                "carnet": "20230001",
                "nombre_completo": "Juan Pérez",
                "video": video,
            },
        )
        self.assertEqual(response.status_code, 201)

        usuario = UserModel.objects.get(username="20230001")
        self.assertEqual(usuario.first_name, "Juan")
        self.assertEqual(usuario.last_name, "Pérez")
        self.assertTrue(
            Estudiante.objects.filter(user=usuario, matricula="20230001").exists()
        )
        self.assertTrue(FaceEmbedding.objects.filter(user=usuario).exists())

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


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class FlujoAsistenciaTests(TestCase):
    """El batch y el en vivo deben escribir la asistencia con las mismas reglas."""

    @classmethod
    def setUpTestData(cls):
        cls.profesor = User.objects.create_user(
            username="prof_test", password="x", role=User.Role.PROFESSOR
        )
        cls.usuario_alumno = User.objects.create_user(
            username="unc_alumno", password="x", role=User.Role.STUDENT
        )
        cls.estudiante = Estudiante.objects.create(
            user=cls.usuario_alumno, matricula="T-001"
        )
        asignatura = Asignatura.objects.create(nombre="Prueba", codigo="T-A")
        salon = Salon.objects.create(nombre="Salón", capacidad=10)
        seccion = Seccion.objects.create(codigo="T-S")
        cls.asignacion = AsignacionClase.objects.create(
            profesor=cls.profesor,
            asignatura=asignatura,
            seccion=seccion,
            salon=salon,
            dia_semana=AsignacionClase.DiaSemana.LUNES,
            horario_inicio="08:00",
            horario_fin="10:00",
        )
        cls.sesion = SesionClase.objects.create(asignacion_clase=cls.asignacion)

    def _frame(self):
        return np.zeros((240, 240, 3), dtype=np.uint8)

    def _decision(self, estado, nombre, sim=0.40, track=1, bbox=(20, 20, 120, 120)):
        return {
            "estado": estado,
            "nombre": nombre,
            "similitud": sim,
            "track_id": track,
            "bbox": bbox,
        }

    def test_estudiante_confirmado_registra_asistencia(self):
        resumen = aplicar_decisiones(
            [self._decision("CONFIRMADO", "alumno")],
            self._frame(), self.sesion, {"alumno": self.estudiante},
        )
        self.assertEqual(resumen["confirmados"], ["alumno"])

        registro = RegistroAsistencia.objects.get(
            sesion=self.sesion, estudiante=self.estudiante
        )
        self.assertEqual(
            registro.tipo_evento, RegistroAsistencia.TipoEvento.ASISTENCIA_NORMAL
        )
        self.assertFalse(registro.es_fraude)
        self.assertIsNotNone(registro.hora_entrada)
        self.assertIsNotNone(registro.ultima_vez_visto)

    def test_desconocido_es_intruso_no_fraude(self):
        aplicar_decisiones(
            [self._decision("DESCONOCIDO", "Desconocido", sim=0.10, track=7)],
            self._frame(), self.sesion, {},
        )
        registro = RegistroAsistencia.objects.get(
            sesion=self.sesion, tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO
        )
        self.assertIsNone(registro.estudiante)
        self.assertFalse(registro.es_fraude)
        self.assertIn("Pista 7", registro.notas_auditoria)

    def test_fraude_marca_es_fraude(self):
        aplicar_decisiones(
            [self._decision("FRAUDE", "alumno", sim=0.5, track=3)],
            self._frame(), self.sesion, {"alumno": self.estudiante},
        )
        registro = RegistroAsistencia.objects.get(
            sesion=self.sesion, tipo_evento=RegistroAsistencia.TipoEvento.SPOOFING
        )
        self.assertTrue(registro.es_fraude)
        self.assertIsNone(registro.estudiante)

    def test_lote_exige_evidencia_en_tres_imagenes(self):
        acumulado = nuevo_acumulado()
        for indice in range(2):
            acumular_decisiones(
                acumulado, [self._decision("CONFIRMADO", "alumno")],
                self._frame(), indice,
            )
        registrar_lote(
            acumulado, self.sesion, {"alumno": self.estudiante}, minimo_imagenes=3
        )
        self.assertFalse(
            RegistroAsistencia.objects.filter(sesion=self.sesion).exists()
        )

        acumular_decisiones(
            acumulado, [self._decision("CONFIRMADO", "alumno")], self._frame(), 2
        )
        registrar_lote(
            acumulado, self.sesion, {"alumno": self.estudiante}, minimo_imagenes=3
        )
        self.assertTrue(
            RegistroAsistencia.objects.filter(
                sesion=self.sesion, estudiante=self.estudiante
            ).exists()
        )

    def test_lote_agrupa_intrusos_en_una_fila(self):
        acumulado = nuevo_acumulado()
        for indice in range(5):
            acumular_decisiones(
                acumulado,
                [self._decision("DESCONOCIDO", "Desconocido", sim=0.1, track=indice)],
                self._frame(), indice,
            )
        registrar_lote(acumulado, self.sesion, {}, minimo_imagenes=3)
        self.assertEqual(
            RegistroAsistencia.objects.filter(
                sesion=self.sesion, tipo_evento=RegistroAsistencia.TipoEvento.INTRUSO
            ).count(),
            1,
        )

    def test_en_vivo_stream_y_polling_conviven_en_paralelo(self):
        from core.servicio_en_vivo import servicio_en_vivo

        # 1. Sin autenticar -> 403 en ambos endpoints
        self.assertEqual(self.client.get("/api/en-vivo/frame/").status_code, 403)
        self.assertEqual(self.client.get("/api/en-vivo/stream/").status_code, 403)

        # 2. Estudiante no autorizado -> 403 en ambos endpoints
        self.client.login(username="unc_alumno", password="x")
        self.assertEqual(self.client.get("/api/en-vivo/frame/").status_code, 403)
        self.assertEqual(self.client.get("/api/en-vivo/stream/").status_code, 403)

        # 3. Profesor autenticado: verificar ambos modos con un frame de prueba
        self.client.login(username="prof_test", password="x")
        with servicio_en_vivo._cond:
            servicio_en_vivo._jpeg = b"\xff\xd8\xff\xe0FAKE_JPEG\xff\xd9"
            servicio_en_vivo._estado["frames"] = 1
            servicio_en_vivo._activo = False

        # Modo original (Polling HTTP)
        resp_frame = self.client.get("/api/en-vivo/frame/")
        self.assertEqual(resp_frame.status_code, 200)
        self.assertEqual(resp_frame["Content-Type"], "image/jpeg")
        self.assertEqual(resp_frame.content, b"\xff\xd8\xff\xe0FAKE_JPEG\xff\xd9")

        # Modo nuevo en paralelo (Streaming MJPEG nativo)
        resp_stream = self.client.get("/api/en-vivo/stream/")
        self.assertEqual(resp_stream.status_code, 200)
        self.assertTrue(resp_stream.streaming)
        self.assertEqual(
            resp_stream["Content-Type"],
            "multipart/x-mixed-replace; boundary=frame",
        )
        chunks = list(resp_stream.streaming_content)
        self.assertEqual(len(chunks), 1)
        self.assertIn(b"--frame\r\nContent-Type: image/jpeg\r\n", chunks[0])
        self.assertIn(b"\xff\xd8\xff\xe0FAKE_JPEG\xff\xd9", chunks[0])

        # Limpiar estado
        with servicio_en_vivo._cond:
            servicio_en_vivo._jpeg = None
            servicio_en_vivo._estado["frames"] = 0


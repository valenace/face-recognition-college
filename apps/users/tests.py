from django.test import TestCase
from django.urls import reverse
from django.contrib.auth import get_user_model
import json

User = get_user_model()

class UserManagementTests(TestCase):
    def setUp(self):
        self.developer = User.objects.create_user(
            username="dev_test",
            password="password123",
            role=User.Role.DEVELOPER,
            first_name="Dev",
            last_name="Test"
        )
        self.professor = User.objects.create_user(
            username="prof_test",
            password="password123",
            role=User.Role.PROFESSOR,
            first_name="Pedro",
            last_name="Ramirez",
            email="pedro@college.edu"
        )
        self.coordinator = User.objects.create_user(
            username="coord_test",
            password="password123",
            role=User.Role.COORDINATOR,
            first_name="Coord",
            last_name="Test"
        )

    def test_gestion_profesores_permissions(self):
        url = reverse("gestion-profesores")
        
        # Anonimo -> Redirecciona a Login
        response = self.client.get(url)
        self.assertEqual(response.status_code, 302)
        
        # Coordinador -> Redirecciona a Dashboard
        self.client.login(username="coord_test", password="password123")
        response = self.client.get(url)
        self.assertRedirects(response, reverse('dashboard'), fetch_redirect_response=False)
        self.client.logout()

        # Profesor -> Redirecciona a Dashboard
        self.client.login(username="prof_test", password="password123")
        response = self.client.get(url)
        self.assertRedirects(response, reverse('dashboard'), fetch_redirect_response=False)
        self.client.logout()

        # Developer -> 200 OK
        self.client.login(username="dev_test", password="password123")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("profesores", response.context)
        self.assertEqual(len(response.context["profesores"]), 1)
        self.client.logout()

    def test_editar_correo_profesor(self):
        url = reverse("api-editar-correo-profesor")
        
        # Iniciar sesión como Developer
        self.client.login(username="dev_test", password="password123")
        
        data = {
            "professor_id": self.professor.id,
            "email": "nuevo_correo@college.edu"
        }
        
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertEqual(resp_json["status"], "success")
        
        # Verificar cambio en DB
        self.professor.refresh_from_db()
        self.assertEqual(self.professor.email, "nuevo_correo@college.edu")

    def test_restablecer_credenciales(self):
        url = reverse("api-restablecer-credenciales-profesor")
        
        # Iniciar sesión como Developer
        self.client.login(username="dev_test", password="password123")
        
        data = {
            "professor_id": self.professor.id
        }
        
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertEqual(resp_json["status"], "success")
        self.assertIn("temp_password", resp_json)
        
        # Intentar iniciar sesión con la nueva clave temporal
        temp_pwd = resp_json["temp_password"]
        self.client.logout()
        
        login_success = self.client.login(username="prof_test", password=temp_pwd)
        self.assertTrue(login_success)

    def test_perfil_usuario_views(self):
        url = reverse("perfil")
        
        # Iniciar sesión como Profesor
        self.client.login(username="prof_test", password="password123")
        
        # 1. Test GET request
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("profile_form", response.context)
        self.assertIn("password_form", response.context)

        # 2. Test POST update profile data
        profile_data = {
            "action": "update_profile",
            "first_name": "Pedro Modificado",
            "last_name": "Ramirez Modificado",
            "email": "pedro_new@college.edu"
        }
        response = self.client.post(url, data=profile_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("profile_success", response.context)
        
        # Verificar DB
        self.professor.refresh_from_db()
        self.assertEqual(self.professor.first_name, "Pedro Modificado")
        self.assertEqual(self.professor.last_name, "Ramirez Modificado")
        self.assertEqual(self.professor.email, "pedro_new@college.edu")

        # 3. Test POST change password
        pwd_data = {
            "action": "change_password",
            "old_password": "password123",
            "new_password1": "newSecurePass123!",
            "new_password2": "newSecurePass123!"
        }
        response = self.client.post(url, data=pwd_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("password_success", response.context)
        
        # Intentar login con la nueva contraseña
        self.client.logout()
        login_success = self.client.login(username="prof_test", password="newSecurePass123!")
        self.assertTrue(login_success)

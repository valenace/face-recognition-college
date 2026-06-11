from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model

class AdministrativeModelBackend(ModelBackend):
    def user_can_authenticate(self, user):
        # Los estudiantes no tienen permiso para iniciar sesión en la plataforma
        if user.role == "STUDENT":
            return False
        return super().user_can_authenticate(user)

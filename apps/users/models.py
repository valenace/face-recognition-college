from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    class Role(models.TextChoices):
        ADMIN = "ADMIN", "Administrador"
        PROFESSOR = "PROFESSOR", "Profesor"
        STUDENT = "STUDENT", "Estudiante"

    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.STUDENT
    )
    avatar = models.ImageField(upload_to="avatars/", null=True, blank=True)
    
    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

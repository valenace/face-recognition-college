from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    class Role(models.TextChoices):
        DEVELOPER = "DEVELOPER", "Desarrollador"
        DIRECTOR = "DIRECTOR", "Directivo"
        COORDINATOR = "COORDINATOR", "Coordinador"
        PROFESSOR = "PROFESSOR", "Profesor"
        STUDENT = "STUDENT", "Estudiante"

    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.STUDENT
    )
    avatar = models.ImageField(upload_to="avatars/", null=True, blank=True)

    @property
    def is_developer(self):
        return self.role == self.Role.DEVELOPER

    @property
    def is_director(self):
        return self.role == self.Role.DIRECTOR

    @property
    def is_coordinator(self):
        return self.role == self.Role.COORDINATOR

    @property
    def is_professor(self):
        return self.role == self.Role.PROFESSOR

    @property
    def is_student(self):
        return self.role == self.Role.STUDENT

    def save(self, *args, **kwargs):
        # Configurar flags de staff y superuser automáticamente según el rol
        if self.role == self.Role.DEVELOPER:
            self.is_superuser = True
            self.is_staff = True
        elif self.role in [self.Role.DIRECTOR, self.Role.COORDINATOR]:
            self.is_superuser = False
            self.is_staff = True
        else:
            self.is_superuser = False
            self.is_staff = False
            
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

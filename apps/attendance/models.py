from django.db import models
from django.conf import settings
from apps.academic.models import ClassSession

class FaceEmbedding(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="face_data")
    embedding = models.JSONField()  # Store the list of floats
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for {self.user.username}"

class AttendanceRecord(models.Model):
    student = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    session = models.ForeignKey(ClassSession, on_delete=models.CASCADE, related_name="attendance_logs")
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField()
    is_verified = models.BooleanField(default=True)

    class Meta:
        unique_together = ("student", "session")

    def __str__(self):
        return f"{self.student.username} - {self.session} at {self.timestamp}"

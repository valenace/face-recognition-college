from django.db import models
from django.conf import settings

class Subject(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=20, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

class Classroom(models.Model):
    name = models.CharField(max_length=50)
    capacity = models.IntegerField()
    location = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.name

class Section(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="sections")
    code = models.CharField(max_length=20)
    professor = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.SET_NULL, 
        null=True, 
        limit_choices_to={"role": "PROFESSOR"},
        related_name="teaching_sections"
    )
    students = models.ManyToManyField(
        settings.AUTH_USER_MODEL, 
        limit_choices_to={"role": "STUDENT"},
        related_name="enrolled_sections"
    )

    def __str__(self):
        return f"{self.subject.name} - {self.code}"

class ClassSession(models.Model):
    section = models.ForeignKey(Section, on_delete=models.CASCADE, related_name="sessions")
    classroom = models.ForeignKey(Classroom, on_delete=models.SET_NULL, null=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    
    def __str__(self):
        return f"{self.section} ({self.start_time.strftime('%Y-%m-%d %H:%M')})"

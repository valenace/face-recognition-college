from django.shortcuts import render
from django.contrib.auth import get_user_model

User = get_user_model()


def registration(request):
    return render(request, "attendance/registration.html")


def live_attendance(request):
    return render(request, "attendance/live.html")


def directorio(request):
    """Vista del directorio de estudiantes con enrolamiento biométrico."""
    estudiantes = (
        User.objects.filter(role=User.Role.STUDENT)
        .select_related("face_data")
        .order_by("-date_joined")
    )
    return render(request, "attendance/directorio.html", {
        "estudiantes": estudiantes,
    })

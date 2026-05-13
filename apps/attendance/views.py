from django.shortcuts import render

def registration(request):
    return render(request, "attendance/registration.html")

def live_attendance(request):
    return render(request, "attendance/live.html")

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    if request.user.is_professor:
        return redirect("academic:dashboard-profesor")
    elif request.user.is_student:
        return redirect("facial_registration")
    return render(request, "dashboard.html")

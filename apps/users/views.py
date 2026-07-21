from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    if request.user.is_professor:
        return redirect("academic:panel-profesor")
    return render(request, "dashboard.html")


import json
import secrets
import string
from django.views.generic import ListView, TemplateView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.http import JsonResponse
from django.core.mail import send_mail
from apps.users.permissions import RoleRequiredMixin
from apps.users.models import User
from apps.users.forms import UserProfileForm

class GestionProfesoresView(RoleRequiredMixin, ListView):
    model = User
    template_name = "developer/gestion_profesores.html"
    context_object_name = "profesores"
    allowed_roles = ["DEVELOPER"]

    def get_queryset(self):
        return User.objects.filter(role=User.Role.PROFESSOR).select_related("face_data").order_by("first_name")


class EditarCorreoProfesorAPI(RoleRequiredMixin, View):
    allowed_roles = ["DEVELOPER"]

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            professor_id = data.get("professor_id")
            nuevo_correo = data.get("email")

            if not nuevo_correo:
                return JsonResponse({"status": "error", "message": "El correo es requerido."}, status=400)

            profesor = User.objects.get(id=professor_id, role=User.Role.PROFESSOR)
            profesor.email = nuevo_correo
            profesor.save()

            return JsonResponse({"status": "success", "message": "Correo electrónico actualizado correctamente."})
        except User.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Profesor no encontrado."}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)


class RestablecerCredencialesAPI(RoleRequiredMixin, View):
    allowed_roles = ["DEVELOPER"]

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            professor_id = data.get("professor_id")

            profesor = User.objects.get(id=professor_id, role=User.Role.PROFESSOR)
            
            # Generar contraseña aleatoria segura de 12 caracteres
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            temp_password = "".join(secrets.choice(alphabet) for i in range(12))

            profesor.set_password(temp_password)
            profesor.save()

            # Enviar correo simulado
            subject = "Restablecimiento de Credenciales - CV College"
            message = (
                f"Hola, {profesor.get_full_name()}.\n\n"
                f"Se han restablecido tus credenciales para la plataforma CV College.\n"
                f"Tus datos de acceso temporales son:\n"
                f"- Usuario: {profesor.username}\n"
                f"- Contraseña temporal: {temp_password}\n\n"
                f"Por favor, inicia sesión y cambia esta contraseña temporal desde tu perfil."
            )
            
            send_mail(
                subject,
                message,
                "soporte@cvcollege.edu",
                [profesor.email],
                fail_silently=False,
            )

            return JsonResponse({
                "status": "success", 
                "message": "Credenciales restablecidas y correo enviado con éxito.",
                "temp_password": temp_password
            })
        except User.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Profesor no encontrado."}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)


class PerfilUsuarioView(LoginRequiredMixin, TemplateView):
    template_name = "registration/perfil.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if "profile_form" not in context:
            context["profile_form"] = UserProfileForm(instance=self.request.user)
        if "password_form" not in context:
            context["password_form"] = PasswordChangeForm(user=self.request.user)
        return context

    def post(self, request, *args, **kwargs):
        action = request.POST.get("action")
        
        if action == "update_profile":
            profile_form = UserProfileForm(request.POST, instance=request.user)
            if profile_form.is_valid():
                profile_form.save()
                return render(request, self.template_name, self.get_context_data(
                    profile_success="Tus datos de perfil han sido actualizados con éxito."
                ))
            else:
                return render(request, self.template_name, self.get_context_data(
                    profile_form=profile_form
                ))

        elif action == "change_password":
            password_form = PasswordChangeForm(user=request.user, data=request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                return render(request, self.template_name, self.get_context_data(
                    password_success="Tu contraseña ha sido cambiada con éxito."
                ))
            else:
                return render(request, self.template_name, self.get_context_data(
                    password_form=password_form
                ))

        return self.get(request, *args, **kwargs)

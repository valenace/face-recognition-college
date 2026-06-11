from django.contrib.auth.forms import AuthenticationForm
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model

User = get_user_model()

class AdministrativeAuthenticationForm(AuthenticationForm):
    def clean(self):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")

        if username and password:
            try:
                user = User.objects.get(username=username)
                if user.role == User.Role.STUDENT:
                    raise ValidationError(
                        "Los estudiantes no tienen acceso de inicio de sesión a esta plataforma.",
                        code="student_login_not_allowed",
                    )
            except User.DoesNotExist:
                pass

        return super().clean()


from django import forms

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["first_name", "last_name", "email"]
        widgets = {
            "first_name": forms.TextInput(attrs={
                "class": "w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-850 text-slate-800 dark:text-slate-100 focus:ring-2 focus:ring-brand-blue focus:border-transparent outline-none transition-all text-sm",
                "placeholder": "Nombre"
            }),
            "last_name": forms.TextInput(attrs={
                "class": "w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-850 text-slate-800 dark:text-slate-100 focus:ring-2 focus:ring-brand-blue focus:border-transparent outline-none transition-all text-sm",
                "placeholder": "Apellido"
            }),
            "email": forms.EmailInput(attrs={
                "class": "w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-zinc-700 bg-slate-50 dark:bg-zinc-850 text-slate-800 dark:text-slate-100 focus:ring-2 focus:ring-brand-blue focus:border-transparent outline-none transition-all text-sm",
                "placeholder": "Correo electrónico"
            }),
        }

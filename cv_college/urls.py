from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from apps.users.forms import AdministrativeAuthenticationForm
from apps.users import views as user_views

urlpatterns = [
    path("admin/", admin.site.urls),
    # La web reducida entra por el directorio (registro y listado de rostros).
    path("", RedirectView.as_view(pattern_name="directorio"), name="dashboard"),

    # ── Attendance & Biometrics ────────────────────────────────
    path("", include("apps.attendance.urls")),

    # Secciones fuera del menú: siguen disponibles por URL para reactivarlas luego.
    path("academic/", include("apps.academic.urls")),
    # urls de autenticación
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="registration/login.html",
            authentication_form=AdministrativeAuthenticationForm,
            redirect_authenticated_user=True
        ),
        name="login"
    ),
    path("logout/", auth_views.LogoutView.as_view(next_page="login"), name="logout"),

    # Gestión de Profesores y Perfiles
    path("gestion/profesores/", user_views.GestionProfesoresView.as_view(), name="gestion-profesores"),
    path("api/profesores/editar-correo/", user_views.EditarCorreoProfesorAPI.as_view(), name="api-editar-correo-profesor"),
    path("api/profesores/restablecer-credenciales/", user_views.RestablecerCredencialesAPI.as_view(), name="api-restablecer-credenciales-profesor"),
    path("perfil/", user_views.PerfilUsuarioView.as_view(), name="perfil"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

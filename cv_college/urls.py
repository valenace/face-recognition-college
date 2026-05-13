from django.contrib import admin
from django.urls import path, include
from apps.users.views import dashboard
from apps.attendance.views import registration, live_attendance
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", dashboard, name="dashboard"),
    path("facial-registration/", registration, name="facial_registration"),
    path("live-attendance/", live_attendance, name="live_attendance"),
    
    # academic placeholders
    # path("academic/", include("apps.academic.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

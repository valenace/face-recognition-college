from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType

class Command(BaseCommand):
    help = "Crea los grupos por defecto y les asigna sus respectivos permisos para el sistema RBAC"

    def handle(self, *args, **options):
        # definición de roles y grupos
        groups_config = {
            "Directivos": {
                "apps": ["academic", "attendance", "users"],
                "actions": ["view", "add", "change", "delete"],
            },
            "Coordinadores": {
                "apps": ["academic"],
                "actions": ["view", "add", "change", "delete"],
                "extra": [
                    ("attendance", "view_registroasistencia"),
                    ("attendance", "view_faceembedding"),
                ]
            },
            "Profesores": {
                "extra": [
                    # permisos sobre asignaturas e infraestructura
                    ("academic", "view_asignatura"),
                    ("academic", "view_salon"),
                    ("academic", "view_seccion"),
                    ("academic", "view_asignacionclase"),
                    # permisos sobre control de asistencia
                    ("attendance", "view_registroasistencia"),
                    ("attendance", "add_registroasistencia"),
                    ("attendance", "change_registroasistencia"),
                    ("attendance", "view_faceembedding"),
                ]
            }
        }

        self.stdout.write(self.style.WARNING("Iniciando creación de grupos y asignación de permisos..."))

        for group_name, config in groups_config.items():
            group, created = Group.objects.get_or_create(name=group_name)
            if created:
                self.stdout.write(self.style.SUCCESS(f"Grupo '{group_name}' creado exitosamente."))
            else:
                self.stdout.write(self.style.WARNING(f"El grupo '{group_name}' ya existe. Actualizando permisos..."))

            # limpiar permisos del grupo antes de reasignar
            group.permissions.clear()

            permissions_to_add = []

            # 1. procesar permisos globales por aplicación (directivos y coordinadores)
            if "apps" in config:
                for app in config["apps"]:
                    for action in config["actions"]:
                        codename_prefix = f"{action}_"
                        # buscar todos los permisos que correspondan a la app y comiencen con la acción
                        perms = Permission.objects.filter(
                            content_type__app_label=app,
                            codename__startswith=codename_prefix
                        )
                        permissions_to_add.extend(perms)

            # 2. procesar permisos extra específicos
            if "extra" in config:
                for app_label, codename in config["extra"]:
                    try:
                        perm = Permission.objects.get(
                            content_type__app_label=app_label,
                            codename=codename
                        )
                        permissions_to_add.append(perm)
                    except Permission.DoesNotExist:
                        self.stdout.write(
                            self.style.ERROR(f"Permiso no encontrado: {app_label}.{codename}")
                        )

            # guardar permisos en el grupo
            group.permissions.add(*permissions_to_add)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Asignados {len(permissions_to_add)} permisos al grupo '{group_name}'."
                )
            )

        self.stdout.write(self.style.SUCCESS("Configuración de roles y permisos (RBAC) completada."))

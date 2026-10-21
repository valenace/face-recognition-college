"""
Importa los centroides de `data/database_embeddings.pkl` a la base vectorial
(PostgreSQL + pgvector), creando el usuario y el perfil de estudiante de cada uno.

Uso:
    python manage.py importar_centroides
    python manage.py importar_centroides --dry-run
    python manage.py importar_centroides --pickle=data/otro.pkl --n-muestras=50
"""

import pickle
from pathlib import Path

import numpy as np
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.academic.models import Estudiante
from apps.attendance.models import FaceEmbedding

User = get_user_model()

# clave en el pickle -> (username, nombre, apellido, matrícula)
ESTUDIANTES = {
    "alexa": ("unc_alexa", "Alexa", "Ramírez", "UNC-2026-001"),
    "bob": ("unc_bob", "Roberto", "Díaz", "UNC-2026-002"),
    "carlos": ("unc_carlos", "Carlos", "Mendoza", "UNC-2026-003"),
    "dani": ("unc_dani", "Daniela", "Torres", "UNC-2026-004"),
    "fran": ("unc_fran", "Francisco", "Rojas", "UNC-2026-005"),
    "jesus": ("unc_jesus", "Jesús", "Herrera", "UNC-2026-006"),
    "jose": ("unc_jose", "José", "Cabrera", "UNC-2026-007"),
    "kate": ("unc_kate", "Katerine", "Vega", "UNC-2026-008"),
    "kevin": ("unc_kevin", "Kevin", "Navarro", "UNC-2026-009"),
}


class Command(BaseCommand):
    help = "Importa los centroides del pickle a FaceEmbedding (pgvector) creando usuarios y estudiantes"

    def add_arguments(self, parser):
        parser.add_argument(
            "--pickle",
            default=str(Path(settings.BASE_DIR) / "data/database_embeddings.pkl"),
            help="Ruta del pickle con los centroides",
        )
        parser.add_argument(
            "--n-muestras",
            type=int,
            default=50,
            help="Número de muestras usadas para calcular cada centroide",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Muestra lo que haría sin escribir en la base de datos",
        )

    def handle(self, *args, **options):
        ruta = Path(options["pickle"])
        if not ruta.exists():
            raise CommandError(f"No existe el pickle: {ruta}")

        with open(ruta, "rb") as f:
            db = pickle.load(f)

        if not db:
            raise CommandError(f"El pickle {ruta} está vacío")

        desconocidos = sorted(set(db) - set(ESTUDIANTES))
        if desconocidos:
            raise CommandError(
                f"Hay centroides sin mapeo a usuario: {desconocidos}. "
                "Agrégalos a ESTUDIANTES en este comando."
            )

        self.stdout.write(f"Centroides en {ruta.name}: {len(db)}")
        creados = actualizados = 0

        for clave, vector in db.items():
            username, nombre, apellido, matricula = ESTUDIANTES[clave]
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vec.shape[0] != 512:
                raise CommandError(f"{clave}: dimensión {vec.shape[0]}, se esperaba 512")
            norma = float(np.linalg.norm(vec))
            if not np.isclose(norma, 1.0, atol=1e-3):
                self.stdout.write(
                    self.style.WARNING(f"  {clave}: norma {norma:.4f}, se normaliza")
                )
                vec = vec / norma

            if options["dry_run"]:
                self.stdout.write(
                    f"  [dry-run] {clave:<8} -> {username:<12} {nombre} {apellido} "
                    f"({matricula}) | dim {vec.shape[0]} | norma {np.linalg.norm(vec):.3f}"
                )
                continue

            usuario, creado = User.objects.get_or_create(
                username=username,
                defaults={
                    "first_name": nombre,
                    "last_name": apellido,
                    "email": f"{username}@unc.edu",
                    "role": User.Role.STUDENT,
                },
            )
            if creado:
                creados += 1
            else:
                usuario.first_name = nombre
                usuario.last_name = apellido
                usuario.role = User.Role.STUDENT
                usuario.save(update_fields=["first_name", "last_name", "role"])
                actualizados += 1

            estudiante, _ = Estudiante.objects.get_or_create(
                user=usuario,
                defaults={"matricula": matricula},
            )
            if estudiante.matricula != matricula:
                estudiante.matricula = matricula
                estudiante.save(update_fields=["matricula"])

            FaceEmbedding.objects.update_or_create(
                user=usuario,
                defaults={
                    "embedding": vec.tolist(),
                    "modelo": "arcface",
                    "n_muestras": options["n_muestras"],
                    "fuente": FaceEmbedding.Fuente.IMPORTADO,
                    "fecha": timezone.now(),
                },
            )
            self.stdout.write(
                f"  ✓ {clave:<8} -> {username:<12} {nombre} {apellido} ({matricula})"
            )

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("\nDry-run: no se escribió nada."))
            return

        total = FaceEmbedding.objects.count()
        self.stdout.write(
            self.style.SUCCESS(
                f"\nUsuarios nuevos: {creados} | actualizados: {actualizados}\n"
                f"Embeddings en la base: {total}"
            )
        )

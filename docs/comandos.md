# Comandos

Ruta base del proyecto:

```bash
cd /home/valentina2/Desktop/Folds/VisionProjects/face-recognition-college
```

---

## 1. Ver el motor en ventana (rápido, sin base de datos)

```bash
venv/bin/python scripts/asistencia_unificada_optimizada.py --fuente data/unc/salon/VID_20260907_102851.mp4
```

Abre la ventana "Control de Asistencia" con cajas, nombre y similitud.
Cerrar con `q`. **No escribe** en la base; solo genera `data/asistencia.csv`.

> Video 3 (el que tiene los 21 no registrados):
> `--fuente data/unc/salon/video5096333890499381162.mp4`

---

## 2. Motor escribiendo en la base (sesión de demo)

```bash
venv/bin/python scripts/asistencia_unificada_optimizada.py --crear-sesion --fuente data/unc/salon/VID_20260907_102851.mp4
```

Imprime `[INFO] Sesión de demo: id=NN`. Escribe en `RegistroAsistencia`.
Para borrar todo al terminar, añade `--limpiar`:

```bash
venv/bin/python scripts/asistencia_unificada_optimizada.py --crear-sesion --limpiar --fuente data/unc/salon/VID_20260907_102851.mp4
```

Con una sesión ya abierta, se reutiliza el ID:

```bash
venv/bin/python scripts/asistencia_unificada_optimizada.py --sesion 12 --fuente 0
```

---

## 3. Detalle técnico del motor (`ver_flujo`)

```bash
venv/bin/python manage.py ver_flujo --video data/unc/salon/VID_20260907_102851.mp4 --desde 36 --hasta 107 --detalle
```

Muestra por frame: `top1`, `top2`, similitud, margen y la decisión.
Añade `--mostrar` para verlo en ventana:

```bash
venv/bin/python manage.py ver_flujo --video data/unc/salon/VID_20260907_102851.mp4 --desde 36 --hasta 60 --mostrar
```

---

## 4. Ver la asistencia que quedó escrita

```bash
venv/bin/python manage.py shell -c "
from apps.attendance.models import RegistroAsistencia
for r in RegistroAsistencia.objects.select_related('sesion__asignacion_clase__asignatura','estudiante__user').order_by('-id')[:15]:
    quien = r.estudiante.user.get_full_name() if r.estudiante else '(sin nombre)'
    print(f'{r.tipo_evento:<17} {quien:<22} sim={r.similitud_ia} entrada={r.hora_entrada}')
"
```

---

## 5. Ver la web (anotado en navegador)

```bash
venv/bin/python manage.py runserver
```

Entrar a `http://localhost:8000/login` con `admin` / `admin123`.
La web reducida tiene tres secciones:

- **Directorio** — `http://localhost:8000/directorio/`
- **Sesiones** — `http://localhost:8000/sesiones/`
- **Control de Asistencia** — `http://localhost:8000/en-vivo/`

En `/en-vivo/`: elegir la sesión, escribir la fuente (`0` = cámara, o una ruta de video)
y **Iniciar**. Si no hay sesiones, aparece el botón **Crear sesión de demo**.
En `/sesiones/<id>/` se ve el detalle y se pueden **descartar** las alertas de intruso
que en realidad sean estudiantes registrados.

---

## 6. Tests

```bash
venv/bin/python -m pytest -m unit          # 74 tests, rápido
venv/bin/python manage.py test              # 32 tests Django
venv/bin/python manage.py check
```

---

## 7. Regresión de identidad (el motor no puede cambiar sin esto)

```bash
venv/bin/python experiments/unc/regresion_identidad.py
```

Sale con código 1 si hay fallos nuevos.

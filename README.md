# face-recognition-college

Sistema de asistencia universitaria por reconocimiento facial: aplicación web en
**Django 6.1** con un motor de visión por computadora (**RetinaFace + ArcFace +
anti-spoofing**) y base vectorial **PostgreSQL + pgvector**.

El foco del proyecto es el motor; la web es una interfaz mínima para registrar
rostros, ver el control de asistencia y revisar las sesiones guardadas.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/resumen_proyecto.md`](docs/resumen_proyecto.md) | Qué es, estado actual, reglas y resultados |
| [`docs/motor.md`](docs/motor.md) | El motor explicado archivo por archivo y línea por línea |
| [`docs/motor_2.md`](docs/motor_2.md) | Capítulo 2: preprocesado, intrusos y escritura |
| [`docs/motor_3.md`](docs/motor_3.md) | Capítulo 3: flujos y registro |
| [`docs/web.md`](docs/web.md) | Web completa: rutas, modelos, vistas, API y plantillas |
| [`docs/computo.md`](docs/computo.md) | Hardware, versiones, rendimiento y migración a GPU |
| [`docs/streaming_mjpeg_vs_polling.md`](docs/streaming_mjpeg_vs_polling.md) | MJPEG vs polling: medición y veredicto |
| [`docs/base_datos.md`](docs/base_datos.md) | Esquema, pgvector y consultas |
| [`docs/comandos.md`](docs/comandos.md) | Comandos listos para ejecutar |
| [`docs/analisis/unc/`](docs/analisis/unc/) | Experimentos y resultados con datos |

## Inicio rápido

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt pytest faiss-cpu

venv/bin/python manage.py migrate
venv/bin/python manage.py importar_centroides   # opcional: migra el pickle

venv/bin/python manage.py runserver
```

Abrir `http://localhost:8000/login` con `admin` / `admin123` (cambiar la
contraseña) y usar **Directorio**, **Sesiones** o **Control de Asistencia**.

## Motor desde consola

```bash
# Solo ver (no escribe en la base)
venv/bin/python scripts/asistencia_unificada_optimizada.py \
  --fuente data/unc/salon/VID_20260907_102851.mp4

# Escribir en una sesión existente
venv/bin/python scripts/asistencia_unificada_optimizada.py --sesion 12 --fuente 0
```

## Tests

```bash
venv/bin/python -m pytest -m unit
venv/bin/python manage.py test
```

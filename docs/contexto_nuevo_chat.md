# Contexto para un chat nuevo — CV College

Documento de traspaso: todo lo necesario para retomar el proyecto sin releer la
conversación anterior. Actualizado el 2026-10-22.

---

## 1. Qué es y dónde está

Sistema de asistencia con reconocimiento facial: app web en **Django 6.1** + pipeline
de visión offline (**RetinaFace + ArcFace** vía `uniface`, anti-spoofing incluido), con
**PostgreSQL + pgvector** como base vectorial.

- Ruta: `/home/valentina2/Desktop/Folds/VisionProjects/face-recognition-college`
- Rama: `scripts-tests` (sin commits pendientes de push; hay cambios locales sin commitear)
- `AGENTS.md` en la raíz tiene las reglas de entorno, comandos y acuerdos de trabajo.

---

## 2. Estado actual

| Aspecto | Estado |
|---|---|
| Tests | **76 pytest** (`pytest -m unit`) y **42 Django** (`manage.py test`), todos en verde |
| `manage.py check` | limpio |
| Base de datos | 10 usuarios base + `prof_flujo`, 9 embeddings; las sesiones de demo 97 y 98 están en curso (12 y 14 filas) |
| Cambios sin commitear | Solo `docs/analisis/unc.zip` (respaldo del usuario) |
| Commits | Ninguno hecho por el asistente: **solo se commitea si el usuario lo pide** |

---

## 3. Base de datos

- **PostgreSQL 16.15**, base `cv_college_db`, usuario `cv_college_user`, contraseña
  `cv_college_2026`, `localhost:5432`. Configurable con `CV_DB_*`; los valores por
  defecto son los de desarrollo.
- **pgvector 0.6.0** activo. Tabla clave: `attendance_faceembedding`.
  - `embedding`: `VectorField(512)`, norma 1, índice **HNSW** con `vector_cosine_ops`.
  - Metadatos: `modelo`, `n_muestras`, `fuente` (`VIDEO`/`IMPORTADO`/`WEB`), `fecha`,
    `created_at`, `updated_at`.
  - Un vector por usuario (restricción única en `user_id`).
- **Estudiantes** (apellidos y matrículas provisionales):

| Usuario | Nombre | Matrícula |
|---|---|---|
| `unc_alexa` | Alexa Ramírez | UNC-2026-001 |
| `unc_bob` | Roberto Díaz | UNC-2026-002 |
| `unc_carlos` | Carlos Mendoza | UNC-2026-003 |
| `unc_dani` | Daniela Torres | UNC-2026-004 |
| `unc_fran` | Francisco Rojas | UNC-2026-005 |
| `unc_jesus` | Jesús Herrera | UNC-2026-006 |
| `unc_jose` | José Cabrera | UNC-2026-007 |
| `unc_kate` | Katerine Vega | UNC-2026-008 |
| `unc_kevin` | Kevin Navarro | UNC-2026-009 |

- **Admin**: `admin` / `admin123`, rol `DEVELOPER` (superusuario). Cambiar contraseña.
- Migración que introdujo el tipo vectorial: `attendance.0010` (borra embeddings previos,
  crea la extensión y reemplaza la columna).
- El pickle `data/database_embeddings.pkl` es **solo respaldo histórico**; lo lee
  `manage.py importar_centroides`.
- Documento detallado: **`docs/base_datos.md`** (esquema, índice, consultas SQL, backup).

---

## 4. Motor de reconocimiento

`scripts/asistencia_unificada_optimizada.py` — configuración final:

| Parámetro | Valor |
|---|---|
| Umbral de similitud | **0.30** (calibrado en Fase 4) |
| Votos | 3 consecutivos |
| Ancla | 30 frames (revalidación) |
| Tamaño mínimo de rostro | 20 px |
| Confianza de detección | 0.75 |
| Anti-spoofing | omitido bajo 60 px |
| Búsqueda | FAISS `IndexFlatIP(512)` sobre los centroides, cargados desde pgvector |

Cada decisión (`procesar_frame`) devuelve: `track_id`, `bbox`, `nombre`, `estado`,
`similitud`, `sim_top2`, `nombre2`, `margen`, `confianza_det`, `candidato`, `motivo`,
`revision`.

Reglas añadidas:

- **`_resolver_conflictos()`**: dos pistas simultáneas no pueden compartir identidad.
  Gana la de mayor similitud; la otra pasa a `Desconocido` y queda vetada 90 frames.
- **`_marcar_revision()`**: marca la banda 0.30–0.40, que en pantalla se muestra como
  **`posible <nombre>`** en amarillo (no cambia la decisión ni la asistencia).

---

## 5. Flujos

| Flujo | Qué hace | Dónde |
|---|---|---|
| Registro CLI | Video → 50 fotos alineadas → centroide → pgvector | `scripts/registro_video.py` |
| Registro web | Igual, desde `/directorio/` | `apps/academic/views.py`, `apps/attendance/api.py` |
| Inferencia consola | Video/cámara → motor → `RegistroAsistencia` (con `--sesion`) o CSV (experimentos) | `scripts/asistencia_unificada_optimizada.py` |
| Batch de imágenes | Carpeta → motor calibrado → `RegistroAsistencia` | `core/procesador_batch.py` (`ProcesarDatasetView`) |
| En vivo | Cámara o video → motor → `RegistroAsistencia` + frame anotado | `core/servicio_en_vivo.py`, `/en-vivo/`, `/api/en-vivo/*` |

- Batch: cada imagen es una observación independiente; se exige confirmación en
  **≥3 imágenes** (o todas, si el lote tiene menos).
- En vivo: 3 votos consecutivos sobre el stream. Acepta una cámara (`0`) o una ruta de
  video, para poder probarlo sin cámara.
- Reglas de escritura: desconocido → `INTRUSO` con `es_fraude=False`; anti-spoofing →
  `SPOOFING` con `es_fraude=True`; horas reales (`hora_entrada`, `ultima_vez_visto`).
- **Intrusos (2026-10-23):** `core/intrusos.py::SeguimientoIntrusos` decide cuándo un
  `DESCONOCIDO` es intruso real: mejor score de la pista < 0.18, 15 frames consecutivos
  y sin conflicto de identidad. Si la pista luego mejora, la fila se borra. La nota
  guarda la mejor coincidencia (`dani 0.14`). Ver `docs/analisis/unc/11_intrusos.md`.
- La vista en vivo está restringida a `PROFESSOR`, `COORDINATOR`, `DIRECTOR` y
  `DEVELOPER`.
- Se eliminó el fallback que guardaba un **vector aleatorio** cuando fallaba la IA: ahora
  el registro devuelve 400 y no escribe nada.

**Web reducida (2026-10-22):** el menú muestra solo **Directorio** (`/directorio/`,
registro con video y listado), **Sesiones** (`/sesiones/`, historial y detalle de la
asistencia guardada) y **Control de Asistencia** (`/en-vivo/`). La raíz `/`
redirige al directorio. Las demás páginas académicas siguen accesibles por URL, pero
no aparecen en el menú hasta que se rehabiliten. Si no hay sesiones abiertas,
`/en-vivo/` ofrece el botón **Crear sesión de demo** (usa `core/sesion_demo.py`).

---

## 6. Datos

- **9 estudiantes registrados**, cada uno con 50 fotos en
  `data/dataset_cv/frames/<persona>/` y un centroide en pgvector.
- Videos de registro: `data/unc/stds/<persona>/*.mp4` (uno por persona, 1080×1920).
- Videos de salón: `VID_20260907_102851.mp4` (video 1, etiquetado, 1920×1080),
  `VID_20260907_105622.mp4` (video 2), `video5096333890499381162.mp4` (video 3, 124 s,
  del que existen 11 partes `lv_0_2026*.mp4`).
- Video 3: **8 registrados presentes** (falta `jose`) y **21 estudiantes no registrados**.
- `data/` está gitignoreado.

---

## 7. Resultados clave

| Medición | Resultado |
|---|---|
| Matriz de similitudes entre centroides | máximo **0.297** (`alexa`–`dani`) |
| Video 1: cobertura | 9/9 |
| Video 1: confirmación en ventana ±0.5 s | 76/80 (95 %) |
| Video 1: falsos positivos | 1 de 157 (0.6 %) |
| Video 3: cobertura | 8/8 presentes; `jose` 0 confirmaciones |
| Video 3: identidad | 1 falso positivo residual (30 frames, sin impacto en asistencia) |
| Video 3: re-confirmación tras cada corte | mediana 1.31 s, máx 13.2 s |
| ROC | AUC 0.744; umbral 0.30 |
| Accesorios | lentes sin efecto (p=0.512); gorra sola sin efecto; **gorra+lentes** mediana 0.315 (2/4 pasan) |
| Intrusos (video 1, 44 s) | 130 pistas desconocidas → **6 filas** finales; 8 confirmados intactos |
| Intrusos (video 3) | 18 de 21 desconocidos detectados con el corte 0.18 |

**Cómo se logra reconocer con accesorios** (jesús, registrado sin ellos): centroide
promediado de 50 frames con diversidad de pose + umbral 0.30 (con 0.35 pasaba 1 de 4) +
3 votos/ancla (una racha de frames buenos lo confirma) + anti-spoofing omitido en caras
pequeñas. Es el caso más frágil del sistema.

---

## 8. Incidente de identidad (importante)

- **Qué pasó:** en el video 1 (36–107 s) aparecieron **dos caras etiquetadas `dani` al
  mismo tiempo**. Una era una estudiante **no registrada** (sim 0.325–0.336); la real
  estaba en 0.63–0.65.
- **Causa:** modelo de **un centroide por persona**; el vector de la desconocida cae
  dentro del vecindario de `dani`. No era umbral ni número de votos.
- **Corrección aplicada (Fase 1):** conflicto entre pistas simultáneas. Resultado medido:
  **2 falsos positivos → 0**, los 9 siguen confirmándose con las mismas similitudes.
- **Límite residual:** el falso positivo del video 3 (única pista que reclama `dani`, sin
  conflicto) sigue ahí. Queda como `baseline=1` en la regresión.
- **Descartado con datos:** subir el umbral global (pierde a `jesus`), umbral por persona
  (pierde a `fran` y `kate`), media podada/medoid (no mueve el falso positivo).
- **Única vía medida:** multi-prototipo (top-3 sobre las 50 fotos), que exige cambiar la
  comparación del motor. Fuera del alcance acordado.
- Documentos: `09_incidente_identidad_dani.md` y `10_plan_mejora_motor.md`.

---

## 9. Comandos útiles

```bash
# Servidor web
python manage.py runserver

# Migraciones y base vectorial
python manage.py migrate
python manage.py importar_centroides          # pickle -> pgvector (idempotente)

# Ver el flujo paso a paso (no escribe en la base)
python manage.py ver_flujo --desde 36 --hasta 107 --detalle
# Con escritura de demo y SQL visible
python manage.py ver_flujo --crear-sesion --sql --limpiar

# Asistencia desde consola, escribiendo en la base (cámara 0 o un video)
# El ID sale de una sesión abierta en /academic/ (panel del profesor) o en /admin/
python scripts/asistencia_unificada_optimizada.py --sesion <ID> --fuente 0
python scripts/asistencia_unificada_optimizada.py --sesion <ID> --fuente data/unc/salon/VID_20260907_102851.mp4
# Si no hay sesión abierta: crea una de demo (y --limpiar la borra al terminar)
python scripts/asistencia_unificada_optimizada.py --crear-sesion --fuente <video|0>
python scripts/asistencia_unificada_optimizada.py --crear-sesion --limpiar --fuente <video>
# Sin --sesion: solo dibuja y escribe data/asistencia.csv (experimentos)

# Regresión de identidad (sale con 1 si hay fallos nuevos)
venv/bin/python experiments/unc/regresion_identidad.py

# Demos
venv/bin/python experiments/unc/grabar_demo_motor.py
venv/bin/python experiments/unc/grabar_demo_registro.py --persona=jesus

# Tests
venv/bin/python -m pytest -m unit
venv/bin/python manage.py test
```

---

## 10. Archivos clave

| Archivo | Función |
|---|---|
| `scripts/asistencia_unificada_optimizada.py` | Motor final (decisión, conflicto, banda) |
| `scripts/registro_video.py` | Registro desde video (CLI) |
| `scripts/registro_unificado.py` | Registro (web) |
| `core/db_config.py` | Lectura/escritura de centroides en pgvector |
| `core/registro_asistencia.py` | Decisiones → `RegistroAsistencia` |
| `core/intrusos.py` | Clasificador de intrusos (sin Django) |
| `core/sesion_demo.py` | Sesión de demo (profesor/asignación/inscripciones) |
| `core/procesador_batch.py` | Batch de imágenes |
| `core/servicio_en_vivo.py` | Servicio de cámara/video en vivo |
| `apps/attendance/models.py` | `FaceEmbedding` (vector), sesiones y asistencia |
| `apps/attendance/management/commands/importar_centroides.py` | Migración del pickle |
| `apps/attendance/management/commands/ver_flujo.py` | Diagnóstico paso a paso |
| `experiments/unc/regresion_identidad.py` | Regresión con fixture |
| `experiments/unc/grabar_demo_*.py` | Generadores de video |
| `docs/base_datos.md` | Esquema y operación de la BD |
| `docs/motor.md` | Documentación del motor, línea por línea (en progreso) |
| `docs/motor_2.md` | Capítulo 2: preprocesado, intrusos, pgvector y escritura |
| `docs/motor_3.md` | Capítulo 3: batch, en vivo, sesión demo y registro |
| `docs/web.md` | Web completa: rutas, settings, modelos, vistas, API y plantillas |
| `docs/streaming_mjpeg_vs_polling.md` | Comparación medida MJPEG vs polling 300 ms |

---

## 11. Videos de demostración

En `experiments/unc/demo_videos/` (gitignoreado), versión normal y `_web.mp4` en H.264:

- `demo_motor.mp4` (70.8 s, 129 MB) / `_web` (25 MB): motor con panel, recorre el video 1
  desde el segundo 36 hasta el final.
- `demo_registro_<persona>.mp4` para los **9** estudiantes (~4–5 MB) / `_web` (~1 MB):
  captura de las 50 fotos, filtros y cierre con el centroide.

Usar los `_web` para enviar; los normales son el máster.

---

## 12. Pendientes / próximos pasos

1. **Validar en vivo con cámara real** en la UNC (hoy probado con archivo de video).
2. **Multi-prototipo (Fase 4)** si se relaja la restricción de no tocar la comparación
   del motor.
3. ~~**Unificar la asistencia**~~ (hecho): la consola escribe en `RegistroAsistencia`
   con `--sesion ID` y las reglas de la web; el CSV queda para los experimentos.
4. ~~**Mover credenciales a variables de entorno**~~ (hecho): `settings.py` y
   `core/db_config.py` leen `CV_DB_*` y `DJANGO_*`; los valores por defecto son los
   de desarrollo.
5. Cambiar la contraseña del admin y completar apellidos/matrículas reales.
6. Revalidar el umbral con más aulas y sesiones.
7. **Bajar los intrusos falsos restantes**: los que no se acercan a 0.18 (9 en 44 s del
   video 1) requieren re-identificación; mientras tanto, se descartan desde
   `/sesiones/<id>/`.

---

## 13. Reglas de trabajo (acordadas)

- **No hacer commits ni push** sin instrucción explícita.
- **No borrar archivos** del usuario sin preguntar (pasó con los videos de registro y
  hubo que regenerarlos).
- Los experimentos y demos viven en `experiments/unc/`; no moverlos al código de la app.
- `docs/analisis/unc.zip` está sin seguimiento y es un respaldo: no tocarlo.
- En pantalla: verde = confirmada, amarillo = `posible`, rojo = desconocido. Sin emojis
  ni fuentes raras.

---

## 14. Entorno

- `venv/` con el pipeline CV; `requirements.txt` incluye `pgvector` pero **omite
  `pytest` y `faiss-cpu`** (instalarlos aparte).
- **Cómputo de todas las pruebas:** Intel **i9-14900KF** (24 núcleos), 62 GB de RAM,
  **CPU únicamente**. La máquina tiene una **RTX 4090**, pero el venv instala
  `onnxruntime` (build de CPU), así que `uniface` no usa CUDA. Rendimiento medido:
  **~7 fps** en 1080p (detección 47 ms/frame, embedding 24 ms/cara, FAISS 0.02 ms/cara).
  Detalle completo en `docs/computo.md`.
- Para tests de Django: `ALTER ROLE cv_college_user CREATEDB;` y `CREATE EXTENSION vector`
  en `template1`.
- `ffmpeg` disponible (se usa para las versiones web de los videos).
- Disco: ~17 GB libres al inicio de la sesión; los videos ocupan ~170 MB.

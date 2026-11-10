# Resumen general del proyecto

**Sistema de asistencia universitaria por reconocimiento facial (CV College)**
**Actualizado:** 2026-10-26

---

## 1. Qué es

Una aplicación web en **Django 6.1** con un **motor de visión por computadora**
propio que automatiza la asistencia en un salón de clase:

- Reconoce a los estudiantes registrados a partir de video o cámara.
- Distingue entre estudiante, desconocido (intruso) e intento de suplantación
  (spoofing).
- Guarda la asistencia en **PostgreSQL + pgvector**.
- Permite registrar rostros con un video y consultar el historial de cada sesión.

El foco del proyecto es el **motor**; la web es una interfaz mínima para
registrarlo, verlo funcionar y revisar lo que quedó guardado.

---

## 2. Estado actual (2026-11-10)

| Aspecto | Estado |
|---|---|
| Motor | Estable, con reglas de identidad, conflicto e intrusos |
| Base de datos | PostgreSQL 16 + pgvector 0.6 (10 centroides de 512 d) |
| Estudiantes registrados | 9 (`unc_alexa`, `unc_bob`, …, `unc_kevin`) + `val` (prueba de liveness) |
| Web visible | Directorio, Sesiones y Control de Asistencia |
| Tests | 91 unitarios + 48 de Django, en verde |
| Cómputo | CPU (i9-14900KF), ~7 fps en 1080p; la RTX 4090 no se usa |
| Limitación principal | 1 falso positivo de identidad residual en el video 3 |

---

## 3. Cómo funciona el reconocimiento

El pipeline está en `scripts/asistencia_unificada_optimizada.py` y usa la
librería vendoreada `uniface`.

```
video / cámara
   │
   ├─ RetinaFace ──────────► cajas + 5 landmarks
   │    filtros: ancho > 40 px (20 en la web), confianza ≥ 0.75
   │
   ├─ ByteTrack ───────────► track_id por persona
   │
   ├─ alineación + CLAHE ──► cara normalizada
   │
   ├─ ArcFace ─────────────► vector de 512 dimensiones (norma 1)
   │
   ├─ FAISS IndexFlatIP ───► similitud coseno contra los centroides registrados
   │    umbral: 0.30
   │
   ├─ anti-spoofing ───────► real / fraude (se omite si la cara < 60 px)
   │
   └─ 3 votos consecutivos ► identidad confirmada
```

### Reglas de decisión

| Regla | Valor | Qué hace |
|---|---|---|
| Umbral de identidad | **0.30** | Por debajo, la cara es desconocida |
| Votos | **3 consecutivos** | Evita confirmar por un frame aislado |
| Ancla | **30 frames** (~1 s) | Reutiliza la identidad sin recalcular |
| Conflicto | margen **0.05** | Dos pistas no pueden ser la misma persona |
| Veto | **90 frames** | La pista que pierde no reclama esa identidad |
| Banda de revisión | **0.30–0.40** | Se muestra como `posible <nombre>` (no cambia la decisión) |
| Intruso | mejor score **< 0.18**, **15 frames** | Evita marcar como intruso a un registrado en un frame malo |

El umbral 0.30 se eligió con la curva ROC (300 recortes etiquetados): precisión
0.987, recall 0.872, F1 0.926. En asistencia, un falso positivo pesa más que un
falso negativo.

---

## 4. Base de datos

PostgreSQL 16 con la extensión **pgvector 0.6**. La tabla vectorial es
`attendance_faceembedding`:

- `embedding`: `VectorField(512)`, norma 1.
- Índice **HNSW** con `vector_cosine_ops`.
- Metadatos: `modelo`, `n_muestras`, `fuente` (`VIDEO` / `IMPORTADO` / `WEB`),
  `fecha`, `created_at`, `updated_at`.
- Un vector por usuario (restricción única en `user_id`).

Tablas principales:

| Tabla | Qué guarda |
|---|---|
| `attendance_faceembedding` | Centroide de 512 d por usuario |
| `attendance_sesionclase` | Una captura concreta (fecha, estado, notas) |
| `attendance_registroasistencia` | Una fila por estudiante y sesión, o por evento de seguridad |
| `academic_*` | Asignatura, salón, sección, estudiante, asignación, inscripción |

`RegistroAsistencia` distingue cuatro tipos de evento:

- `ASISTENCIA_NORMAL`: estudiante inscrito y confirmado.
- `OTRA_SECCION`: registrado, pero no pertenece a esa clase.
- `INTRUSO`: persona no registrada (`es_fraude=False`).
- `SPOOFING`: intento de suplantación (`es_fraude=True`, con captura).

El detalle completo está en `docs/base_datos.md`.

---

## 5. La web

La web quedó reducida a tres secciones visibles:

| Ruta | Qué hace |
|---|---|
| `/directorio/` | Registra rostros con un video, re-enrola o revoca la biometría y lista a los registrados |
| `/sesiones/` | Historial de capturas con sus contadores |
| `/sesiones/<id>/` | Detalle: confirmados, intrusos, spoofing, **ausentes**, **revisión 0.30–0.40** (aceptar/rechazar) y **marcar presente** |
| `/en-vivo/` | Control de asistencia con cámara o video, frame anotado |

- La raíz `/` redirige al directorio.
- El login usa `admin` / `admin123` (cambiar la contraseña).
- La gestión de cada sesión está limitada al profesor dueño, coordinación y
  dirección; borrar todas las sesiones exige rol directivo.
- El resto de páginas académicas (panel del profesor, reportes, auditoría, mapa de
  salones) **siguen existiendo por URL**, pero están fuera del menú hasta que se
  rehabiliten.
- Si no hay sesiones, `/en-vivo/` ofrece un botón para crear una de demo.

---

## 6. Flujos de asistencia

| Flujo | Entrada | Escritura |
|---|---|---|
| Consola | Cámara o video | `RegistroAsistencia` con `--sesion`; CSV sin él |
| Batch | Carpeta de imágenes | Una fila consolidada por sesión |
| En vivo | Cámara o video | `RegistroAsistencia` en tiempo real |
| Diagnóstico | Video | `ver_flujo` muestra el paso a paso (opcionalmente escribe) |

La consola escribe en la misma base y con las mismas reglas que la web; el CSV
(`data/asistencia.csv`) queda solo para experimentos.

---

## 7. Resultados de la evaluación

| Medición | Resultado |
|---|---|
| Video 1: cobertura | 9/9 estudiantes |
| Video 1: confirmación en ventana ±0.5 s | 76/80 (95 %) |
| Video 3: cobertura | 8/8 presentes (`jose` ausente, 0 confirmaciones) |
| Video 3: identidad | 1 falso positivo residual (~1 s, sin impacto en asistencia) |
| ROC | AUC 0.962; AP 0.953; EER 0.080; umbral 0.30 |
| Accesorios | Lentes sin efecto; gorra+lentes mediana 0.315 (el caso más frágil es `jesus`) |
| Intrusos (video 1, 44 s) | 7 filas (todas falsas, regla v2) |
| Intrusos (video 3) | 29 filas: 21 desconocidos reales + 8 registrados |

Los informes completos están en `docs/analisis/unc/`.

---

## 8. Limitaciones conocidas

- **Un falso positivo de identidad** en el video 3: una estudiante no registrada
  se confirma como `dani` durante ~1 s. La única vía medida para corregirlo es
  multi-prototipo (top-3 sobre las 50 fotos), fuera del alcance acordado.
- **Intrusos:** el clasificador reduce los falsos, pero no los elimina. Quedan
  caras pequeñas que no se acercan a 0.18; el profesor puede descartarlas desde
  `/sesiones/<id>/`.
- **mAP de detección:** no se calculó. Requiere ground truth exhaustivo de todas
  las caras de los frames evaluados; no hay esa anotación. Se reportan IoU,
  estrés de emparejamiento y tasas de detección.
- **Tiempo real:** a ~7 fps en CPU, el flujo en vivo va más lento que una cámara
  de 30 fps.
- **Credenciales** de PostgreSQL por variables de entorno (`CV_DB_*`); los valores
  por defecto son los de desarrollo.
- **Apellidos y matrículas** de los 9 estudiantes son provisionales.

---

## 9. Estructura del repositorio

```
scripts/
  asistencia_unificada_optimizada.py   motor (decisión, conflicto, intrusos)
  registro_video.py                    registro por video (CLI)
  registro_unificado.py                registro (web)
core/
  utils_facial.py                      alineación, CLAHE, embedding
  intrusos.py                          clasificador de intrusos
  registro_asistencia.py               decisiones → RegistroAsistencia
  db_config.py                         lectura/escritura en pgvector
  procesador_batch.py                  flujo por lotes
  servicio_en_vivo.py                  flujo en vivo
  sesion_demo.py                       sesión de demostración
apps/
  users/                               usuario, roles, permisos
  academic/                            asignaturas, salones, estudiantes
  attendance/                          modelos, vistas, API, sesiones
experiments/unc/                       fases, regresión y demos
docs/                                  documentación
```

---

## 10. Documentación

| Documento | Contenido |
|---|---|
| `docs/motor.md` | Motor explicado archivo por archivo y línea por línea |
| `docs/computo.md` | Hardware, versiones, rendimiento y migración a GPU |
| `docs/base_datos.md` | Esquema, pgvector, consultas y operación |
| `docs/comandos.md` | Comandos listos para ejecutar |
| `docs/contexto_nuevo_chat.md` | Traspaso completo para retomar el proyecto |
| `docs/analisis/unc/` | Experimentos, resultados y decisiones con datos |
| `experiments/unc/metricas/` | Métricas consolidadas (ROC/PR/AUC/AP/EER/d'), CSV y `resumen.md` para la tesis |

---

## 11. Cómo empezar

```bash
cd /home/valentina2/Desktop/Folds/VisionProjects/face-recognition-college
venv/bin/python manage.py runserver
```

Abrir `http://localhost:8000/login` con `admin` / `admin123`.

Para probar el motor sin la web:

```bash
venv/bin/python scripts/asistencia_unificada_optimizada.py \
  --fuente data/unc/salon/VID_20260907_102851.mp4
```

Todos los comandos están en `docs/comandos.md`.

---

## 12. En una frase

Un motor de reconocimiento facial (RetinaFace + ArcFace + anti-spoofing) que
decide identidad con umbral 0.30 y consenso temporal, guarda la asistencia en
PostgreSQL + pgvector y se controla desde una web mínima de tres pantallas.

# Motor de asistencia — Documentación técnica

## Capítulo 3 — Flujos, sesión de demo y registro

**Archivos cubiertos:**

| Archivo | Líneas | Rol |
|---|---|---|
| `core/procesador_batch.py` | 162 | Asistencia desde una carpeta de imágenes |
| `core/servicio_en_vivo.py` | 177 | Asistencia en vivo (cámara o video) con hilo |
| `core/sesion_demo.py` | 98 | Crea y limpia la sesión de demostración |
| `scripts/registro_video.py` | 209 | Registro por video desde la terminal |
| `scripts/registro_unificado.py` | 329 | Registro interactivo y headless (web) |

**Cómo encajan:**

```
                    ┌─ core/procesador_batch.py ──┐
decisiones ─────────┤                             ├── RegistroAsistencia
                    └─ core/servicio_en_vivo.py ──┘
                              ▲
                              │ hilo + JPEG
                          /en-vivo/ (HTTP)

registro_video.py ──┐
                    ├── centroide 512 ── pgvector
registro_unificado.py┘   (web vía API)
```

---

# Parte A — `core/procesador_batch.py` (162 líneas)

**Rol:** procesar una carpeta de imágenes como observaciones independientes y
escribir la asistencia consolidada de una sesión.

**Dependencias:** `cv2`, `json`, `os`, `time`, Django (`settings`, `timezone`,
`close_old_connections`), `core.registro_asistencia` (el flujo por lotes).

## A.1 Docstring, imports y constantes (líneas 1-36)

```python
"""
Procesamiento por lotes de imágenes para una sesión de clase.

Usa el motor calibrado (`MotorAsistencia`) sobre las imágenes en orden, como si
fueran frames de un video: mismo umbral (0.30), mismo filtro de tamaño (20 px),
misma confianza de detección (0.75), mismo anti-spoofing y la misma regla de
3 votos consecutivos que se validó en las fases 0–8.

Escribe la asistencia en la base (`RegistroAsistencia`). **No simula ni inventa
datos**: si no hay imágenes reales, deja el estado en `error` y termina.

Genera en `MEDIA_ROOT/batch_processed/<sesion_id>/`:
  status.json   progreso que consume el panel del profesor
  latest.jpg    último frame procesado y anotado
"""

import json
import os
import time

import cv2
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from apps.attendance.models import RegistroAsistencia, SesionClase
from core.db_config import alias_de
from core.registro_asistencia import (
    acumular_decisiones,
    mapa_estudiantes,
    nuevo_acumulado,
    registrar_lote,
    reiniciar_pistas,
)

MAX_IMAGENES = 60  # muestreo uniforme; el motor tarda ~130 ms por imagen
```

- **Líneas 1-14:** el contrato del archivo. Destaca que **no simula datos**: si no
  hay imágenes, deja `status.json` en `error`.
- **Línea 17-18:** `json` para el estado, `os` para rutas, `time` para el
  timestamp.
- **Línea 21-22:** `settings.MEDIA_ROOT` y `close_old_connections` (el batch corre
  en un hilo, y las conexiones de Django no se comparten entre hilos).
- **Línea 23:** `timezone` para medir la duración.
- **Línea 25:** los modelos que se van a escribir/leer.
- **Línea 26:** `alias_de` para el alias del profesor.
- **Línea 27-33:** las cuatro funciones del flujo por lotes.
- **Línea 36:** máximo de imágenes a procesar. Con 60 imágenes a ~130 ms cada una,
  el lote tarda ~8 s de cómputo, más la carga de modelos.

## A.2 `_listar_imagenes()` (líneas 39-49)

```python
def _listar_imagenes(ruta_carpeta):
    if not ruta_carpeta or not os.path.isdir(ruta_carpeta):
        return []
    archivos = sorted(
        f for f in os.listdir(ruta_carpeta)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(archivos) > MAX_IMAGENES:
        paso = max(1, len(archivos) // MAX_IMAGENES)
        archivos = archivos[::paso][:MAX_IMAGENES]
    return [os.path.join(ruta_carpeta, f) for f in archivos]
```

- **Línea 40-41:** si la ruta está vacía o no es un directorio, devuelve lista
  vacía.
- **Línea 42-46:** lista y ordena solo imágenes (`.jpg`, `.jpeg`, `.png`). El orden
  alfabético da una secuencia estable.
- **Línea 47-49:** si hay más de 60, toma una muestra uniforme:
  - `paso = len // 60` (al menos 1),
  - `archivos[::paso]` toma uno de cada `paso`,
  - `[:60]` corta el excedente.
- **Línea 50:** devuelve rutas completas.

**Por qué muestrear:** procesar 500 fotos de un salón no mejora la decisión y
multiplica el tiempo. 60 imágenes son suficientes para exigir evidencia repetida.

## A.3 `_dibujar()` (líneas 52-66)

```python
def _dibujar(frame, decisiones):
    from scripts.asistencia_unificada_optimizada import COLORES

    for d in decisiones:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORES.get(d["estado"], (0, 0, 255))
        if d.get("revision"):
            color = (0, 215, 255)  # amarillo: identidad posible, revisar
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = f"posible {d['nombre']}" if d.get("revision") else d["nombre"]
        if d.get("similitud") is not None:
            texto += f" {d['similitud']:.2f}"
        cv2.putText(frame, texto, (x1 + 2, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame
```

- **Línea 53:** import local de la tabla de colores. Evita cargar `uniface` al
  importar este módulo.
- **Línea 55:** recorre las decisiones.
- **Línea 56-58:** color según el estado.
- **Línea 59-60:** si la identidad está en la banda 0.30-0.40, la caja se pinta
  amarilla.
- **Línea 61:** rectángulo de la cara.
- **Línea 62:** texto: `posible <nombre>` si está en revisión; si no, el nombre.
- **Línea 63-64:** añade la similitud con dos decimales si existe.
- **Línea 65-66:** escribe el texto encima de la caja, sin salirse del frame
  (`max(14, y1 - 6)`).
- **Línea 67:** devuelve el frame anotado (para guardar `latest.jpg`).

## A.4 `procesar_batch_imagenes()` (líneas 69-162)

```python
def procesar_batch_imagenes(sesion_id, ruta_carpeta=None):
    """Procesa la carpeta y escribe la asistencia de la sesión en la base."""
    close_old_connections()
    print(f"[BATCH] Sesión {sesion_id} | carpeta: {ruta_carpeta}")

    try:
        sesion = SesionClase.objects.select_related(
            "asignacion_clase__profesor"
        ).get(id=sesion_id)
    except SesionClase.DoesNotExist:
        print(f"[BATCH] No existe la sesión {sesion_id}")
        return

    asignacion = sesion.asignacion_clase
    status_dir = os.path.join(settings.MEDIA_ROOT, "batch_processed", str(sesion_id))
    os.makedirs(status_dir, exist_ok=True)

    def update_status(current, total, state="processing", mensaje=""):
        with open(os.path.join(status_dir, "status.json"), "w", encoding="utf-8") as sf:
            json.dump({
                "current": current,
                "total": total,
                "percent": int((current / total) * 100) if total > 0 else 0,
                "state": state,
                "mensaje": mensaje,
                "timestamp": time.time(),
            }, sf)

    # Se empieza de cero en cada corrida
    RegistroAsistencia.objects.filter(sesion=sesion).delete()
    sesion.notas_profesor = ""
    sesion.save(update_fields=["notas_profesor"])

    imagenes = _listar_imagenes(ruta_carpeta)
    if not imagenes:
        mensaje = f"No se encontraron imágenes en '{ruta_carpeta or 'data/images/'}'."
        print(f"[BATCH] {mensaje}")
        update_status(0, 0, "error", mensaje)
        return

    total = len(imagenes)
    update_status(0, total, "processing", "Cargando modelos...")

    try:
        from scripts.asistencia_unificada_optimizada import MotorAsistencia

        motor = MotorAsistencia(registrar_csv=False)
        motor.tamano_minimo_rostro = 20
        # Cada imagen es una observación independiente: se confirma con el umbral
        # calibrado y luego se exige evidencia repetida en varias imágenes.
        motor.votos_requeridos = 1
    except Exception as e:
        mensaje = f"No se pudo iniciar el motor de reconocimiento: {e}"
        print(f"[BATCH] {mensaje}")
        update_status(0, total, "error", mensaje)
        return

    estudiantes_por_alias = mapa_estudiantes(asignacion)
    profesor = asignacion.profesor
    alias_profesor = alias_de(profesor.username, profesor.first_name)
    minimo_imagenes = min(3, total)
    acumulado = nuevo_acumulado()
    inicio = timezone.localtime()

    try:
        for indice, ruta in enumerate(imagenes, 1):
            frame = cv2.imread(ruta)
            if frame is None:
                update_status(indice, total, "processing", f"Ilegible: {os.path.basename(ruta)}")
                continue

            reiniciar_pistas(motor)
            decisiones = motor.procesar_frame(frame)
            acumular_decisiones(acumulado, decisiones, frame, indice)
            cv2.imwrite(os.path.join(status_dir, "latest.jpg"), _dibujar(frame, decisiones))
            update_status(indice, total, "processing",
                          f"{os.path.basename(ruta)} | caras: {len(decisiones)}")

        registrados = registrar_lote(
            acumulado, sesion, estudiantes_por_alias,
            alias_profesor=alias_profesor, profesor=profesor,
            minimo_imagenes=minimo_imagenes,
        )
        print(f"[BATCH] Confirmados con >= {minimo_imagenes} imágenes: {sorted(registrados)}")
    except Exception as e:
        mensaje = f"Error durante el procesamiento: {e}"
        print(f"[BATCH] {mensaje}")
        update_status(0, total, "error", mensaje)
        return
    finally:
        close_old_connections()

    update_status(total, total, "completed", "Procesamiento finalizado.")
    print(f"[BATCH] Terminado en {(timezone.localtime() - inicio).total_seconds():.1f} s")
```

### Preparación

- **Línea 71:** `close_old_connections()` al entrar. El batch corre en un hilo
  distinto al de la petición; las conexiones heredadas no son válidas.
- **Línea 72:** log con sesión y carpeta.
- **Línea 75-81:** busca la sesión con su asignación y profesor. Si no existe,
  termina.
- **Línea 83-85:** crea `media/batch_processed/<sesion_id>/`.
- **Línea 87-96:** `update_status` escribe `status.json`:
  - `current` y `total`: progreso,
  - `percent`: `int(current/total*100)` (protegido para `total=0`),
  - `state`: `processing` / `completed` / `error`,
  - `mensaje`: texto para el panel,
  - `timestamp`: `time.time()`.
- **Línea 99:** **borra los registros previos de la sesión**. Cada corrida del
  batch empieza de cero.
- **Línea 100-101:** limpia las notas del profesor.
- **Línea 103-108:** lista las imágenes. Si no hay, deja `error` y termina.
- **Línea 110-111:** total y estado inicial.

### Motor

- **Línea 114-116:** crea el motor **sin CSV**.
- **Línea 117:** baja el tamaño mínimo a 20 px (caras del salón).
- **Línea 120:** `votos_requeridos = 1`. Cada imagen es una observación
  independiente; no hay frames consecutivos para votar. La evidencia repetida se
  exige después, en `registrar_lote`.
- **Línea 121-125:** si el motor no carga, estado `error`.

### Procesamiento

- **Línea 127-132:** mapa de inscritos, profesor, alias y mínimo de imágenes
  (`min(3, total)`). Si el lote tiene 2 fotos, se exigen 2.
- **Línea 134-147:** por cada imagen:
  - `cv2.imread` (si es ilegible, avanza),
  - `reiniciar_pistas(motor)`: limpia tracker, anclas, votos y vetos,
  - `procesar_frame`,
  - `acumular_decisiones`,
  - guarda `latest.jpg` con las cajas,
  - actualiza el estado.
- **Línea 148-152:** `registrar_lote` escribe la asistencia consolidada.
- **Línea 154-158:** cualquier excepción deja `error` y no escribe.
- **Línea 159-160:** cierra conexiones.
- **Línea 162-163:** estado `completed` y tiempo total.

### Contrato de `status.json`

```json
{
  "current": 3,
  "total": 10,
  "percent": 30,
  "state": "processing",
  "mensaje": "foto_003.jpg | caras: 4",
  "timestamp": 1761700000.123
}
```

## A.5 Notas de diseño

- **No simula**: si no hay imágenes o el motor falla, el estado queda en `error`.
- **Una fila por estudiante**: lo garantiza `registrar_lote` con
  `update_or_create` y la restricción única `(sesion, estudiante)`.
- **Borrado previo**: volver a lanzar el batch sobre una sesión borra la
  asistencia anterior de esa sesión. Es intencional: el batch es una corrida
  completa, no un incremento.
- **Evidencia mínima**: 3 imágenes (o todas si hay menos). Un estudiante que
  aparece en una sola foto no se registra.
- **Intrusos**: se filtran por similitud `< 0.18` y por conflictos; se escribe
  **una sola fila** por lote con el número de detecciones.

---

# Parte B — `core/servicio_en_vivo.py` (177 líneas)

**Rol:** hilo que lee una cámara o un video, procesa cada frame con el motor,
escribe la asistencia y publica el último frame anotado como JPEG para la web.

**Dependencias:** `threading`, `time`, `cv2`, Django (`close_old_connections`),
`core.db_config`, `core.intrusos`, `core.registro_asistencia`.

## B.1 `_dibujar()` (líneas 24-38)

```python
def _dibujar(frame, decisiones):
    from scripts.asistencia_unificada_optimizada import COLORES

    for d in decisiones:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORES.get(d["estado"], (0, 0, 255))
        if d.get("revision"):
            color = (0, 215, 255)  # amarillo: identidad posible, revisar
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = f"posible {d['nombre']}" if d.get("revision") else d["nombre"]
        if d.get("similitud") is not None:
            texto += f" {d['similitud']:.2f}"
        cv2.putText(frame, texto, (x1 + 2, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame
```

Idéntica a la del batch: verde confirmado, amarillo `posible`, rojo desconocido;
nombre + similitud junto a la caja. No usa sombra ni fuentes raras.

## B.2 Estado y API pública (líneas 41-92)

```python
class ServicioEnVivo:
    def __init__(self):
        self._lock = threading.Lock()
        self._hilo = None
        self._activo = False
        self._jpeg = None
        self._estado = {
            "activo": False,
            "fuente": None,
            "sesion_id": None,
            "frames": 0,
            "fps": 0.0,
            "caras": 0,
            "confirmados": [],
            "error": None,
        }
```

- **Línea 43:** un `Lock` protege el estado y el JPEG; el hilo escribe y la
  petición HTTP lee.
- **Línea 44:** referencia al hilo (no se une, es `daemon`).
- **Línea 45:** bandera de control del bucle.
- **Línea 46:** último frame codificado en JPEG (bytes) o `None`.
- **Línea 47-56:** estado que consume `/api/en-vivo/estado/`.

```python
    def iniciar(self, fuente, sesion_id=None):
        with self._lock:
            if self._activo:
                return False, "Ya hay una captura en curso."
            self._activo = True
            self._jpeg = None
            self._estado = {
                "activo": True,
                "fuente": str(fuente),
                "sesion_id": sesion_id,
                "frames": 0,
                "fps": 0.0,
                "caras": 0,
                "confirmados": [],
                "error": None,
            }
        self._hilo = threading.Thread(
            target=self._bucle, args=(fuente, sesion_id), daemon=True
        )
        self._hilo.start()
        return True, "Captura iniciada."
```

- **Línea 60:** entra a la sección crítica.
- **Línea 61-62:** si ya hay una captura, rechaza. Solo una por proceso.
- **Línea 63:** activa el bucle.
- **Línea 64:** limpia el frame anterior.
- **Línea 65-74:** reinicia el estado (confirmados vuelve a vacío).
- **Línea 75-78:** lanza el hilo. `daemon=True` hace que muera con el proceso.
- **Línea 79:** devuelve éxito y mensaje.

```python
    def detener(self):
        with self._lock:
            self._activo = False
        return True, "Captura detenida."

    def frame_jpeg(self):
        with self._lock:
            return self._jpeg

    def estado(self):
        with self._lock:
            return dict(self._estado)
```

- **Línea 82-83:** pone `_activo` en `False`; el bucle termina en la siguiente
  iteración.
- **Línea 87-88:** devuelve el último JPEG (o `None`).
- **Línea 91-92:** devuelve una **copia** del estado para que el llamador no
  modifique el original.

## B.3 `_bucle()` (líneas 95-168)

```python
    def _bucle(self, fuente, sesion_id):
        close_old_connections()
        sesion = estudiantes = profesor = alias_profesor = None
        motor = None
        try:
            from apps.attendance.models import SesionClase
            from scripts.asistencia_unificada_optimizada import MotorAsistencia

            if sesion_id:
                sesion = SesionClase.objects.select_related(
                    "asignacion_clase__profesor"
                ).get(id=sesion_id)
                asignacion = sesion.asignacion_clase
                estudiantes = mapa_estudiantes(asignacion)
                profesor = asignacion.profesor
                alias_profesor = alias_de(profesor.username, profesor.first_name)

            motor = MotorAsistencia(registrar_csv=False)
            motor.tamano_minimo_rostro = 20
            seguimiento = SeguimientoIntrusos()

            origen = int(fuente) if str(fuente).isdigit() else fuente
            captura = cv2.VideoCapture(origen)
            if not captura.isOpened():
                self._fallar(f"No se pudo abrir la fuente: {fuente}")
                return

            fps_video = captura.get(cv2.CAP_PROP_FPS) or 0
            espera = 1.0 / fps_video if fps_video > 0 else 0
            inicio = time.time()

            while self._activo:
                arranque = time.time()
                ok, frame = captura.read()
                if not ok:
                    break

                decisiones = motor.procesar_frame(frame)

                if sesion is not None:
                    resumen = aplicar_decisiones(
                        decisiones, frame, sesion, estudiantes,
                        alias_profesor=alias_profesor, profesor=profesor,
                        seguimiento=seguimiento,
                    )
                else:
                    resumen = {"confirmados": []}

                anotado = _dibujar(frame, decisiones)
                ok, buffer = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buffer.tobytes()
                        self._estado["frames"] += 1
                        self._estado["caras"] = len(decisiones)
                        self._estado["fps"] = self._estado["frames"] / max(1e-6, time.time() - inicio)
                        confirmados = set(self._estado["confirmados"])
                        confirmados.update(resumen.get("confirmados", []))
                        self._estado["confirmados"] = sorted(confirmados)

                if espera:
                    time.sleep(max(0.0, espera - (time.time() - arranque)))

            captura.release()
        except Exception as e:
            self._fallar(str(e))
            return
        finally:
            close_old_connections()
            with self._lock:
                self._activo = False
                self._estado["activo"] = False
                if self._estado["error"] is None:
                    self._estado["mensaje"] = "Captura finalizada."
```

- **Línea 96:** cierra conexiones heredadas; este hilo abre las suyas.
- **Línea 97-98:** inicializa el contexto a `None`.
- **Línea 100-101:** imports perezosos de los modelos y del motor.
- **Línea 103-110:** si hay sesión, carga inscritos, profesor y alias. Sin sesión,
  solo se visualiza.
- **Línea 112-113:** motor sin CSV y con caras de 20 px.
- **Línea 114:** clasificador de intrusos para esta captura.
- **Línea 116:** convierte la fuente: `"0"` → `0` (cámara); si no, ruta.
- **Línea 117-120:** abre la fuente; si falla, `_fallar`.
- **Línea 122-124:** FPS del video y tiempo de espera por frame
  (`espera = 1/fps`). Si el video no reporta FPS, no espera.
- **Línea 126:** bucle mientras `_activo`.
- **Línea 127-130:** lee un frame; si no hay, termina.
- **Línea 132:** procesa el frame.
- **Línea 134-141:** si hay sesión, escribe la asistencia y obtiene el resumen;
  si no, resumen vacío.
- **Línea 143:** dibuja.
- **Línea 144:** codifica a JPEG con calidad 80.
- **Línea 145-153:** sección crítica:
  - guarda el JPEG,
  - incrementa frames,
  - actualiza caras,
  - calcula FPS promedio (`frames / tiempo`),
  - acumula confirmados en un `set` y los guarda ordenados.
- **Línea 155-156:** espera lo que falte para respetar el ritmo del video. Si el
  procesamiento ya tardó más que `espera`, no duerme (`max(0.0, ...)`).
- **Línea 158:** libera la captura.
- **Línea 159-161:** cualquier excepción se reporta en el estado.
- **Línea 162-168:** `finally`: cierra conexiones, marca inactivo y deja
  `mensaje` si no hubo error.

## B.4 `_fallar()` y singleton (líneas 170-177)

```python
    def _fallar(self, mensaje):
        with self._lock:
            self._activo = False
            self._estado["activo"] = False
            self._estado["error"] = mensaje


servicio_en_vivo = ServicioEnVivo()
```

- **Línea 171-174:** apaga el servicio y guarda el error.
- **Línea 177:** instancia única por proceso. La importa `apps/attendance/api.py`.

## B.5 Estado que consume la web

```json
{
  "activo": true,
  "fuente": "0",
  "sesion_id": 94,
  "frames": 120,
  "fps": 6.8,
  "caras": 5,
  "confirmados": ["alexa", "bob"],
  "error": null,
  "mensaje": "Captura finalizada."
}
```

## B.6 Notas de diseño

- **Un solo servicio por proceso:** `iniciar` rechaza una segunda captura.
- **Hilo daemon:** no bloquea el apagado de Django.
- **Sin sesión:** se visualiza, pero no se escribe nada.
- **Ritmo:** se intenta respetar el FPS del video; con CPU a ~7 fps, un video de
  30 fps va más lento que tiempo real.
- **Reconexión:** si la cámara falla, el bucle termina; no reintenta solo.

---

# Parte C — `core/sesion_demo.py` (98 líneas)

**Rol:** crear y limpiar una sesión de demostración cuando no hay datos
académicos (profesor, asignatura, salón, sección, asignación, inscripciones).

**Dependencias:** solo `os` al importar; Django se carga dentro de las funciones.

## C.1 `configurar_django()` (líneas 16-23)

```python
def configurar_django():
    """Configura Django una sola vez, sin importarlo al cargar el módulo."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cv_college.settings")
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()
```

- **Línea 18:** variable de entorno con el módulo de settings.
- **Línea 19-20:** imports perezosos.
- **Línea 22-23:** `apps.ready` evita llamar `django.setup()` dos veces (rompería
  la inicialización).

## C.2 `crear_sesion_demo()` (líneas 26-69)

```python
def crear_sesion_demo():
    """Crea una sesión de demo nueva y devuelve la `SesionClase`."""
    configurar_django()

    from django.contrib.auth import get_user_model

    from apps.academic.models import (
        AsignacionClase,
        Asignatura,
        Estudiante,
        Inscripcion,
        Salon,
        Seccion,
    )
    from apps.attendance.models import SesionClase

    User = get_user_model()
    profesor, _ = User.objects.get_or_create(
        username="prof_flujo",
        defaults={"first_name": "Profesor", "last_name": "Flujo",
                  "role": User.Role.PROFESSOR},
    )
    asignatura, _ = Asignatura.objects.get_or_create(
        codigo="DEMO-FLUJO", defaults={"nombre": "Demo de flujo"}
    )
    salon, _ = Salon.objects.get_or_create(
        nombre="Salón Demo", defaults={"capacidad": 30}
    )
    seccion, _ = Seccion.objects.get_or_create(codigo="DEMO-S")
    asignacion, _ = AsignacionClase.objects.get_or_create(
        profesor=profesor, asignatura=asignatura, seccion=seccion, salon=salon,
        defaults={"dia_semana": AsignacionClase.DiaSemana.LUNES,
                  "horario_inicio": "08:00", "horario_fin": "10:00"},
    )
    for estudiante in Estudiante.objects.all():
        Inscripcion.objects.get_or_create(
            estudiante=estudiante, asignacion_clase=asignacion
        )

    # Cada llamada crea una sesión nueva: no se reutiliza ni se borran datos de
    # una captura anterior.
    return SesionClase.objects.create(
        asignacion_clase=asignacion, estado=SesionClase.Estado.EN_CURSO
    )
```

- **Línea 28:** configura Django.
- **Línea 30-40:** imports de los modelos (después de `setup`).
- **Línea 42:** modelo de usuario activo.
- **Línea 43-47:** profesor demo `prof_flujo`, con rol `PROFESSOR`.
- **Línea 48-50:** asignatura `DEMO-FLUJO`.
- **Línea 51-53:** salón `Salón Demo`.
- **Línea 54:** sección `DEMO-S`.
- **Línea 55-59:** asignación que une profesor, asignatura, sección y salón.
- **Línea 60-63:** inscribe a **todos** los estudiantes existentes (los 9).
- **Línea 65-69:** crea una sesión **nueva** cada vez. No reutiliza la anterior
  ni borra sus filas. El comentario lo deja explícito.

**Por qué así:** una versión anterior reutilizaba la sesión demo y borraba sus
registros, lo que destruyó datos de una prueba. Ahora cada llamada es una sesión
independiente.

## C.3 `limpiar_sesion_demo()` (líneas 72-98)

```python
def limpiar_sesion_demo(sesion):
    """Borra la sesión de demo y, si no quedan otras, su andamiaje académico."""
    configurar_django()

    from django.contrib.auth import get_user_model

    from apps.academic.models import (
        Asignatura,
        Inscripcion,
        Salon,
        Seccion,
    )
    from apps.attendance.models import SesionClase

    asignacion = sesion.asignacion_clase
    sesion.delete()

    # Si quedan otras sesiones de la asignación de demo, se conservan.
    if SesionClase.objects.filter(asignacion_clase=asignacion).exists():
        return

    Inscripcion.objects.filter(asignacion_clase=asignacion).delete()
    asignacion.delete()
    Asignatura.objects.filter(codigo="DEMO-FLUJO").delete()
    Salon.objects.filter(nombre="Salón Demo").delete()
    Seccion.objects.filter(codigo="DEMO-S").delete()
    get_user_model().objects.filter(username="prof_flujo").delete()
```

- **Línea 86:** guarda la asignación antes de borrar la sesión.
- **Línea 87:** borra la sesión (y sus registros por cascada).
- **Línea 90-91:** si quedan otras sesiones de la demo, **no** borra el andamiaje.
- **Línea 93-98:** si era la última, borra inscripciones, asignación, asignatura,
  salón, sección y el profesor demo.

## C.4 Quién la usa

| Llamador | Para qué |
|---|---|
| `scripts/asistencia_unificada_optimizada.py --crear-sesion` | Sesión de prueba |
| `manage.py ver_flujo --crear-sesion` | Sesión de diagnóstico |
| `/en-vivo/sesion-demo/` | Botón "Crear sesión de demo" |

---

# Parte D — `scripts/registro_video.py` (209 líneas)

**Rol:** registrar a una persona desde un video, extrayendo 50 fotos alineadas y
calculando su centroide.

**Dependencias:** `cv2`, `numpy`, `pathlib`, `sys`, `os`, `uniface`,
`core.utils_facial`, `core.db_config` (dentro de la función).

## D.1 Constantes (líneas 14-24)

```python
RAIZ = Path(__file__).resolve().parents[1]
DATASET_PATH = RAIZ / "data/dataset_cv/frames"
DB_FILE = RAIZ / "data/database_embeddings.pkl"

MAX_FOTOS = 50
MIN_FOTOS = 15
CANDIDATOS_OBJETIVO = 300
UMBRAL_CALIDAD_BLUR = 60
UMBRAL_BRILLO_MIN = 20
UMBRAL_LIVENESS_MIN = 0.90
UMBRAL_POSE_REPETIDA = 0.95
```

- **Línea 14:** raíz del proyecto.
- **Línea 15:** carpeta donde se guardan las fotos por persona.
- **Línea 16:** ruta del pickle histórico. **Ya no se usa** (la base es pgvector);
  queda como referencia.
- **Línea 18 `MAX_FOTOS = 50`:** objetivo de fotos por persona.
- **Línea 19 `MIN_FOTOS = 15`:** mínimo para aceptar el registro.
- **Línea 20 `CANDIDATOS_OBJETIVO = 300`:** frames a muestrear del video. Con 50
  fotos buenas de 300 candidatos hay margen para descartar.
- **Línea 21 `UMBRAL_CALIDAD_BLUR = 60`:** varianza mínima del Laplaciano.
- **Línea 22 `UMBRAL_BRILLO_MIN = 20`:** media de grises mínima.
- **Línea 23 `UMBRAL_LIVENESS_MIN = 0.90`:** confianza mínima de que sea real.
- **Línea 24 `UMBRAL_POSE_REPETIDA = 0.95`:** si una foto se parece demasiado a
  otra ya guardada, se descarta. Evita 50 fotos casi idénticas.

## D.2 `_mejor_rostro()` (líneas 27-28)

```python
def _mejor_rostro(rostros):
    return max(rostros, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
```

- **Línea 28:** devuelve la detección con mayor área (ancho × alto). En una imagen
  con varias caras, la más grande es la que se está registrando.

## D.3 `registrar_desde_video()` (líneas 31-199)

```python
def registrar_desde_video(
    nombre,
    ruta_video,
    db_file=DB_FILE,
    dataset_path=DATASET_PATH,
    check_liveness=False,
    mostrar=True,
):
    """
    Extrae hasta MAX_FOTOS frames alineados (una sola cara por frame, muestreo uniforme
    en todo el video), genera el centroide normalizado y lo guarda en la DB.

    Returns:
        dict: la DB de embeddings resultante, o None si falló o hay menos de MIN_FOTOS.
    """
    ruta_video = Path(ruta_video)
    if not ruta_video.exists():
        print("Error: el archivo de video no existe.")
        return None

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print("Error: no se pudo abrir el video.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_value = max(1, total_frames // CANDIDATOS_OBJETIVO) if total_frames > 0 else 1

    print("Cargando modelos...")
    detector = create_detector("retinaface")
    recognizer = create_recognizer("arcface")
    spoofer = create_spoofer() if check_liveness else None

    path_estudiante = Path(dataset_path) / nombre
    path_estudiante.mkdir(parents=True, exist_ok=True)
    for jpg_previo in path_estudiante.glob("*.jpg"):
        jpg_previo.unlink()
    print(f"Directorio: {path_estudiante} (limpiado)")
    print(f"Paso de muestreo: {step_value} frames | Liveness: {check_liveness}")

    window_name = "Registro desde Video"
    if mostrar:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    count = 0
    frames_counter = 0
    vectores_cap = []

    try:
        while count < MAX_FOTOS:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            frames_counter += 1
            if frames_counter % step_value != 0:
                continue

            rostros = detector.detect(frame)
            if len(rostros) != 1:
                continue

            rostro = rostros[0]
            landmarks = rostro.landmarks
            if landmarks is None or len(landmarks) < 2:
                continue

            rostro_alineado = alinear_rostro(frame, landmarks)
            if rostro_alineado is None:
                continue
            if es_imagen_borrosa(rostro_alineado, umbral=UMBRAL_CALIDAD_BLUR):
                continue

            gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < UMBRAL_BRILLO_MIN:
                continue

            if check_liveness:
                try:
                    resultado_spoof = spoofer.predict(frame, rostro.bbox)
                    if not resultado_spoof.is_real or resultado_spoof.confidence < UMBRAL_LIVENESS_MIN:
                        continue
                except Exception:
                    continue

            rostros_crop = detector.detect(rostro_alineado)
            if not rostros_crop:
                continue
            vec = embedding_facial(recognizer, rostro_alineado, _mejor_rostro(rostros_crop).landmarks)

            max_sim_pose = max((float(np.dot(vec, v)) for v in vectores_cap), default=0.0)
            if max_sim_pose > UMBRAL_POSE_REPETIDA:
                continue

            filename = path_estudiante / f"{nombre}_{count}.jpg"
            cv2.imwrite(str(filename), rostro_alineado)
            vectores_cap.append(vec)
            count += 1
            print(f"  Foto {count}/{MAX_FOTOS} (sim max con prev: {max_sim_pose:.3f})")

            if mostrar:
                display = frame.copy()
                x1, y1, x2, y2 = map(int, rostro.bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Fotos: {count}/{MAX_FOTOS}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Cancelado por el usuario.")
                    break
    finally:
        cap.release()
        if mostrar:
            cv2.destroyAllWindows()
        print(f"\nExtraccion finalizada: {count} fotos")

    if count < MIN_FOTOS:
        print(f"Error: solo {count} fotos válidas (mínimo {MIN_FOTOS}). No se guarda en la DB.")
        return None

    print("\nGenerando embeddings...")
    from core.db_config import cargar_centroides, guardar_centroide

    existentes = cargar_centroides()
    print(f"Base vectorial cargada: {len(existentes)} personas")

    if nombre in existentes:
        print(f"Aviso: '{nombre}' ya existía en la base y será reemplazado.")

    vectores_estudiante = []
    for ruta_imagen in sorted(path_estudiante.glob("*.jpg")):
        img = cv2.imread(str(ruta_imagen))
        if img is None or img.size == 0:
            print(f"No se pudo leer: {ruta_imagen.name}")
            continue
        rostros = detector.detect(img)
        if not rostros:
            continue
        vectores_estudiante.append(
            embedding_facial(recognizer, img, _mejor_rostro(rostros).landmarks)
        )

    if len(vectores_estudiante) < MIN_FOTOS:
        print(f"Error: solo {len(vectores_estudiante)} embeddings válidos (mínimo {MIN_FOTOS}).")
        return None

    centroide = np.mean(vectores_estudiante, axis=0)
    centroide = centroide / np.linalg.norm(centroide)

    print("\nGuardando en la base vectorial (pgvector)...")
    guardado = guardar_centroide(
        nombre,
        centroide,
        modelo="arcface",
        n_muestras=len(vectores_estudiante),
        fuente="VIDEO",
        crear_usuario=True,
    )
    if not guardado:
        print("Error: no se pudo guardar el embedding en la base.")
        return None

    total = cargar_centroides()
    print(f"\nTotal registradas: {len(total)}")
    for n in total:
        print(f"  {n}")

    return total
```

### Fase 1 — validación y muestreo (líneas 46-57)

- **Línea 46-49:** comprueba que el video exista.
- **Línea 51-54:** abre con OpenCV; si falla, termina.
- **Línea 56:** total de frames.
- **Línea 57:** paso de muestreo: `total_frames // 300`. Con un video de 3000
  frames, `paso=10` (se procesan 300 candidatos). Con menos de 300 frames,
  `paso=1`.

### Fase 2 — modelos y carpeta (líneas 59-69)

- **Línea 60-62:** carga RetinaFace y ArcFace; el spoofer solo si se pidió
  liveness.
- **Línea 64-65:** crea `data/dataset_cv/frames/<nombre>/`.
- **Línea 66-67:** **borra las fotos anteriores** de esa persona.
- **Línea 68-69:** logs de la carpeta y del paso.

### Fase 3 — extracción de fotos (líneas 71-146)

- **Línea 72-73:** ventana opcional.
- **Línea 75-77:** contadores y vectores ya aceptados.
- **Línea 80:** hasta 50 fotos.
- **Línea 81-84:** lee frame; fin del video.
- **Línea 86-88:** descarta frames que no tocan según el paso.
- **Línea 90-92:** detecta; exige **exactamente una cara**. Con 0 o 2+, no se
  puede saber cuál es la persona.
- **Línea 94-97:** landmarks suficientes.
- **Línea 99-101:** alinea; si falla, descarta.
- **Línea 102-103:** nitidez (varianza del Laplaciano < 60).
- **Línea 105-107:** brillo medio (grises < 20).
- **Línea 109-115:** liveness opcional; un error del spoofer descarta el frame.
- **Línea 117-120:** vuelve a detectar sobre la cara alineada y calcula el
  embedding con `embedding_facial`.
- **Línea 122:** similitud máxima con los vectores ya aceptados:
  `max(v · v_prev)`.
- **Línea 123-124:** si supera 0.95, es una pose repetida.
- **Línea 126-129:** guarda la foto `<nombre>_<n>.jpg` y el vector.
- **Línea 130:** log con la similitud de pose.
- **Línea 132-141:** ventana: caja, contador, `q` para cancelar.
- **Línea 142-146:** `finally`: libera video, cierra ventanas y reporta el total.

### Fase 4 — centroide (líneas 148-199)

- **Línea 148-150:** si hay menos de 15 fotos, no se guarda.
- **Línea 153:** imports de pgvector.
- **Línea 155-156:** carga los centroides existentes (para avisar si se
  reemplaza).
- **Línea 158-159:** aviso de reemplazo.
- **Línea 161-172:** **recalcula** el embedding de cada foto guardada. Aunque ya
  se calcularon al vuelo, se vuelven a obtener del archivo para que el centroide
  salga exactamente de las imágenes persistidas.
- **Línea 174-176:** si tras recalcular quedan menos de 15, falla.
- **Línea 178-179:** centroide:
  `c = (1/n) Σ vᵢ`, luego `ĉ = c / ‖c‖`.
- **Línea 181-192:** guarda con `guardar_centroide`, `fuente="VIDEO"` y
  `crear_usuario=True` (crea el usuario `unc_<nombre>` si no existe).
- **Línea 194-197:** recarga y lista todos los registrados.
- **Línea 199:** devuelve el diccionario final.

## D.4 CLI (líneas 202-209)

```python
if __name__ == "__main__":
    nombre = input("Nombre de la persona a registrar: ").strip()
    if not nombre:
        print("Error: el nombre no puede estar vacio.")
        exit()

    ruta_video = input("Ruta del archivo de video: ").strip()
    registrar_desde_video(nombre, ruta_video)
```

- **Línea 203-206:** pide y valida el nombre.
- **Línea 208-209:** pide la ruta y registra.

## D.5 Notas

- **`db_file` y `DB_FILE` no se usan:** quedaron de la época del pickle. La base
  real es pgvector.
- **Nombre como alias:** `guardar_centroide(nombre, ...)` crea
  `unc_<nombre>`; el motor lo verá como `<nombre>`.
- **Una cara por frame:** si en el video aparece otra persona, esos frames se
  descartan. Es intencional.

---

# Parte E — `scripts/registro_unificado.py` (329 líneas)

**Rol:** registro biométrico interactivo y **headless** (para la web). Comparte
las reglas de calidad con `registro_video.py`, pero usa 15 muestras y tiene una
versión sin ventana ni `input()`.

**Dependencias:** `cv2`, `pickle` (histórico), `numpy`, `pathlib`, `sys`, `os`,
`uniface`, `core.utils_facial`, `core.db_config` (dentro de métodos).

## E.1 Clase y constructor (líneas 15-27)

```python
class RegistroBiometrico:
    def __init__(self, db_path="data/database_embeddings.pkl", max_muestras=15):
        self.db_path = Path(db_path)
        self.max_muestras = max_muestras
        self.umbral_blur = 60
        self.umbral_brillo = 20
        self.umbral_liveness = 0.90

        self.detector = create_detector('retinaface')
        self.recognizer = create_recognizer('arcface')
        self.spoofer = None # Carga perezosa (lazy load)

        self.db_embeddings = self._cargar_db()
```

- **Línea 16:** `db_path` es herencia del pickle; **no se usa** para leer ni
  escribir.
- **Línea 17:** 15 muestras por registro (menos que las 50 del script de video).
- **Línea 19-21:** umbrales de calidad.
- **Línea 23-24:** modelos de detección y reconocimiento.
- **Línea 25:** spoofer perezoso: se carga solo si se pide liveness.
- **Línea 27:** carga los centroides desde pgvector.

## E.2 Carga y guardado (líneas 29-57)

```python
    def _cargar_db(self):
        """Carga los centroides desde la base vectorial (pgvector)."""
        try:
            from core.db_config import cargar_centroides

            return cargar_centroides()
        except Exception as e:
            print(f"[AVISO] No se pudo leer la base vectorial: {e}")
            return {}
```

- **Línea 32:** import perezoso.
- **Línea 34:** devuelve el diccionario de centroides.
- **Línea 35-37:** si la base falla, avisa y devuelve vacío. El registro queda
  operativo, pero sin comparar con lo existente.

```python
    def _guardar_db(self, clave=None):
        """Persiste en pgvector el último embedding calculado.

        Desde la web el Usuario lo crea la API después del cálculo, así que aquí
        solo se actualiza si el usuario ya existe (`crear_usuario=False`).
        """
        if clave is None or clave not in self.db_embeddings:
            return
        try:
            from core.db_config import guardar_centroide

            if guardar_centroide(
                clave, self.db_embeddings[clave], fuente="WEB", crear_usuario=False
            ):
                print(f"[BD] Embedding actualizado para {clave}")
            else:
                print(f"[BD] '{clave}' aún no existe como usuario; lo guarda la API.")
        except Exception as e:
            print(f"Error al guardar en la base vectorial: {e}")
```

- **Línea 45-46:** sin clave o sin embedding, no hace nada.
- **Línea 50-52:** intenta guardar con `crear_usuario=False`.
- **Línea 53-55:** si el usuario no existe, no lo crea; avisa que la API lo hará.
- **Línea 56-57:** cualquier error se reporta, no rompe el registro.

## E.3 `_evaluar_calidad()` (líneas 59-94)

```python
    def _evaluar_calidad(self, frame, rostro, check_liveness=False):
        x1, y1, x2, y2 = map(int, rostro.bbox)

        # Tamaño mínimo
        if (x2 - x1) < 60 or (y2 - y1) < 60:
            return False, "MUY LEJOS", (0, 165, 255)

        # Landmarks
        if rostro.landmarks is None or len(rostro.landmarks) < 5:
            return False, "SIN LANDMARKS", (0, 165, 255)

        rostro_alineado = alinear_rostro(frame, rostro.landmarks)
        if rostro_alineado is None:
            return False, "FALLO ALINEACION", (0, 0, 255)

        # Borrosidad
        if es_imagen_borrosa(rostro_alineado, umbral=self.umbral_blur):
            return False, "BORROSO", (0, 0, 255)

        # Iluminación
        gray = cv2.cvtColor(rostro_alineado, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < self.umbral_brillo:
            return False, "MUY OSCURO", (0, 0, 255)

        # Liveness Gatekeeper
        if check_liveness:
            try:
                if self.spoofer is None:
                    self.spoofer = create_spoofer()
                resultado_spoof = self.spoofer.predict(frame, rostro.bbox)
                if not resultado_spoof.is_real or resultado_spoof.confidence < self.umbral_liveness:
                    return False, f"SPOOF ({resultado_spoof.confidence:.2f})", (0, 0, 255)
            except Exception:
                return False, "ERROR LIVENESS", (0, 0, 255)

        return True, "OK", (0, 255, 0), rostro_alineado
```

Tabla de rechazos:

| Comprobación | Umbral | Mensaje |
|---|---|---|
| Tamaño | < 60×60 px | `MUY LEJOS` |
| Landmarks | < 5 puntos | `SIN LANDMARKS` |
| Alineación | `None` | `FALLO ALINEACION` |
| Nitidez | varianza < 60 | `BORROSO` |
| Brillo | media < 20 | `MUY OSCURO` |
| Liveness | confianza < 0.90 | `SPOOF (0.xx)` |
| Error del spoofer | excepción | `ERROR LIVENESS` |

- **Línea 60:** caja en enteros.
- **Línea 63-64:** la cara debe medir al menos 60×60. Es más estricto que el motor
  (40 px o 20 px) porque aquí se está **creando** el centroide.
- **Línea 67-68:** se necesitan los 5 landmarks.
- **Línea 70-72:** alineación.
- **Línea 75-76:** nitidez.
- **Línea 79-81:** brillo.
- **Línea 84-92:** liveness opcional, con carga perezosa del spoofer.
- **Línea 94:** devuelve `(True, "OK", verde, rostro_alineado)`. El cuarto
  elemento solo existe cuando pasa.

## E.4 `enrolar_usuario()` interactivo (líneas 96-200)

- **Línea 97-99:** si el usuario ya existe en la base, pregunta si sobrescribir.
- **Línea 101-104:** abre el video o cámara.
- **Línea 106:** distingue archivo (`str`) de cámara (`int`).
- **Línea 109-117:** paso de muestreo:
  - video: `max(1, min(8, total_frames // 100))`,
  - video sin FPS: 5,
  - cámara: 1 (procesa todos los frames).
- **Línea 120:** ventana.
- **Línea 126-128:** hasta `max_muestras` (15) vectores.
- **Línea 133-134:** muestreo solo para archivos.
- **Línea 137:** detecta.
- **Línea 142-146:** si hay más de una cara, `MULTIPLES ROSTROS`.
- **Línea 145-177:** si hay una:
  - evalúa calidad,
  - si pasa: CLAHE, detecta de nuevo, embedding,
  - filtro de pose: `max(v · v_prev) > 0.95` → `POSE REPETIDA`,
  - si no, guarda el vector y muestra un recorte en la esquina.
- **Línea 180-184:** barra de progreso y mensaje de estado.
- **Línea 186-187:** muestra y `q` para salir.
- **Línea 193-200:** centroide (`mean` + normalización), guarda en memoria y en
  pgvector con `_guardar_db`.

## E.5 `enrolar_usuario_headless()` (líneas 202-310)

Versión para Django: **sin ventanas, sin `input()`**.

- **Línea 217-223:** si el usuario ya existe y `sobrescribir=False`, devuelve
  error.
- **Línea 225-232:** abre el video; si falla, error.
- **Línea 234-240:** paso dinámico (mismo cálculo que la versión interactiva).
- **Línea 245-287:** bucle:
  - lee frame,
  - muestrea,
  - detecta y exige **exactamente una** cara,
  - evalúa calidad,
  - CLAHE + detección + embedding,
  - filtro de pose (> 0.95),
  - acumula.
- **Línea 291-303:** si hay vectores:
  - centroide,
  - normalización,
  - guarda en `db_embeddings`,
  - `_guardar_db`,
  - devuelve `{exito, muestras, mensaje, embedding}` con el vector como lista.
- **Línea 304-310:** si no hay, devuelve error con el número de frames analizados.

**Respuesta típica:**

```python
{
    "exito": True,
    "muestras": 15,
    "mensaje": "Registro exitoso con 15 muestras de 87 frames.",
    "embedding": [0.0123, -0.0456, ...]  # 512 números
}
```

## E.6 CLI (líneas 312-329)

```python
if __name__ == "__main__":
    motor = RegistroBiometrico()

    while True:
        opcion = input("\n1. Webcam\n2. Video MP4\n3. Salir\nOpcion: ").strip()
        if opcion == '3': break

        if opcion in ['1', '2']:
            id_alumno = input("ID Alumno (Ej: 1234_Nombre): ").strip()
            if not id_alumno: continue

            if opcion == '1':
                motor.enrolar_usuario(id_alumno, origen_video=0)
            else:
                ruta = input("Ruta MP4 (Ej: data/video.mp4): ").strip()
                if Path(ruta).exists():
                    motor.enrolar_usuario(id_alumno, origen_video=ruta)
                else:
                    print("El archivo no existe.")
```

- Menú de tres opciones, validación de id y de ruta, y llamada a la versión
  interactiva.

## E.7 Integración con la web

El flujo real de `/api/enrolar/` es:

1. La API exige autenticación y rol `DEVELOPER`, `COORDINATOR` o `DIRECTOR`;
   valida el video y que el carnet no exista.
2. Llama a `enrolar_usuario_headless(id_usuario=carnet, ...)`.
3. El método calcula el centroide y llama a `_guardar_db`, que usa
   `crear_usuario=False`; como el usuario `unc_<carnet>` aún no existe, **no
   escribe** y avisa.
4. La API, dentro de una transacción, crea:
   - el `User` con `username=carnet`,
   - el `Estudiante` con `matricula=carnet`,
   - el `FaceEmbedding` con el `embedding` devuelto.

**Alias:** `alias_de(username, first_name)` devuelve el **username** cuando no
tiene el prefijo `unc_`, así que el motor ve el carnet, único, y coincide con el
usuario y con su `Estudiante`. Ya no hay diferencia entre el id del registro y el
alias final.

## E.8 Diferencias con `registro_video.py`

| Aspecto | `registro_video.py` | `registro_unificado.py` |
|---|---|---|
| Muestras | 50 (mínimo 15) | 15 (mínimo 1) |
| Muestreo | 300 candidatos | 1 de cada `min(8, total//100)` |
| Ventana | Sí | Sí (interactivo) / No (headless) |
| Calidad | 60 px, blur 60, brillo 20 | 60×60 px, blur 60, brillo 20 |
| Pose repetida | > 0.95 | > 0.95 |
| Centroide | media + normalización | media + normalización |
| Guardado | `guardar_centroide(crear_usuario=True)` | `_guardar_db(crear_usuario=False)` |
| Uso | Terminal | Terminal y web |

---

## Cierre del capítulo

| Archivo | Qué se llevó |
|---|---|
| `procesador_batch.py` | El flujo por lotes y el contrato de `status.json` |
| `servicio_en_vivo.py` | El hilo de captura, el estado y el JPEG |
| `sesion_demo.py` | La sesión de demostración y su limpieza segura |
| `registro_video.py` | El registro de 50 fotos desde un video |
| `registro_unificado.py` | El registro de 15 muestras y su versión web |

**Siguiente:** `docs/web.md` — rutas, modelos, vistas, API y plantillas.

---

## Referencias

| Tema | Documento |
|---|---|
| Capítulo 1 (motor) | `docs/motor.md` |
| Capítulo 2 (preprocesado, intrusos, DB, escritura) | `docs/motor_2.md` |
| Intrusos (medición) | `docs/analisis/unc/11_intrusos.md` |
| Base vectorial | `docs/base_datos.md` |
| Cómputo | `docs/computo.md` |
| Comandos | `docs/comandos.md` |

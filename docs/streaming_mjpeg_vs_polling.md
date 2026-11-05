# Streaming MJPEG vs Polling HTTP 300 ms

**Fecha:** 2026-10-29
**Autor:** análisis asistido
**Estado:** medido el 2026-10-29; el MJPEG se implementó como único modo el 2026-11-05
**Video de prueba:** `data/unc/salon/VID_20260907_102851.mp4` desde el segundo 36

---

## 1. Objetivo

Comparar las dos formas de enviar el video anotado desde el servidor al navegador
en `/en-vivo/`:

1. **Polling HTTP 300 ms** (el modo original): el navegador pide
   `/api/en-vivo/frame/` cada 300 ms.
2. **Streaming MJPEG nativo** (el modo nuevo): el navegador abre una sola
   conexión a `/api/en-vivo/stream/` y recibe los frames por
   `multipart/x-mixed-replace`.

La pregunta concreta: **¿cuál entrega mejor el video y a qué coste?**

---

## 2. Qué dicen los documentos del proyecto

- `docs/diagnostico_motor_y_panel.md` §2.1 ya identificaba el problema del polling:
  "3.3 peticiones HTTP GET por segundo, cabeceras repetidas, parpadeo visual y
  sobrecarga innecesaria", y proponía el MJPEG como **P1 (crítica)**.
- `docs/roadmap_version_2.md` deja para la V2 el multi-prototipo, el benchmark
  GPU, la exportación y los dashboards.

Este informe aporta la **medición empírica** que faltaba.

---

## 3. Cómo está implementado

### 3.1 Polling (original)

- Endpoint: `GET /api/en-vivo/frame/` → devuelve el último JPEG.
- Cliente: `setInterval(..., 300)` en `live.html`; cada 300 ms cambiaba
  `img.src` con un `?t=` para romper la caché.
- El servidor solo guarda `self._jpeg` (bytes) y lo devuelve.
- **Retirado de la UI el 2026-11-05.** El endpoint `/api/en-vivo/frame/` sigue
  existiendo como respaldo, pero la vista ya no lo usa.

### 3.2 MJPEG (nuevo)

- Endpoint: `GET /api/en-vivo/stream/` → `StreamingHttpResponse` con
  `multipart/x-mixed-replace; boundary=frame`.
- Servicio: `ServicioEnVivo.stream_mjpeg()` usa un `threading.Condition`.
  El bucle de captura hace `notify_all()` al publicar un frame; el generador
  espera el aviso y entrega el JPEG.
- Cliente: `frame.src = '/api/en-vivo/stream/?t=...'`; una sola conexión.
- **El JPEG se codifica una sola vez** en el bucle (`cv2.imencode`), y ambos
  transportes comparten los mismos bytes. El stream no recodifica.

---

## 4. Metodología

### 4.1 Preparación

```bash
cd /home/valentina2/Desktop/Folds/VisionProjects/face-recognition-college

# Clip desde el segundo 36 (1080p, 750 frames)
ffmpeg -y -loglevel error -ss 36 -t 25 -i data/unc/salon/VID_20260907_102851.mp4 \
  -an /tmp/opencode/clip36.mp4

# Clip largo (110 s) para la medición de CPU: se repite el clip
ffmpeg -y -loglevel error -stream_loop 4 -i /tmp/opencode/clip36.mp4 -t 110 \
  -c copy /tmp/opencode/clip_largo.mp4

# Servidor aislado en otro puerto
venv/bin/python manage.py runserver 127.0.0.1:8011 --noreload
```

Se autenticó como `admin`/`admin123` y se inició la captura con:

```json
POST /api/en-vivo/iniciar/
{"fuente": "/tmp/opencode/clip36.mp4", "sesion_id": null}
```

Sin `sesion_id` la captura **solo visualiza**, no escribe en la base. Así la
primera medición no contamina datos.

### 4.2 Ventanas de medición

- **Polling:** una petición cada 300 ms durante 12 s.
- **MJPEG:** lectura del flujo multipart durante 12 s.
- **CPU:** ventanas intercaladas **A/B/B/A** de 12 s para cancelar la deriva del
  contenido (el clip se repite y el coste del motor varía según las caras).
- La CPU del servidor se leyó de `/proc/<pid>/stat` (`utime + stime`).

### 4.3 Verificación de identidad

Se creó una sesión de demo temporal (id 99), se inició la captura con
`sesion_id=99` y se revisaron las filas de `RegistroAsistencia`. Al terminar se
borró la sesión temporal con `limpiar_sesion_demo`, sin tocar las sesiones del
usuario (97 y 98).

---

## 5. Resultados

### 5.1 Tabla comparativa

| Métrica | Polling 300 ms | **MJPEG nativo** |
|---|---|---|
| Peticiones HTTP | 3.32 req/s | **1 conexión permanente** |
| Frames producidos por el motor | 6.65 fps | 6.87 fps |
| **Frames entregados al navegador** | **3.17 fps** | **6.91 fps** |
| **Entregados / producidos** | **48 %** | **101 %** |
| Frames duplicados | 0 % | 0 % |
| Latencia añadida | **0–300 ms** (media ~150 ms) | **~0 ms** |
| Time-to-first-frame | — | **14 ms** |
| Latencia de red (req→res) | 16.1 ms media / 23.4 ms máx | — |
| Separación entre frames | 300 ms fijos | 164 ms media; p95 211 ms; máx 232 ms |
| Tamaño medio del frame | 298 KB | 308 KB |
| Ancho de banda | 992 KB/s | **1 874 KB/s** |
| **CPU del servidor** | **31.3 cpu/s** | **31.3 cpu/s** |
| Coste por frame producido | 4.1–5.5 s (ruido del motor) | igual |

### 5.2 Detalle del polling

```
ventana medida      : 12.03 s
peticiones          : 40 (3.32 req/s)
frames producidos   : 75 (6.23 fps)
frames recibidos    : 40 (3.32 fps)
entregados/producidos: 53.3 %
duplicados          : 0
latencia red        : media 16.1 ms | max 23.4 ms
tamaño medio        : 298.4 KB
ancho de banda      : 992 KB/s
```

### 5.3 Detalle del MJPEG

```
ventana medida      : 12.19 s
conexiones HTTP     : 1
time-to-first-frame : 14.0 ms
frames recibidos    : 74 (6.07 fps)
frames producidos   : 74 (6.07 fps)
entregados/producidos: 100 %
separación          : media 164.3 ms | p95 211.2 ms | max 231.8 ms
tamaño medio        : 308.5 KB
ancho de banda      : 1 874 KB/s
```

### 5.4 CPU (ventanas A/B/B/A)

```
POLLING  cpu/s=31.3  fps_motor=7.56  entregados=39 (3.17 fps)
MJPEG    cpu/s=31.4  fps_motor=6.75  entregados=82 (6.83 fps)
MJPEG    cpu/s=31.3  fps_motor=6.99  entregados=85 (6.99 fps)
POLLING  cpu/s=31.3  fps_motor=5.74  entregados=38 (3.16 fps)

POLLING  cpu/s medio=31.3 | entregados=3.17 fps | 48 % de lo producido
MJPEG    cpu/s medio=31.3 | entregados=6.91 fps | 101 % de lo producido
```

La CPU es **idéntica**: ~31.3 s de CPU por segundo de pared (≈31 hilos de
ONNX, el motor). La diferencia entre transportes queda por debajo del ruido
(< 0.5 %). **El transporte no es el cuello de botella.**

### 5.5 Identidad

Con el clip desde el segundo 36, el motor confirmó **8 de los 9**:

```
ASISTENCIA_NORMAL unc_carlos  sim=0.41
ASISTENCIA_NORMAL unc_alexa   sim=0.45
ASISTENCIA_NORMAL unc_bob     sim=0.57
ASISTENCIA_NORMAL unc_dani    sim=0.51
ASISTENCIA_NORMAL unc_jose    sim=0.66
ASISTENCIA_NORMAL unc_jesus   sim=0.41
ASISTENCIA_NORMAL unc_fran    sim=0.44
ASISTENCIA_NORMAL unc_kate    sim=0.46
```

Falta `kevin` en esa ventana. En las sesiones 97 y 98 (pruebas del usuario) hay
**9 confirmados** y 2–3 intrusos.

Frame extraído del stream MJPEG (media del segundo 36): `carlos` en verde con
0.55, `alexa` en amarillo como "posible" (0.32), y varios desconocidos en rojo.

---

## 6. Veredicto

**MJPEG es claramente mejor para este caso de uso.**

| Criterio | Ganador | Por qué |
|---|---|---|
| Fluidez | **MJPEG** | 6.91 fps vs 3.17 fps (100 % vs 48 % de los frames) |
| Latencia | **MJPEG** | ~0 ms vs 0–300 ms (media 150 ms) |
| CPU | Empate | 31.3 vs 31.3 cpu/s |
| Ancho de banda | Polling | 992 KB/s vs 1 874 KB/s (~1.9×) |
| Complejidad del cliente | **MJPEG** | Una conexión, sin `setInterval` ni cache-buster |
| Tolerancia a caídas | Polling | El polling reintenta solo; el stream no (ver §7) |

El coste de MJPEG es **ancho de banda**: envía todos los frames (7 fps × 308 KB)
frente a los 3.3 del polling. En localhost es irrelevante; en red, la diferencia
es entre un video fluido y uno entrecortado. A cambio, el polling **pierde la
mitad de lo que produce el motor** y añade hasta 300 ms de retraso.

---

## 7. Defectos y riesgos encontrados

### 7.1 El stream no reconecta solo

En `live.html`, `streamConectado` se pone en `true` al asignar `frame.src` y solo
se reinicia cuando la captura se detiene. **No hay `onerror` en el `<img>`**: si
la conexión MJPEG se cae con la captura activa, el navegador queda congelado
hasta pulsar "Iniciar" de nuevo.

**Fix propuesto (2 líneas):**

```js
frame.onerror = () => { streamConectado = false; };
```

`pintarEstado` corre cada segundo y volvería a llamar a `conectarRecepcionVideo`,
que reconecta al ver `streamConectado === false`.

### 7.2 Un hilo del servidor ocupado

MJPEG mantiene una petición HTTP abierta durante toda la captura. Con
`runserver` (threaded) no hay problema; en producción con gunicorn hay que usar
`--worker-class gthread --threads N`, o el stream bloquearía un worker entero.

### 7.3 Ancho de banda

A 1080p y calidad JPEG 80, cada frame pesa ~300 KB. Opciones:

- bajar la calidad a ~65 (≈ mitad de peso, diferencia visual mínima),
- enviar a 960×540 (≈ 4× menos píxeles),
- limitar el stream a ~10 fps (el motor ya va a 7).

### 7.4 Modo de prueba

El selector "Modo Video (A/B)" de `/en-vivo/` es útil para comparar, pero no
debería quedar en la versión final; el MJPEG debería ser el único camino (o
mantener el polling como respaldo automático).

---

## 8. El cuello de botella real

Ninguno de los dos transportes cambia el límite del motor: **~6.7 fps
consumiendo ~31 hilos** (≈ 4.5 s de CPU por frame con 2–6 caras). La RTX 4090
sigue sin usarse (`onnxruntime` build de CPU).

El salto de 7 a 30+ fps viene de `onnxruntime-gpu`, no del transporte. Ese
benchmark está en `docs/roadmap_version_2.md` y exige revalidar la regresión de
identidad.

---

## 9. Pasos propuestos

| # | Paso | Prioridad |
|---|---|---|
| 1 | Añadir `frame.onerror` y reconexión automática del stream | Hecho (2026-11-05) |
| 2 | Dejar MJPEG como único modo (o respaldo automático) | Hecho (2026-11-05) |
| 3 | Registrar esta medición en la documentación | Hecho (este archivo) |
| 4 | Actualizar `AGENTS.md`: la sesión demo 94 ya no existe (ahora 97 y 98) | Hecho |
| 5 | Bajar calidad JPEG a 65 o resolución a 960×540 si la red lo pide | P2 |
| 6 | Documentar `gthread`/`--threads` para producción | P2 |
| 7 | Benchmark CPU vs GPU (el verdadero salto de fps) | P2 (V2) |

---

## 10. Cómo reproducirlo

```bash
# 1. Clip desde el segundo 36
ffmpeg -y -loglevel error -ss 36 -t 25 -i data/unc/salon/VID_20260907_102851.mp4 \
  -an /tmp/opencode/clip36.mp4

# 2. Servidor
venv/bin/python manage.py runserver 127.0.0.1:8011 --noreload

# 3. Login (obtener cookie y CSRF)
curl -s -c /tmp/cj http://127.0.0.1:8011/login/ -o /dev/null
CSRF=$(awk '$6=="csrftoken" {print $7}' /tmp/cj)
curl -s -b /tmp/cj -c /tmp/cj -e http://127.0.0.1:8011/login/ \
  -d "username=admin&password=admin123&csrfmiddlewaretoken=$CSRF" \
  -o /dev/null http://127.0.0.1:8011/login/

# 4. Iniciar captura (sin sesión: solo visualiza)
CSRF=$(awk '$6=="csrftoken" {print $7}' /tmp/cj)
curl -s -b /tmp/cj -H "X-CSRFToken: $CSRF" -H 'Content-Type: application/json' \
  -d '{"fuente":"/tmp/opencode/clip36.mp4","sesion_id":null}' \
  http://127.0.0.1:8011/api/en-vivo/iniciar/
```

### 10.1 Medir polling

```python
import time, requests, statistics, hashlib
s = requests.Session(); s.cookies.update({"sessionid": "..."})
DUR = 12.0
t0 = time.time(); lat = []; hashes = []
while time.time() - t0 < DUR:
    a = time.time()
    r = s.get(f"http://127.0.0.1:8011/api/en-vivo/frame/?t={a}")
    lat.append(time.time() - a)
    hashes.append(hashlib.md5(r.content).hexdigest())
    time.sleep(max(0, 0.3 - (time.time() - a)))
print("fps entregados:", len(lat) / DUR)
print("latencia media:", statistics.mean(lat) * 1000, "ms")
print("duplicados:", sum(1 for i in range(1, len(hashes)) if hashes[i] == hashes[i-1]))
```

### 10.2 Medir MJPEG

```python
import time, requests
s = requests.Session(); s.cookies.update({"sessionid": "..."})
DUR = 12.0; t0 = time.time(); n = 0; buf = b""
r = s.get(f"http://127.0.0.1:8011/api/en-vivo/stream/?t={t0}", stream=True)
while time.time() - t0 < DUR:
    trozo = r.raw.read(65536)
    if not trozo: break
    buf += trozo
    n += buf.count(b"--frame")
    buf = buf[-16:]
print("fps entregados:", n / DUR)
```

### 10.3 Medir CPU

```python
def cpu_seg(pid):
    with open(f"/proc/{pid}/stat") as f:
        c = f.read().rsplit(") ", 1)[1].split()
    return (int(c[11]) + int(c[12])) / 100.0

c0 = cpu_seg(PID); ...; c1 = cpu_seg(PID)
print("cpu/s:", (c1 - c0) / (t1 - t0))
```

---

## 11. Limitaciones del experimento

- **Un solo video y una sola máquina** (i9-14900KF, localhost). El ancho de banda
  y la latencia reales dependen de la red del aula.
- **La CPU del transporte está por debajo del ruido** del motor; por eso se
  reporta "idéntica" y no un porcentaje fino.
- El clip se repite para la prueba de CPU; el contenido (número de caras) varía
  entre ventanas, lo que explica el rango de 5.7–7.6 fps del motor.
- El frame de 300 KB depende de la calidad JPEG 80 y de 1080p.
- La sesión temporal 99 se borró al terminar; las sesiones 97 y 98 son del
  usuario y no se tocaron.

---

## 12. Conclusión en una frase

**El MJPEG entrega el doble de frames con la misma CPU y sin latencia; el polling
solo gana en ancho de banda. El límite real sigue siendo el motor en CPU
(~7 fps), y el siguiente salto es la GPU.**

# Entorno de cómputo y ejecución

**Fecha:** 2026-10-26
**Alcance:** todas las ejecuciones del proyecto: motor, experimentos, demos, tests
y flujos web. Este documento responde a una pregunta concreta: **¿con qué hardware
y software se obtuvieron los resultados?**

---

## Resumen

| | |
|---|---|
| CPU | Intel Core **i9-14900KF** (24 núcleos / 32 hilos, hasta 6.0 GHz) |
| RAM | **62 GiB** |
| GPU | **NVIDIA GeForce RTX 4090** (24 GB) — presente, **no utilizada** |
| Aceleración | **CPU únicamente** (`onnxruntime` build de CPU, `faiss-cpu`) |
| SO | Ubuntu 24.04.5 LTS, kernel 6.8.0-147 |
| Python | 3.12.7 (`venv/`) |
| Rendimiento | **~7 fps** en video 1080p (rango medido 6.6-7.5) |
| Tiempo por video | Video 1 (~71 s) ≈ 5 min · Video 3 (~124 s) ≈ 8-9 min |

---

## 1. Hardware

| Componente | Valor |
|---|---|
| CPU | Intel Core i9-14900KF |
| Núcleos / hilos | 24 núcleos físicos, 32 hilos (8 P-cores + 16 E-cores) |
| Frecuencia | 800 MHz - 6.0 GHz |
| RAM | 61.8 GiB (~2.7 GiB libres, 40 GiB en caché al medir) |
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB |
| Driver NVIDIA | 595.99.02 (CUDA 13.2 a nivel de driver) |
| Almacenamiento | NVMe (`/dev/nvme0n1p5`), 870 GB totales |
| SO | Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic |

La GPU está operativa (no es una máquina sin GPU). El pipeline simplemente no la
usa, por la razón de la sección 5.

---

## 2. Software base

| Componente | Versión | Uso |
|---|---|---|
| Python | 3.12.7 | Intérprete del `venv/` |
| PostgreSQL | 16.15 | Base de datos relacional y vectorial |
| pgvector | 0.6.0 | Tipo `vector(512)`, índice HNSW |
| ffmpeg | 6.1.1 | Versiones `_web.mp4` de los videos |
| psycopg2-binary | 2.9.12 | Acceso a PostgreSQL desde los scripts |
| Django | 6.1 | Aplicación web |

> La base de datos y las credenciales están descritas en `docs/base_datos.md`.
> Se leen de `CV_DB_*` y `DJANGO_*`; los valores por defecto son los de desarrollo.

---

## 3. Stack de visión

| Componente | Versión | Rol |
|---|---|---|
| `uniface` | 3.3.0 | Detección, reconocimiento y anti-spoofing |
| `onnxruntime` | **1.24.4 (CPU)** | Ejecuta los modelos ONNX |
| `faiss-cpu` | 1.13.2 | Índice vectorial (`IndexFlatIP`) |
| `opencv-python` | 4.13.0.92 | Video, dibujo, CLAHE, codificación |
| `numpy` | 2.4.4 | Vectores y matrices |
| `supervision` | 0.27.0.post2 | ByteTrack y estructuras `Detections` |

Los modelos son **ONNX**, no PyTorch. No se instaló `torch`, `torchvision` ni
`faiss-gpu`.

---

## 4. Modelos ONNX

Se descargan en el primer uso y quedan cacheados en `~/.uniface/models/`
(**43 MB** en total):

| Archivo | Tamaño | Uso |
|---|---|---|
| `retinaface_mnet_v2.onnx` | 12 MB | Detección de rostros (pipeline principal) |
| `arcface_mnet.onnx` | 13 MB | Embedding de 512 dimensiones |
| `minifasnet_v2.onnx` | 1.7 MB | Anti-spoofing |
| `scrfd_10g.onnx` | 17 MB | Variante SCRFD (solo A/B, no la usa la web) |

La carga inicial de los tres modelos del pipeline principal tarda **~0.1 s** una
vez cacheados. La primera ejecución en una máquina nueva descarga los 43 MB.

---

## 5. CPU, no GPU

`uniface` selecciona el proveedor de ejecución automáticamente:

```python
# uniface/onnx_utils.py
if 'CUDAExecutionProvider' in available:
    providers.append('CUDAExecutionProvider')
    Logger.info('CUDA acceleration enabled (NVIDIA GPU)')
providers.append('CPUExecutionProvider')
```

En este entorno, `onnxruntime` reporta:

```python
onnxruntime.get_available_providers()
→ ['AzureExecutionProvider', 'CPUExecutionProvider']
```

Es decir: **no hay `CUDAExecutionProvider`**, así que todo cae en CPU. La causa es
que `requirements.txt` instala **`onnxruntime`** (rueda de CPU). La variante con
GPU es otro paquete, **`onnxruntime-gpu`**, que exige CUDA 12.x y cuDNN 9
compatibles.

**Consecuencia:** detección, embedding y anti-spoofing corren en CPU. La búsqueda
FAISS también es CPU (`faiss-cpu`), aunque con 9 vectores es irrelevante.

**Los resultados no cambian por esto.** Los modelos son los mismos; la GPU solo
acelera la ejecución. Puede haber diferencias en el último decimal por el orden de
las operaciones en punto flotante, pero el umbral 0.30, la matriz de similitudes,
la ROC y la regresión de identidad se calcularon y validaron con este mismo
entorno.

---

## 6. Rendimiento medido

Medición sobre **200 frames** de `data/unc/salon/VID_20260907_102851.mp4`
(1920×1080, 30 fps, 3.4 rostros por frame), separando las etapas:

| Etapa | Tiempo | % del total | Por unidad |
|---|---|---|---|
| Detección (RetinaFace) | 9.4 s | 35 % | **47 ms/frame** |
| Embedding (CLAHE + ArcFace) | 16.6 s | 62 % | **24 ms/cara** |
| Búsqueda FAISS | 0.02 s | ~0 % | **0.02 ms/cara** |
| Overhead (lectura, emparejamiento) | 0.6 s | 3 % | 3 ms/frame |
| **Total** | **26.6 s** | 100 % | **7.52 fps** |

Esta medición aísla detección, embedding y búsqueda; **no incluye el tracker ni el
anti-spoofing**. El ritmo del pipeline completo es algo menor (sección siguiente).

El embedding domina porque hay ~3.4 caras por frame: 3.4 × 24 ms ≈ 82 ms por
frame solo en reconocimiento. La detección es el segundo costo.

**Ritmo real del pipeline completo:** entre **6.6 y 7.5 fps**, según el tramo y
cuántas caras haya en cuadro. La diferencia con los 7.52 fps de la tabla es el
anti-spoofing y el tracker completo, que la medición por etapas no incluye.

---

## 7. Tiempos por video

Con ~7 fps de referencia:

| Video | Duración | Frames | Tiempo de proceso |
|---|---|---|---|
| Video 1 (`VID_20260907_102851.mp4`) | 70.8 s | ~2 124 | **~5 min** |
| Video 2 (`VID_20260907_105622.mp4`) | — | — | ~4-5 min (similar) |
| Video 3 completo (`video5096333890499381162.mp4`) | 124 s | ~3 720 | **~8-9 min** |
| Video 3 en 11 partes | 124 s | 3 808 | **~9 min** |

La generación de los videos de demostración tarda más porque además dibuja el
panel y codifica el MP4.

---

## 8. Implicaciones

- **Tiempo real:** a ~7 fps, el flujo en vivo no alcanza una cámara de 30 fps. El
  servicio en vivo procesa más lento que la cámara y la imagen se retrasa. Para
  tiempo real estricto haría falta GPU o bajar la resolución del stream.
- **Reproducibilidad:** cualquier máquina con CPU similar y las mismas versiones
  debe dar FPS comparables. Los resultados de identidad no dependen del hardware.
- **Experimentos:** usan el mismo motor y el mismo CPU; los tiempos publicados en
  los informes son de este entorno.
- **Base de datos:** pgvector corre en PostgreSQL local, sin GPU; el índice HNSW
  se usa solo en `importar_centroides` y en las consultas SQL de diagnóstico.

---

## 9. Cómo reproducir la medición

```bash
cd /home/valentina2/Desktop/Folds/VisionProjects/face-recognition-college

venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0, '.')
import cv2
from scripts.asistencia_unificada_optimizada import MotorAsistencia

motor = MotorAsistencia(registrar_csv=False)
motor.tamano_minimo_rostro = 20

cap = cv2.VideoCapture("data/unc/salon/VID_20260907_102851.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, int(40 * 30))  # segundo 40

inicio = time.time()
frames = 0
for _ in range(200):
    ok, frame = cap.read()
    if not ok:
        break
    motor.procesar_frame(frame)
    frames += 1
cap.release()

print(f"{frames / (time.time() - inicio):.2f} fps")
PY
```

Este fragmento incluye el pipeline completo (detección, tracker, decisión y
liveness), por lo que da el ritmo real, no solo el de las tres etapas aisladas.

Para el desglose por etapa, instrumentar `detector.detect`, `embedding_facial` y
`indice_faiss.search` con `time.time()` alrededor de cada llamada, como se hizo
para la tabla de la sección 6.

---

## 10. Preparar un entorno equivalente

```bash
# 1. Dependencias de Python
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt pytest faiss-cpu

# 2. PostgreSQL + pgvector
sudo apt install postgresql-16 postgresql-16-pgvector
sudo -u postgres psql -c "CREATE DATABASE cv_college_db;"
sudo -u postgres psql -c "CREATE USER cv_college_user WITH PASSWORD '...';"
sudo -u postgres psql -d cv_college_db -c "CREATE EXTENSION vector;"

# 3. Esquema y datos
venv/bin/python manage.py migrate
venv/bin/python manage.py importar_centroides

# 4. Tests de Django (una sola vez)
sudo -u postgres psql -c "ALTER ROLE cv_college_user CREATEDB;"
sudo -u postgres psql -d template1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

> `requirements.txt` omite `pytest` y `faiss-cpu`; hay que instalarlos aparte.
> Las credenciales se configuran con `CV_DB_*` y `DJANGO_SECRET_KEY`; los valores
> por defecto son los de desarrollo.

---

## 11. Lo que NO se usó

| Componente | Estado |
|---|---|
| GPU / CUDA | Presente, **no utilizada** |
| `onnxruntime-gpu` | No instalado |
| `faiss-gpu` | No instalado |
| PyTorch / `torchvision` | No instalado (los modelos son ONNX) |
| Docker | No usado |
| Entrenamiento | No hubo: se usan modelos preentrenados |

---

## 12. Disco

| Ruta | Tamaño |
|---|---|
| `data/` (videos, frames, pickles) | 1.6 GB |
| `experiments/unc/demo_videos/` | 200 MB |
| `~/.uniface/models/` | 43 MB |
| Disponible en `/` | ~12 GB (99 % usado) |

El disco está ajustado. Antes de generar más videos de demostración conviene
liberar espacio o mover los masters a otro volumen.

---

## 13. Tests ejecutados en esta máquina

| Suite | Cantidad | Tiempo |
|---|---|---|
| `pytest -m unit` | 76 | ~1 s |
| `manage.py test` | 36 | ~34 s (incluye crear la base de test y migrar) |
| `manage.py check` | — | <1 s |

Los tests de Django usan PostgreSQL (no SQLite) porque el tipo `vector` no existe
en SQLite.

---

## 14. Migrar a GPU (opcional, no hecho)

Si se quiere acelerar:

1. `venv/bin/pip uninstall onnxruntime`
2. `venv/bin/pip install onnxruntime-gpu`
3. Verificar que `get_available_providers()` incluya `CUDAExecutionProvider`.
4. `uniface` lo selecciona solo; no hay que cambiar código.
5. **Pasar la regresión de identidad**:
   `venv/bin/python experiments/unc/regresion_identidad.py`
   (debe seguir dando 11 OK + 1 baseline; sale con 1 si hay fallos nuevos).
6. Volver a medir FPS y comparar.

Riesgo: el redondeo numérico de CUDA puede mover algún caso que esté justo en el
umbral 0.30. La regresión es obligatoria antes de aceptar el cambio. `faiss-cpu`
seguiría en CPU; con 9 vectores no compensa cambiarlo.

---

## 15. Referencias

| Tema | Documento |
|---|---|
| Documentación del motor | `docs/motor.md` (capítulo 1) |
| Base de datos | `docs/base_datos.md` |
| Comandos | `docs/comandos.md` |
| Regresión de identidad | `experiments/unc/regresion_identidad.py` |
| Incidente de identidad | `docs/analisis/unc/09_incidente_identidad_dani.md` |

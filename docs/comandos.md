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

## 6. Probar con cámara real

Esta máquina **no tiene cámara** (`/dev/video*` no existe), así que la prueba con
hardware hay que hacerla en el aula:

```bash
# 1. Cámara como fuente (0 = cámara por defecto)
venv/bin/python scripts/asistencia_unificada_optimizada.py --fuente 0

# 2. O en la web, con la sesión abierta
venv/bin/python manage.py runserver
# http://localhost:8000/en-vivo/  ->  Fuente: 0  ->  Iniciar
```

Lista de verificación:

- [ ] La cámara abre sin error (`VideoCapture(0)`).
- [ ] El frame llega al navegador por el stream MJPEG (sin parpadeo).
- [ ] Los 9 registrados aparecen con su nombre y similitud.
- [ ] Al menos uno queda en amarillo (`posible`) si está de perfil o con gorra.
- [ ] Un desconocido genera una fila `INTRUSO` con su captura.
- [ ] Al terminar, `/sesiones/<id>/` lista presentes, ausentes y la banda de revisión.

Ayudas: buena luz frontal, la cámara a la altura de los ojos y a 2–3 m del grupo;
si la imagen sale pequeña, baja el tamaño del frame o la distancia.

---

## 7. Tests

```bash
venv/bin/python -m pytest -m unit          # 91 tests, rápido
venv/bin/python manage.py test              # 48 tests Django
venv/bin/python manage.py check
```

---

## 8. Regresión de identidad (el motor no puede cambiar sin esto)

```bash
venv/bin/python experiments/unc/regresion_identidad.py
```

Sale con código 1 si hay fallos nuevos.

---

## 9. Métricas consolidadas para la tesis

```bash
venv/bin/python experiments/unc/metricas/metricas.py
```

Una sola ejecución lee los CSV de las fases 0, 4, 6 y 7 y regenera
`experiments/unc/metricas/salida/`:

- `identidad/`: ROC y PR exactas, AUC, AP, EER, d', Youden, tabla de 181
  umbrales, matriz de confusión, métricas por estudiante, top-k, intervalos de
  confianza, error por ancho y accesorios, multiframe y 6 gráficas.
- `deteccion/`: tasas de detección, IoU, estrés, tamaños y filtro.
- `asistencia/`: cobertura de los videos 1 y 3, matriz de sesión, intrusos,
  barrido de la regla v2, fragmentación de pistas y spoofing.
- `rendimiento/`: FPS, latencia por etapa y CPU/GPU.
- `resumen.md`: informe con todos los números y su archivo fuente.

No usa scikit-learn ni pandas; solo NumPy y matplotlib. La semilla del bootstrap
es fija (`20261105`). El detalle de cada métrica y sus límites está en
`experiments/unc/metricas/README.md`.

---

## 10. Prueba de liveness con webcam (foto en papel / teléfono)

```bash
venv/bin/python experiments/unc/prueba_liveness.py --fuente 0
```

Se abre la cámara y se marca el escenario con el teclado antes de presentar el
rostro:

- `1` = cara real
- `2` = foto impresa (papel)
- `3` = foto en el teléfono
- `4` = otro ataque
- `e` = cerrar el escenario actual y pausar
- `q` = terminar y escribir el informe

Cada frame se evalúa con el mismo MiniFASNet del motor; si hay identidad
cargada, también se reconoce a quién pertenece el rostro (ArcFace + pgvector).
La corrida queda en `experiments/unc/salida_liveness/run_FechaHora/` con
`liveness_frames.csv`, `liveness_segmentos.csv`, `umbrales.csv`,
`matriz_confusion.csv`, `capturas/` e `informe_liveness.md`.

Para el ataque con teléfono hay fotos listas de tres estudiantes en
`experiments/unc/salida_liveness/fotos_telefono/` (`dani`, `alexa`, `bob`).
Pasalas al teléfono y mostralas a pantalla completa.

Para re-evaluar una grabación sin cámara:

```bash
venv/bin/python experiments/unc/prueba_liveness.py --video prueba.mp4 --etiqueta telefono --sin-ventana
```

Para regenerar el informe de una corrida ya guardada (sin cámara ni modelos):

```bash
venv/bin/python experiments/unc/prueba_liveness.py --reinforme experiments/unc/salida_liveness/run_20261110_091811
```

El informe incluye una matriz **Por sujeto mostrado** que separa al operador
de la foto del estudiante; si el operador queda en cuadro durante el ataque,
esa es la lectura correcta (la matriz cruda mezcla ambos).

`--umbral 0.90` es el del motor; la tabla del informe también muestra 0.50 y
0.75 para ver la sensibilidad. `--sin-identidad` acelera la corrida si solo
interesa el anti-spoofing.

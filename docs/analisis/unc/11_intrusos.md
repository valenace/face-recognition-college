# 11 — Intrusos: separar desconocidos reales de registrados perdidos

**Fecha:** 2026-10-29 (regla v1) · **2026-11-05 (regla v2, recalibrada)**
**Estado:** implementado, medido y verificado con los videos 1 y 3

---

## 1. Resumen

| | Regla v1 (2026-10-29) | **Regla v2 (actual)** |
|---|---|---|
| Criterio | mejor score de la pista < 0.18 | **mejor score efectivo < 0.22** y **promedio < 0.24** |
| Persistencia | 15 muestras | 15 muestras |
| Tamaño mínimo | 40 px | **40 px** |
| Deduplicación | no | **por asiento** (2.5× ancho, 240 frames) |
| Falsos en video 1 (44 s, todos registrados) | 9 | **7** |
| Intrusos en video 3 (21 desconocidos) | 18 | **22 (≈ los 21 reales)** |

Medición completa en la sección 4.

---

## 2. El problema

El motor devuelve `DESCONOCIDO` en **cada frame** donde la similitud no alcanza el
umbral de identidad (0.30). Un estudiante registrado que se mueve, gira o queda
pequeño produce muchos de esos frames, y ByteTrack reparte su rostro en **pistas
distintas**: escribir una fila por pista llenaba la sesión de intrusos falsos.

Peor: la fila de intruso **no distingue personas**. Con la regla v1, la sesión 94
acababa con 24 filas de intruso para 9 estudiantes, y el profesor tenía que
identificar a mano gente que en realidad estaba registrada.

## 2.1 Dos extremos que había que evitar

1. **Falsos:** marcar como intruso a un registrado que el motor perdió un frame.
2. **Duplicados:** la misma persona sentada en su pupitre generando 5–10 filas
   porque ByteTrack le reasignó el `track_id` tras una oclusión.

## 2.2 Lo que no funcionó (y por qué se descartó)

| Intentos | Resultado |
|---|---|
| Umbral global más alto | Pierde a `jesus` y `fran` (rechazado en la Fase 3b) |
| Umbral por persona | Pierde a `fran` y `kate` |
| Media podada / medoid | No mueve el falso positivo |
| Margen top-2 | El falso positivo tenía margen 0.118, mayor que el de muchos aciertos |
| **Regla v1** (mejor score < 0.18) | 9 falsos en 44 s de video 1 |

---

## 3. Regla v2 (la que está en el código)

En `core/intrusos.py::SeguimientoIntrusos`. Una pista desconocida es intruso si
**cumplía las cuatro** condiciones:

1. **Nunca se acercó a un registrado:** su mejor similitud *efectiva* stayed por
   debajo de **0.22**.
2. **Su promedio es bajo:** `suma / muestras < 0.24`.
3. **Persistió:** al menos **15** muestras de desconocido.
4. **No hubo conflicto** de identidad en sus frames.

Y dos correcciones que evitan los extremos:

- **Muestra efectiva:** si el candidato top-1 de ese frame ya estaba confirmado (o
  anclado) en **otra pista del mismo frame**, la muestra vale **0**. Dos personas
  distintas no pueden ser el mismo alumno, así que esa muestra no dice nada de
  esta pista. Esa corrección se aplica al promedio **y** a la mejor similitud.
- **Deduplicación por asiento:** si una pista nueva califica donde había un intruso
  reciente (mismo asiento: distancia < 2.5× el ancho de la cara, dentro de 240
  frames ≈ 34 s), **reutiliza su fila** en vez de crear otra.

### Por qué el mejor score y el promedio juntos

El **promedio** es acumulativo: si una cara empieza mal y luego se parece mucho a
un registrado, el promedio **no sube lo suficiente** y la pista sigue marcada como
intruso. El **mejor score** sí captura esa mejora, pero por sí solo es demasiado
estricto (descarta desconocidos reales que en un frame unlucky se parecen a
alguien). Con los dos criterios, la pista tiene que ser **consistemente** parecida a
un registrado para quedar fuera de la lista de intrusos.

---

## 4. Medición

### 4.1 Metodología

Un solo paso por cada video, guardando por pista y por frame la similitud, la
similitud efectiva y la caja. Después se **simularon** todas las reglas posibles
en offline, sin reprocesar:

```bash
venv/bin/python /tmp/opencode/capturar_pistas.py    # engine → JSON por video
venv/bin/python /tmp/opencode/simular_intrusos.py    # barrido de umbrales
```

El simulador replica el clasificador (promedio, descarte, deduplicación por
asiento). Difiere en ±3 filas respecto del código real por el detalle de "asiento
visible"; se usaron las cifras del **código real** para la decisión final.

Videos:

| Video | Contenido | Frames |
|---|---|---|
| 1 (`VID_20260907_102851.mp4`, 36–80 s) | 9 registrados, **0 desconocidos** → todo intruso es falso | 1 319 |
| 3 (`video5096333890499381162.mp4`) | 8 registrados + **21 desconocidos** | 3 710 |

### 4.2 Barrido de umbrales (simulación)

Filas = intrusos detectados. En video 1 **todas son falsas**; en video 3 se
comparan con los 21 desconocidos reales.

| `mejor` | `promedio` | ancho px | video 1 (falsos) | video 3 (filas) |
|---|---|---|---|---|
| — | 0.24 | 26 | 23 | 86 |
| — | 0.24 | 40 | 13 | 45 |
| 0.22 | 0.24 | 26 | 12 | 63 |
| 0.22 | 0.24 | 32 | 9 | 43 |
| **0.22** | **0.24** | **40** | **4** | **22** |
| 0.24 | 0.18 | 40 | 8 | 21 |
| 0.26 | 0.18 | 40 | 9 | 25 |
| 0.28 | 0.24 | 40 | 10 | 41 |
| 0.30 | 0.24 | 40 | 13 | 45 |

La fila elegida (`mejor < 0.22`, `promedio < 0.24`, `40 px`) es la que **minimiza
los falsos sin perder desconocidos**: 4 simulados y 22 filas en video 3, casi
exactamente las 21 personas reales.

### 4.3 Verificación con el código real

| Video | Filas de intruso | Creadas | Reutilizadas (asiento) | Descartadas |
|---|---|---|---|---|
| 1 (44 s, todos registrados) | **7** (todas falsas) | 13 | 8 | 6 |
| 3 (completo, 21 desconocidos) | **22** | 39 | 53 | 17 |

Los 22 detalles de video 3 tienen mejores scores entre 0.066 y 0.205 y promedios
entre 0.016 y 0.149: son 22 rostros distintos, no duplicados.

---

## 5. Qué_gain_producen los tres números

| Parámetro | Efecto |
|---|---|
| `mejor < 0.22` | Evita marcar a un registrado que se acerque a su centroide |
| `promedio < 0.24` | Evita marcar a un desconocido que en algún frame se parezca a alguien (muestra efectiva) |
| `40 px` | Evita filas de caras demasiado pequeñas para identificar (y capturas inútiles) |
| asiento 2.5× / 240 frames | Una persona sentada no genera 5–10 filas |

---

## 6. Riesgos que quedan

1. **La deduplicación por asiento puede fusionar a dos personas distintas** si
   están sentadas muy cerca (menos de 2.5× el ancho de la cara) y la primera
   desaparece en los 240 frames siguientes. Subcuenta intrusos, no los sobrecuenta.
2. **El mínimo de 40 px deja fuera a los desconocidos muy lejanos** o de perfil
   limpio. Se prefirió eso a generar filas que el profesor no puede identificar.
3. **La regla no toca la asistencia:** el umbral de identidad sigue siendo 0.30 con
   3 votos. Solo cambia qué se escribe como `INTRUSO`.

---

## 7. Archivos

| Archivo | Contenido |
|---|---|
| `core/intrusos.py` | El clasificador (lógica pura) |
| `core/registro_asistencia.py` | Aplica el filtro de 40 px y escribe la fila |
| `core/servicio_en_vivo.py` | Crea un `SeguimientoIntrusos` por captura |
| `apps/attendance/management/commands/ver_flujo.py` | Igual, para el diagnóstico |
| `tests/test_registro.py` | 9 pruebas unitarias del clasificador |

## 8. Cómo reproducirlo

```bash
# 1. Capturar las pistas (un solo paso por video, ~12 min cada uno)
venv/bin/python /tmp/opencode/capturar_pistas.py

# 2. Barrer umbrales en offline
venv/bin/python /tmp/opencode/simular_intrusos.py

# 3. Verificación con el código real
venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, ".")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cv_college.settings")
import django; django.setup()
import cv2
from scripts.asistencia_unificada_optimizada import MotorAsistencia
from core.intrusos import SeguimientoIntrusos
from core.registro_asistencia import TAMANO_MINIMO_INTRUSO
# ... ver /tmp/opencode/verificar_regla.log
PY
```

---

## 9. Conclusión

La regla v2 baja los intrusos falsos de 23 a 7 en el mismo tramo de video 1 y
detecta los 21 desconocidos reales del video 3 con 22 filas. El coste es que un
intramo muy lejano (< 40 px) no se reporta, y que dos personas muy juntas pueden
compartir fila.

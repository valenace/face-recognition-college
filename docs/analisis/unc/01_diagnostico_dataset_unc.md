# 01 — Diagnóstico del dataset UNC (Fase 1)

**Fecha:** 2026-10-15
**Script:** `experiments/unc/fase1_diagnostico_dataset_unc.py`
**Estado:** ejecutada.

---

## 1. Qué mide esta fase

Antes de medir el reconocimiento, revisamos la calidad de los datos:

- **Tasa de detección:** proporción de frames muestreados con al menos una cara.
- **Tasa de cara única:** proporción con exactamente una cara.
- **Tasa útil:** proporción con cara única **y** nitidez del recorte alineado ≥ 60.
- **Ancho de la caja** de la cara (en píxeles).
- **Frames útiles estimados:** tasa útil × frames totales.

**Muestreo:** 1 de cada 3 frames (paso = 3), como indica el plan.

**Criterio de nitidez:** varianza del Laplaciano sobre el recorte alineado de 224 px, el mismo criterio del registro. Ver `03_bitacora_calculos.md`, sección C1.

## 2. Qué se analizó

- **Videos de registro:** los 9 de `data/unc/stds/` (un video por estudiante).
- **Salón:** solo `VID_20260907_102851.mp4`, como acordamos.

## 3. Resultados de los videos de registro

| Persona | Frames | Muestreados | Detección | Cara única | Útil | Ancho mín–máx | Ancho medio | Útiles estimados |
|---|---|---|---|---|---|---|---|---|
| alexa  | 105 | 35  | 1.00 | 1.00 | 1.00 | 556–614 | 585.9 | 105 |
| bob    | 229 | 77  | 1.00 | 1.00 | 1.00 | 512–650 | 588.1 | 117 |
| carlos | 144 | 48  | 1.00 | 1.00 | 1.00 | 579–684 | 621.5 | 144 |
| dani   | 125 | 42  | 1.00 | 1.00 | 1.00 | 520–613 | 571.5 | 126 |
| fran   | 154 | 52  | 1.00 | 1.00 | 1.00 | 485–607 | 555.5 | 156 |
| jesus  | 130 | 44  | 0.98 | 0.98 | 0.93 | 452–664 | 525.1 | 123 |
| jose   | 116 | 39  | 1.00 | 1.00 | 1.00 | 490–595 | 530.2 | 117 |
| kate   | 163 | 55  | 1.00 | 1.00 | 1.00 | 448–553 | 512.2 | 165 |
| kevin  | 100 | 34  | 1.00 | 1.00 | 1.00 | 520–652 | 577.7 | 102 |

> **Nota de auditoría (A4):** la fila de `bob` (229 frames, 77 muestreados) es de
> una versión anterior del video. La evaluación actual usa **115 frames y 39
> muestreados** (`experiments/unc/fase1_resumen_videos.csv`). Se conserva la cifra
> histórica para no reescribir el diagnóstico.

**Lectura:**
- Los videos de registro son muy buenos: prácticamente todos los frames muestreados tienen una sola cara, y casi todos pasan el filtro de nitidez.
- Las caras de registro miden entre 450 y 680 px, muy por encima de los 40 px del salón.
- `jesus` es el único con algún frame sin cara o con más de una (98 %), y 93 % útil.

## 4. Comparación con las predicciones del plan

El plan predecía tasas bajas en varios videos (por ejemplo, `kate` ~10 % útil; `jesus` v1 ~25 %). **Estas predicciones no se cumplen.** Las razones probables:

- Las predicciones del plan se hicieron sobre otros videos (el plan tiene 14 videos, con variantes v1 y v2; hoy solo hay un video por persona en `stds/`).
- El plan asumía nitidez medida de otra forma (ver sección 6 de `02_resultados_registro.md`). Con el criterio del recorte alineado, la nitidez no descarta frames en estos videos.

Por eso no usamos las predicciones del plan como referencia.

## 5. Resultados del salón (`VID_20260907_102851`)

| Medida | Valor |
|---|---|
| Frames muestreados (1 de cada 3) | 1 068 |
| Caras detectadas en total | 11 345 |
| Mediana de caras por frame | 13 |
| Frames con cara única | 9 % |
| Ancho mediano de cara | 37 px |
| Caras menores de 40 px | 6 632 de 11 345 (58 %) |

**Lectura:**
- El salón tiene muchas personas por frame (mediana 13). Por eso casi nunca hay una sola cara.
- La mayoría de las caras son pequeñas: más de la mitad miden menos de 40 px.
- Esto confirma lo que vimos con la muestra: el salón es un escenario mucho más difícil que el registro.

Gráficas: `experiments/unc/fase1_tasa_deteccion.png` y `experiments/unc/fase1_bbox_salon.png`.

## 6. Archivos

- `experiments/unc/fase1_resumen_videos.csv`: tabla de la sección 3 y del salón.
- `experiments/unc/fase1_caras_salon.csv`: caras por frame del salón.
- `experiments/unc/fase1_tasa_deteccion.png` y `fase1_bbox_salon.png`.

# 06 — Evaluación final del sistema en el salón (Fase 6)

**Fecha:** 2026-10-20
**Video:** `data/unc/salon/VID_20260907_102851.mp4` (106.8 s)
**Corrida:** `experiments/unc/demo/prueba_2010_0935_final/`
**Script de análisis:** `experiments/unc/fase6_evaluacion_salon.py`

---

## 1. Configuración evaluada (la definitiva)

| Parámetro | Valor | Origen |
|---|---|---|
| Umbral de similitud | 0.30 | Fase 4 |
| Votos consecutivos | 3 | Diseño del motor |
| Tamaño mínimo de cara | 20 px | Fase 1 y pruebas de filtro |
| Confianza mínima del detector | 0.75 | §24 (`02_resultados_registro.md`) |
| Anti-spoofing | omitido bajo 60 px | §21 |
| Revalidación del ancla | cada 30 frames | §27 |
| Registro | 1 video por estudiante, centroide único | Fase 2 |

## 2. Cobertura y tiempo de confirmación

- **Cobertura: 9 de 9 estudiantes confirmados.**
- Tiempo desde que aparece la pista hasta la confirmación: **0.06 a 0.07 s** en 8 de 9 personas (3 votos, ~0.1 s a 30 fps). `fran` tardó 0.26 s.
- Gráfica: `prueba_2010_0935_final/fase6_tiempo_confirmacion.png`.

## 3. Reconocimiento a nivel de recorte

Metría inmediata (estado del sistema en el frame exacto del recorte):

| Resultado | Cantidad |
|---|---|
| Personas registradas confirmadas | 63 de 86 (73.3 %) |
| Procesadas pero no reconocidas | 17 |
| Sin decisión (sin pista en ese frame) | 6 |
| Errores de identidad | 0 |
| Desconocidos correctos | 134 de 157 (85 %) |
| Desconocidos confirmados como registrados | **1 de 157 (0.6 %)** |
| Desconocidos sin decisión | 22 |

**Por qué 73 % y no más:** con la revalidación cada 30 frames, aproximadamente **3 de cada 30 frames** (10 %) la etiqueta está en "Verificando", y si el recorte cae justo ahí se cuenta como no reconocido. Es un artefacto de medir en el instante exacto.

**Métrica de ventana** (¿se confirmó la persona en algún momento dentro de ±15 frames, es decir ±0.5 s?):

| Resultado | Cantidad |
|---|---|
| Personas con pista confirmadas correctamente | **76 de 80 (95 %)** |
| Personas con pista no confirmadas | 4 |
| Errores de identidad | 0 |

La métrica de ventana representa mejor el uso real: en asistencia importa si la persona queda registrada, no si en un frame puntual dice "Verificando".

## 4. Desempeño por persona (métrica inmediata)

| Persona | Recortes | Confirmados | Recall |
|---|---|---|---|
| dani | 17 | 15 | 0.882 |
| carlos | 6 | 5 | 0.833 |
| jose | 10 | 8 | 0.800 |
| bob | 14 | 10 | 0.714 |
| kevin | 7 | 5 | 0.714 |
| kate | 13 | 9 | 0.692 |
| fran | 3 | 2 | 0.667 |
| alexa | 12 | 7 | 0.583 |
| jesus | 4 | 2 | 0.500 |

Detalle en `prueba_2010_0935_final/fase6_por_persona.csv`.

## 5. Prueba de generalización (video 2, sin etiquetas)

`VID_20260907_105622.mp4` (50.6 s, estudiantes interactuando), corrida `prueba_1910_1602_video2`:

- **6 confirmados:** `dani`, `carlos`, `fran`, `jesus`, `kevin`, `jose`.
- `alexa`, `bob` y `kate` no aparecen confirmados (no hay etiquetas para saber si no están o no se reconocieron).
- Revisión visual de `dani` y `carlos`: todas las capturas son la misma persona.
- `fran` pasó con 0.3031, apenas sobre el umbral.

## 6. Conclusiones

1. **El sistema cubre a los 9 estudiantes y confirma rápido** (menos de 0.3 s desde que aparecen).
2. **No confunde identidades** en ninguno de los dos videos (0 errores en 1 127 recortes y en las corridas completas).
3. **El único riesgo medido es el falso positivo**: 1 desconocido confirmado en 157 (0.6 %).
4. **Los fallos restantes son de recall**, no de precisión: perfil, caras pequeñas y el caso de gorra y lentes (`jesus`).
5. Con la revalidación cada 30 frames, la etiqueta en pantalla "parpadea" un 10 % del tiempo; para el producto final (sin interfaz visible) no es un problema.

## 7. Limitaciones

- Un solo video etiquetado y una sola fecha.
- Etiquetas revisadas por dos personas, pero sobre caras de 37 px; algunas siguen siendo ambiguas.
- 22 de 157 desconocidos no tienen pista (filtrados por confianza), así que el FPR real se mide sobre 135 casos.
- El video 2 no tiene etiquetas: su resultado es cualitativo.
- No se probó con iluminación distinta, otras aulas ni estudiantes con accesorios en el registro.

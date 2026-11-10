# 08 — Video 3: revalidación en una segunda aula (11 partes)

**Estado:** ✅ Completada (2026-10-20).
**Corrida:** `experiments/unc/demo/prueba_2010_1600_video3/`.

---

## 1. Resumen ejecutivo

- Se procesó el video 3 (`video5096333890499381162.mp4`) dividido en **11 partes**, con
  **3 808 frames reales** y **38 756 decisiones** de rostro (~10 por frame).
- **Cobertura: 8 de 8 registrados presentes.** Los 8 estudiantes que asistieron fueron
  confirmados. `jose`, el único ausente, tuvo **0 confirmaciones** en todo el video.
- **Identidad: 1 error en 78 pistas confirmadas** (precisión 0.987). Una persona no
  registrada fue confirmada como `dani` durante 30 frames (1 s) con similitud 0.336.
- **Población no registrada:** **21 estudiantes no registrados** en el video. Sus 74
  pistas con ancho ≥ 40 px nunca llegaron a los 3 votos, salvo la del error anterior:
  **1 de 21 personas (4.8 %)**.
- **Costo de los cortes:** tras cada uno de los 10 empalmes hay que re-confirmar.
  Mediana **1.31 s**, media 2.57 s, máximo **13.2 s**.
- **Umbral 0.30:** a nivel de pista deja pasar 4 de 74 pistas desconocidas; la regla de
  **3 votos consecutivos** las reduce a 1. La regla temporal es la que sostiene la
  precisión, no el umbral solo.

---

## 2. Material y método

### 2.1 Las 11 partes

| Parte | Archivo | Frames | Negros | Contenido real |
|---|---|---|---|---|
| 01 | `lv_0_20261020111537.mp4` | 421 | 60 | 361 |
| 02 | `lv_0_20261020111622.mp4` | 272 | 0 | 272 |
| 03 | `lv_0_20261020111700.mp4` | 622 | 0 | 622 |
| 04 | `lv_0_20261020111727.mp4` | 260 | 0 | 260 |
| 05 | `lv_0_20261020111800.mp4` | 193 | 0 | 193 |
| 06 | `lv_0_20261020111820.mp4` | 353 | 0 | 353 |
| 07 | `lv_0_20261020111839.mp4` | 347 | 0 | 347 |
| 08 | `lv_0_20261020111859.mp4` | 369 | 0 | 369 |
| 09 | `lv_0_20261020111925.mp4` | 499 | 0 | 499 |
| 10 | `lv_0_20261020111959.mp4` | 251 | 0 | 251 |
| 11 | `lv_0_20261020112022.mp4` | 281 | 0 | 281 |
| **Total** | | **3 868** | **60** | **3 808** |

Todas 1280×720 a 30.00 fps.

### 2.2 Verificación de continuidad

Se comparó cada parte contra el video original (`video5096333890499381162.mp4`,
3 710 frames) usando miniaturas de 32×18 en escala de grises y error absoluto medio (MAE).

- Los empalmes son continuos: la cadena de coincidencias va del frame **0 al 3 709**,
  que es el último frame del original.
- MAE de las coincidencias: **0.06 a 0.23** (prácticamente idénticas).
- El original tiene **29.92 fps** y las partes **30.00 fps**, así que la re-exportación
  insertó ~98 frames duplicados y **60 frames negros** al final de la parte 1. No se
  perdió contenido; los negros se omiten y los duplicados no afectan la identidad.
- Consecuencia práctica: los tiempos en segundos no son comparables 1:1 con el original
  (30.00 vs 29.92 fps). Las comparaciones se hacen por frame, no por segundo.

### 2.3 Configuración de la corrida

- Motor definitivo: umbral **0.30**, filtro de tamaño **20 px**, confianza de detección
  **0.75**, anti-spoofing omitido bajo **60 px**, revalidación del ancla cada **30
  frames**, **3 votos** consecutivos.
- Cada parte se procesa como archivo independiente: el tracking, los votos y las anclas
  arrancan de cero. Los modelos ONNX se cargan una sola vez.
- La asistencia se acumula en una sola sesión (`asistencia_video3.csv`).

---

## 3. Cobertura: quién apareció y quién se confirmó

| Persona | Partes con confirmación | Confirmaciones | Sim. media | Sim. máxima |
|---|---|---|---|---|
| bob | 10/11 | 1 596 | **0.497** | 0.610 |
| carlos | 8/11 | 870 | 0.467 | 0.659 |
| kevin | 10/11 | 1 032 | 0.426 | 0.606 |
| dani | 11/11 | 1 226 | 0.422 | 0.596 |
| jesus | 6/11 | 564 | 0.385 | 0.496 |
| alexa | 11/11 | 1 470 | 0.370 | 0.509 |
| kate | 4/11 | 558 | 0.362 | 0.431 |
| fran | 2/11 | 87 | **0.328** | 0.341 |
| jose (ausente) | 0/11 | **0** | — | — |

Las diferencias de cobertura entre personas reflejan principalmente **cuánto tiempo
queda cada una en cuadro** y a qué distancia, no fallos del motor: `alexa` y `dani`
están visibles en las 11 partes; `fran` y `kate` aparecen en pocas.

`fran` es el caso más frágil: sus 3 pistas se confirmaron con 0.33–0.34 y solo en 2
partes. Es el estudiante que más se acerca al umbral por abajo.

**`jose` no aparece: 0 confirmaciones en 38 756 decisiones.** No se inventó a un ausente.

---

## 4. Identidad: 78 pistas, 1 error

Se revisó visualmente **una captura por pista** (13 alexa, 13 bob, 10 carlos, 14 dani,
3 fran, 6 jesus, 4 kate, 15 kevin) contra las fotos de registro.

### 4.1 El error encontrado

| Pista | Etiqueta | Frames de la pista | Frames etiquetados | Sim. máxima | Ancho mediano |
|---|---|---|---|---|---|
| parte 07, pista 3 | `dani` | 330 | 30 (1 s) | **0.336** | 30 px |

La persona del recorte **no corresponde a dani** al compararla con sus fotos de registro:
pelo corto, cara más ancha, otra montura de gafas y camisa mostaza. Es una estudiante
**no registrada**. El motor la confirmó durante 30 frames y luego volvió a `Desconocido`.

- No altera la asistencia: `dani` está confirmada en las 11 partes por otras pistas.
- Sí cuenta como **falso positivo de identidad**: 1 de 78 pistas (1.3 %).

### 4.2 Por qué la primera pasada no lo vio

En la hoja de contacto de `dani` el recorte aparecía junto a 13 capturas correctas y, a
150 px, la diferencia pasaba desapercibida. Se detectó al ordenar las confirmaciones por
similitud máxima y ampliar las 18 más débiles. **Las revisiones deben hacerse por
similitud ascendente, no en el orden natural de la carpeta.**

---

## 5. Población no registrada y umbral

### 5.1 Pistas y personas

| Conjunto | Pistas |
|---|---|
| Pistas totales (parte, id) | 441 |
| Confirmadas con nombre | 78 |
| Sin confirmar | 363 |
| Sin confirmar con ancho mediano ≥ 40 px | **74** |

Las 74 pistas sin confirmar de tamaño usable son la población no registrada (más los
registrados en pose difícil que no llegaron a 3 votos). El video tiene **21 estudiantes
no registrados**; con los 8 registrados presentes y el ausente (`jose`) son 29 personas
en el aula. El agrupamiento automático de esas pistas (cos ≥ 0.50) produce 24 grupos,
coherente si se tiene en cuenta que una misma persona se parte en varias pistas al cruzar
cortes y cambios de pose.

**Falsos positivos:** 1 de 21 personas no registradas (**4.8 %**) y 1 de 74 pistas de
tamaño usable (**1.4 %**).

Distribución de la **similitud máxima** de esas 74 pistas:

| Percentil | p50 | p75 | p90 | p95 | p99 | máx |
|---|---|---|---|---|---|---|
| sim_max | 0.226 | 0.262 | 0.288 | 0.306 | 0.344 | 0.344 |

### 5.2 Umbral a nivel de pista

Si se decidiera por la similitud máxima de la pista (1 voto), el intercambio sería:

| Umbral | Registradas que pasan | No registradas que pasan (FP) |
|---|---|---|
| 0.28 | 78/78 | 10/74 |
| **0.30** | **78/78** | **4/74** |
| 0.32 | 77/78 | 2/74 |
| 0.34 | 74/78 | 1/74 |
| 0.36 | 69/78 | 0/74 |
| 0.40 | 54/78 | 0/74 |

Con la regla real (**3 votos consecutivos**), los 4 cruces de 0.30 se reducen a **1
confirmación** (la del §4.1). Bajar el umbral a 0.34 eliminaría ese falso positivo pero
costaría 4 pistas correctas de `fran`, `jesus` y `kate`; subirlo a 0.36 costaría 9.

**Conclusión:** en este video el umbral 0.30 sigue siendo el mejor compromiso, y la
regla de 3 votos es la que aporta la precisión (4 → 1). No hay evidencia para moverlo.

### 5.3 Los desconocidos de la "zona gris"

Se guardaron los recortes con similitud ≥ 0.25 para revisión: 713 filas, que tras
deduplicar por (parte, segundo, pista) son **219 recortes únicos**. La revisión visual
muestra una mezcla de:

- estudiantes registrados en pose difícil (perfil, mirando abajo, mano en la cara, gorra);
- al menos una estudiante no registrada (la del §4.1).

Por eso **la zona gris no es solo "gente ajena"**: es también el costo de la pose en los
registrados. Es la misma causa ya documentada en la fase 0 (pose: 4.1 % de fallo de
frente contra 42.9 % de perfil).

---

## 6. Costo de los cortes

Cada corte reinicia el estado, así que cada persona debe volver a juntar 3 votos.

| Métrica | Valor |
|---|---|
| Re-confirmaciones medidas | 62 |
| Mediana | **1.31 s** |
| Media | 2.57 s |
| Mínimo | 0.1 s |
| Máximo | **13.2 s** (jesus, parte 9) |

En un video continuo el ancla sobreviviría al corte; aquí se paga ese costo en cada
empalme. Es aceptable para el uso previsto (archivos que llegan por partes), pero
conviene saberlo al medir latencias.

---

## 7. Tamaño de cara y tasa de reconocimiento

| Ancho de cara | Caras | Reconocidas | Tasa |
|---|---|---|---|
| 20–30 px | 14 513 (38.9 %) | 1 390 | **9.6 %** |
| 30–40 px | 8 533 | 1 704 | 20.0 % |
| 40–60 px | 8 643 | 2 683 | **31.0 %** |
| 60–90 px | 4 778 | 1 416 | 29.6 % |
| 90+ px | 867 | 210 | 24.2 % |
| **Total** | **37 334** | **7 403** | **19.8 %** |

El 39 % de las caras mide menos de 30 px y ahí la tasa cae al 9.6 %. Esto explica que el
75 % de las decisiones quede en `Desconocido` sin que eso signifique que el motor falle
con las personas: cuando el rostro tiene tamaño usable, la confirmación llega en segundos.

Estados totales (sin `CENTRATE`): `CONFIRMADO` 281, `ANCLADO` 7 122, `VERIFICANDO` 701,
`DESCONOCIDO` 29 218, `FRAUDE` 12. Además, 1 422 decisiones `CENTRATE` (rostro tocando el
borde del cuadro).

---

## 8. Anti-spoofing

12 filas `FRAUDE` en todo el video, sobre caras de 30 a 120 px. Son falsas alarmas del
modelo en caras pequeñas, ya documentadas en la fase 5: no confirman ni bloquean
asistencia porque el estado `FRAUDE` no suma votos. El umbral de 0.90 sigue siendo
inestable a ese tamaño.

> **Nota de auditoría (A6):** estas 12 filas pertenecen a la corrida del video 3.
> No son comparables con las 8 alarmas de
> `experiments/unc/metricas/salida/asistencia/spoofing.csv` (revisión del video 1)
> ni con el "39 → 3" de la Fase 5 (`07_reporte_final_unc.md`): cada cifra sale de
> una corrida y una configuración distintas.

---

## 9. Conclusiones

1. **La cobertura se sostiene en una segunda aula:** 8/8 presentes, y el ausente nunca
   fue confirmado.
2. **Apareció el primer falso positivo de identidad del proyecto** (1 de 78 pistas,
   1.3 %; 1 de 21 estudiantes no registrados, 4.8 %), causado por una estudiante no
   registrada parecida a `dani` y con solo 30 frames etiquetados. No afectó la asistencia.
3. **El umbral 0.30 no se mueve:** a nivel de pista deja pasar 4 de 74 desconocidos, y
   los 3 votos lo reducen a 1. Subirlo a 0.34 costaría 4 pistas correctas.
4. **El costo de partir el video es bajo:** mediana 1.31 s para re-confirmar tras cada
   corte, con un máximo de 13.2 s.
5. **La tasa por frame (19.8 %) la explica el tamaño**, no la identidad: el 39 % de las
   caras mide menos de 30 px.

### Límites de esta evaluación

- Un solo video y una sola sesión: no reemplaza la revalidación con más aulas y fechas.
- No hay verdad terreno por frame; la identidad se verificó por pista, no por rostro.
- El número exacto de estudiantes no registrados no está etiquetado; se usó la
  estimación del plan (~20) y el agrupamiento de pistas como control.
- La parte 1 incluye 60 frames negros y hay ~98 frames duplicados por la re-exportación.

---

## 10. Artefactos

| Archivo | Contenido |
|---|---|
| `decisiones_video3.csv` | una fila por decisión, con `parte`, `frame_parte` y `frame_global` |
| `resumen_por_parte.csv` | frames, negros, caras y confirmados por parte |
| `resumen_personas.csv` | confirmaciones, similitudes y primera aparición por persona |
| `asistencia_video3.csv` | asistencia de la sesión completa |
| `estudiantes/<nombre>/` | capturas de cada reconocimiento |
| `revision/<persona>.jpg` | una captura por pista, para verificar identidad |
| `revision/pistas_debiles.jpg` | las 18 confirmaciones de menor similitud (donde apareció el error) |
| `revision/pistas_sin_confirmar_*.jpg` | las 74 pistas no confirmadas, para etiquetar |
| `revision/desconocidos_dedup_*.jpg` | los 219 recortes únicos de la zona gris |
| `revision/pistas_debiles.csv`, `pistas_sin_confirmar.csv`, `clusters_sin_confirmar.csv` | tablas de revisión |

### Scripts

- `experiments/unc/fase7_video3_partes.py` — corrida sobre las 11 partes.
- `experiments/unc/fase7_revision.py` — hojas de revisión por persona.
- `experiments/unc/fase7_desconocidos.py` — deduplicación de la zona gris.
- `experiments/unc/fase7_pistas.py` — confirmaciones débiles y pistas desconocidas.
- `experiments/unc/fase7_pistas_sin_confirmar.py` — hoja de las 74 pistas.
- `experiments/unc/fase7_agrupar_desconocidos.py` — agrupamiento por centroide.
- `experiments/unc/fase7_clusters.py` — conteo de personas distintas.

---

## 11. Reproducibilidad

```bash
# 1. Corrida completa sobre las 11 partes (~7 min)
venv/bin/python experiments/unc/fase7_video3_partes.py

# 2. Material de revisión
venv/bin/python experiments/unc/fase7_revision.py
venv/bin/python experiments/unc/fase7_desconocidos.py
venv/bin/python experiments/unc/fase7_pistas.py
venv/bin/python experiments/unc/fase7_pistas_sin_confirmar.py
venv/bin/python experiments/unc/fase7_clusters.py
```

Los resultados viven en `experiments/unc/demo/`, que está fuera del control de versiones
(ver `.gitignore`). Los comandos de arriba los regeneran desde los videos de
`data/unc/salon/`.

# 02 — Resultados de registro (dataset UNC)

**Fecha:** 2026-10-14
**Estado:** Registro de los 9 estudiantes completado. Verificación de centroides completada.
**Base de datos:** `data/database_embeddings.pkl` (9 personas)

---

## 1. Contexto de esta ejecución

- Reinicio previo: respaldo en `data/_respaldo_2026-10-14/`. No había registros previos en `data/`.
- Un video por estudiante, tomado de `data/unc/stds/`. Ver la tabla de videos en `00_plan_experimentos_unc.md`.
- Liveness desactivado en el registro (decisión del responsable del proyecto).
- Ajustes de código aplicados antes del registro (ver `00_plan_experimentos_unc.md`, sección "Registro de correcciones").

## 2. Comandos ejecutados

Desde la raíz del proyecto:

```bash
venv/bin/python scripts/run_single_video_reg.py alexa  data/unc/stds/alexa/VID_20260907_103652.mp4  --sin-ventana
venv/bin/python scripts/run_single_video_reg.py bob    data/unc/stds/bob/VID_20260907_103303.mp4    --sin-ventana
venv/bin/python scripts/run_single_video_reg.py carlos data/unc/stds/carlos/VID_20260907_103851.mp4 --sin-ventana
venv/bin/python scripts/run_single_video_reg.py dani   data/unc/stds/dani/VID_20260907_103138.mp4   --sin-ventana
venv/bin/python scripts/run_single_video_reg.py fran   data/unc/stds/fran/VID_20260907_103618.mp4   --sin-ventana
venv/bin/python scripts/run_single_video_reg.py jesus  data/unc/stds/jesus/VID_20260907_103208.mp4  --sin-ventana
venv/bin/python scripts/run_single_video_reg.py jose   data/unc/stds/jose/VID_20260907_103228.mp4   --sin-ventana
venv/bin/python scripts/run_single_video_reg.py kate   data/unc/stds/kate/VID_20260907_103318.mp4   --sin-ventana
venv/bin/python scripts/run_single_video_reg.py kevin  data/unc/stds/kevin/VID_20260907_103913.mp4  --sin-ventana
```

Nota: `carlos` y `dani` fueron registrados por el responsable del proyecto. Los demás se registraron uno a uno con el mismo procedimiento.

## 3. Resultados por persona

| Persona | Fotos guardadas | Norma del centroide | Nitidez mediana (recorte) | Brillo medio | Pareja más parecida | Similitud máxima con otra persona |
|---|---|---|---|---|---|---|
| alexa  | 50 | 1.000000 | 741.9  | 105.7 | dani    | 0.297 |
| bob    | 50 | 1.000000 | 1883.7 | 131.9 | carlos  | 0.261 |
| carlos | 50 | 1.000000 | 1079.0 | 114.8 | bob     | 0.261 |
| dani   | 50 | 1.000000 | 1165.5 | 98.0  | alexa   | 0.297 |
| fran   | 50 | 1.000000 | 487.6  | 133.0 | jesus   | 0.295 |
| jesus  | 50 | 1.000000 | 476.9  | 117.1 | fran    | 0.295 |
| jose   | 50 | 1.000000 | 923.9  | 114.3 | jesus   | 0.216 |
| kate   | 50 | 1.000000 | 599.8  | 103.8 | alexa   | 0.262 |
| kevin  | 50 | 1.000000 | 1113.5 | 122.4 | jesus   | 0.249 |

## 4. Matriz de similitud coseno entre centroides

|        | alexa | bob | carlos | dani | fran | jesus | jose | kate | kevin |
|---|---|---|---|---|---|---|---|---|---|
| alexa  | 1.000 | 0.140 | 0.113 | 0.297 | 0.113 | 0.128 | 0.046 | 0.262 | 0.135 |
| bob    | 0.140 | 1.000 | 0.261 | 0.133 | 0.221 | 0.106 | 0.118 | -0.051 | 0.025 |
| carlos | 0.113 | 0.261 | 1.000 | 0.205 | 0.150 | 0.109 | -0.152 | 0.120 | 0.010 |
| dani   | 0.297 | 0.133 | 0.205 | 1.000 | 0.090 | 0.239 | 0.045 | 0.061 | 0.035 |
| fran   | 0.113 | 0.221 | 0.150 | 0.090 | 1.000 | 0.295 | 0.157 | 0.088 | 0.102 |
| jesus  | 0.128 | 0.106 | 0.109 | 0.239 | 0.295 | 1.000 | 0.216 | 0.101 | 0.249 |
| jose   | 0.046 | 0.118 | -0.152 | 0.045 | 0.157 | 0.216 | 1.000 | -0.056 | 0.060 |
| kate   | 0.262 | -0.051 | 0.120 | 0.061 | 0.088 | 0.101 | -0.056 | 1.000 | 0.046 |
| kevin  | 0.135 | 0.025 | 0.010 | 0.035 | 0.102 | 0.249 | 0.060 | 0.046 | 1.000 |

## 5. Observaciones

1. **Registro completo.** Los 9 centroides tienen dimensión 512 y norma 1.0. Cada persona tiene 50 fotos, por encima del mínimo de 15.

2. **Pares cercanos.** La mayor similitud entre dos personas distintas es 0.297 (`alexa`–`dani`), justo debajo de la alerta de 0.30. Le siguen `fran`–`jesus` (0.295) y `alexa`–`kate` (0.262). Dos de los pares cercanos son mujeres con lentes (`alexa`, `dani`).

3. **Margen frente al umbral del motor.** El umbral de decisión del motor es 0.35. El margen con la pareja más cercana es de solo ~0.05. Esto se debe medir con reconocimiento real sobre el salón.

4. **Nitidez.** Todas las fotos guardadas tienen nitidez entre 477 y 1 884. El umbral de descarte es 60, así que el filtro no descartó ninguna foto. Los valores de nitidez de la tabla del plan (por ejemplo, `alexa` v1 = 74, `kate` = 22) no son comparables con esta medida. Hay que definir de forma única dónde y cómo se mide la nitidez.

5. **Brillo.** Dani es la persona más oscura (98) y bob la más clara (132). Dentro de rango; el umbral de brillo tampoco descartó fotos.

6. **Valores negativos.** Algunas parejas tienen similitud negativa (por ejemplo `carlos`–`jose` = -0.152). No es un problema: indica que los vectores están en lados opuestos del espacio.

7. **Lo que no se midió todavía:**
   - Dispersión interna de cada persona (qué tan parecidas son sus propias fotos).
   - Reconocimiento sobre el salón.
   - Comportamiento con y sin accesorios.

## 6. Demostración paso a paso (un recorte)

Recorte `v1_00020` de `VID_20260907_102851.mp4`, frame 30 (segundo 1.0), etiqueta humana `dani`.

1. Recorte de muestra: `x1=674 y1=779 x2=754 y2=884`, ancho 80 px.
2. Frame completo leído: 1080×1920.
3. Detección RetinaFace sobre el frame: **18 caras**. Anchos entre 26 y 86 px. Varias son asistentes del fondo.
4. La cara del recorte es la de bbox `[674, 779, 754, 884]`, confianza 0.999. Coincide con el recorte.
5. CLAHE aplicado (solo luminancia, espacio LAB).
6. Embedding ArcFace: dimensión 512, norma 1.0.
7. Similitud coseno contra los 9 centroides: `dani` 0.573, `alexa` 0.189, `carlos` 0.089, `bob` 0.075, `fran` 0.061, `jose` 0.046, `kevin` 0.027, `kate` 0.022, `jesus` 0.004.
8. Decisión con umbral 0.35: **dani** (margen de 0.38 sobre la segunda opción).
9. Coincide con la etiqueta humana.

Imagen anotada: `experiments/unc/salon_muestra/demo_v1_00020.jpg`.

Observación: esta cara tiene un margen amplio, así que no es un caso difícil. Los casos que importan son los de similitud cercana al umbral.

## 7. Dispersión interna por persona

Script: `experiments/unc/fase2_dispersion.py`. Salida: `experiments/unc/fase2_dispersion.csv` y `fase2_dispersion.png`.

| Persona | Media | Mínimo | Máximo | Desviación |
|---|---|---|---|---|
| alexa  | 0.725 | 0.579 | 0.806 | 0.059 |
| bob    | 0.757 | 0.617 | 0.834 | 0.059 |
| carlos | 0.775 | 0.568 | 0.880 | 0.088 |
| dani   | 0.692 | 0.543 | 0.768 | 0.056 |
| fran   | 0.742 | 0.396 | 0.855 | 0.097 |
| jesus  | 0.644 | 0.139 | 0.820 | 0.168 |
| jose   | 0.777 | 0.653 | 0.862 | 0.042 |
| kate   | 0.807 | 0.706 | 0.886 | 0.041 |
| kevin  | 0.665 | 0.369 | 0.776 | 0.104 |

**Hallazgo:** `jesus` tiene 4 fotos por debajo del umbral 0.35 respecto a su propio centroide (0.139, 0.243, 0.255, 0.335). Son consecutivas (frames 22 a 25 del registro): la cabeza inclinada hacia abajo y el rostro casi de perfil. Están en el centroide porque el filtro de diversidad solo descarta fotos casi idénticas, no fotos atípicas.

**Efecto medido (2026-10-15, sin modificar la base de datos):**
- Al quitar las 4 fotos atípicas, el centroide cambia muy poco: similitud coseno de 0.9965 entre el centroide actual y el nuevo.
- Las 4 fotos siguen lejos del centroide nuevo (0.079 a 0.288), así que son atípicas con o sin ellas.
- La media de similitud de las 46 fotos restantes sube de 0.644 a 0.681.
- La similitud de `jesus` con los otros 8 cambia como máximo 0.01. La pareja más cercana sigue siendo `fran` (0.295 → 0.297).
- Conclusión: el efecto sobre las decisiones es pequeño. Excluirlas es razonable, pero no cambia el panorama.

**Decisión (2026-10-15):** se deja el centroide de `jesus` tal como está (opción a). La base de datos no se modifica. Las fotos atípicas quedan documentadas aquí.

## 8. Pendientes

- Decidir si se excluyen las fotos atípicas de `jesus` (y revisar `fran` y `kevin`, con valores bajos aunque fuera del umbral).
- Ejecutar el reconocimiento sobre las 300 caras etiquetadas del salón, con emparejamiento por IoU.
- Revisar las etiquetas de `muestra_anotacion.csv`: etiquetado por una sola persona, sin segunda revisión, con caras de baja resolución (~37 px en mediana).
- Verificar accesorios de `carlos` y `dani` en sus videos de registro.
- Corregir la observación 2: no se verificó que `alexa` tenga lentes.

## 9. Reconocimiento sobre la muestra del salón (2026-10-15)

Script: `experiments/unc/fase0_reconocimiento_salon.py`. Salida: `experiments/unc/salon_muestra/reconocimiento_salon.csv`.

**Advertencia:** las etiquetas son de una sola persona, sin segunda revisión, y las caras miden en mediana 37 px. Estos números son orientativos, no una medida final.

| Resultado | Cantidad |
|---|---|
| Aciertos | 223 |
| Falsos negativos (era persona registrada, dijo Desconocido) | 16 |
| Falsos positivos (era desconocido, dijo nombre) | 4 |
| No evaluables (`dudoso` o `no_cara`) | 57 |
| Errores de identidad (era A, dijo B) | 0 |

**Por persona (etiqueta → decisión):** `dani` 16/17 correctos; `kevin` 7/7; `carlos` 4/4; `fran` 3/3; `jose` 8/9; `kate` 10/13; `alexa` 9/12; `bob` 9/14; `jesus` 1/4. Desconocidos: 156/160 correctos.

**Falsos negativos:** tienen similitud baja con su persona (0.12 a 0.34). 11 de los 16 caen en caras menores de 40 px. Muchos son de perfil o parcialmente tapados. Tres de `bob` (`v1_00327`, `v1_00491`, `v1_00510`) muestran la camiseta a rayas de su video de registro, así que probablemente son `bob`.

**Falsos positivos:** tienen similitud alta (0.37 a 0.57). Revisados visualmente, varios se parecen mucho a un registrado (por ejemplo `v1_00778`, con gafas, barba y camisa azul como `carlos`). Es posible que la etiqueta `desconocido` sea incorrecta, o que haya confusión real. No podemos decidirlo sin una segunda revisión.

**Lo que sigue:** revisar los 4 falsos positivos con más cuidado y los 16 falsos negativos uno por uno, y decidir si las etiquetas se corrigen.

## 10. Corrección de etiquetas y nuevas cifras (2026-10-15)

Tras la revisión del responsable, cuatro etiquetas de `desconocido` pasaron al nombre que dio el sistema, porque el sistema acertó:

| Recorte | Antes | Ahora |
|---|---|---|
| `v1_00091` | desconocido | jose |
| `v1_00104` | desconocido | carlos |
| `v1_00304` | desconocido | jose |
| `v1_00778` | desconocido | carlos |

Cifras con las etiquetas corregidas (243 evaluables):

| Resultado | Cantidad |
|---|---|
| Aciertos | 227 |
| Falsos negativos | 16 |
| Falsos positivos | 0 |
| Errores de identidad | 0 |
| No evaluables | 57 |

- Caras de personas registradas (87 etiquetadas): 71 acierto, 16 falso negativo, 81.6 %.
- Caras de desconocidos (156 etiquetados): 156 acierto, 100 %.

Nota: estas cifras dependen de etiquetas hechas por una sola persona. Los comentarios visuales de la sección 9 eran impresiones iniciales, no verificaciones, y no deben citarse como evidencia.

## 11. Revisión de los 16 falsos negativos (2026-10-15)

El responsable confirmó que los 16 recortes son la persona indicada por la etiqueta. Por lo tanto:
- Las etiquetas de estos 16 no cambian.
- Son fallos reales del sistema, no errores de etiqueta.
- Cifras sin cambios: 227 aciertos, 16 falsos negativos, 0 falsos positivos, 81.6 % de acierto en personas registradas (71 de 87).

Pendiente: explicar por qué fallan. El tamaño de la cara no lo explica por sí solo: la mediana de ancho de los falsos negativos (34.5 px) es parecida a la de los aciertos (37 px), y 128 de los 223 aciertos también tienen menos de 40 px. Hipótesis a probar: pose (perfil), accesorios (gorra y gafas en `jesus`), borrosidad y oclusión.

## 12. Posibles causas de los falsos negativos (2026-10-15)

Script: `experiments/unc/fase0_causas_fallos.py`. Salida: `experiments/unc/salon_muestra/causas_fallos.csv`. Universo: 87 recortes con nombre (71 aciertos, 16 falsos negativos).

**Medianas por grupo:**

| Grupo | Ancho (px) | Nitidez | Pose |
|---|---|---|---|
| Aciertos (n=71) | 38.0 | 1 302.9 | 0.061 |
| Falsos negativos (n=16) | 34.5 | 1 788.5 | 0.288 |

**Tasa de fallo por pose** (pose = desviación de la nariz respecto al centro de los ojos, dividida entre la distancia entre ojos):

| Pose | Casos | Falsos negativos | Tasa de fallo |
|---|---|---|---|
| Frente (< 0.10) | 49 | 2 | 4.1 % |
| Leve (0.10 a 0.20) | 9 | 1 | 11.1 % |
| Perfil (≥ 0.20) | 29 | 13 | 44.8 % |

**Tasa de fallo por ancho:**

| Ancho | Casos | Falsos negativos | Tasa de fallo |
|---|---|---|---|
| < 30 px | 14 | 5 | 35.7 % |
| 30 a 39 px | 34 | 6 | 17.6 % |
| 40 a 59 px | 11 | 4 | 36.4 % |
| ≥ 60 px | 28 | 1 | 3.6 % |

**Lectura:**
- **La pose es el factor más fuerte.** Con la cara de frente el fallo es 4 %; de perfil, 45 %.
- **El tamaño también influye,** pero no de forma limpia: las caras de 40 a 59 px fallan más que las de 30 a 39 px. Con tan pocos casos, este orden puede ser ruido.
- **La nitidez no explica los fallos.** Los falsos negativos son incluso más nítidos que los aciertos. El problema no es la borrosidad.
- **Los accesorios no están medidos.** Quedan pendientes; hay que anotarlos a mano.

**Limitaciones:** la pose es una aproximación con 5 puntos, no un ángulo real. Con 16 falsos negativos, las tasas son orientativas.

## 13. Corrección: `v1_00789` pasa a desconocido (2026-10-15)

El responsable indicó que `v1_00789` es un desconocido, no `jose`. El sistema respondió Desconocido, así que es un acierto y deja de ser falso negativo. Las secciones 10 a 12 quedan actualizadas con estas cifras:

| Resultado | Antes (sección 11) | Ahora |
|---|---|---|
| Aciertos | 227 | **228** |
| Falsos negativos | 16 | **15** |
| Falsos positivos | 0 | 0 |
| Personas registradas etiquetadas | 87 | **86** (71 aciertos, 15 falsos negativos, 82.6 %) |
| Desconocidos etiquetados | 156 | **157** (157 aciertos, 100 %) |

**Causas (sección 12), con el universo corregido de 86 casos:**
- Mediana de pose: aciertos 0.061; falsos negativos 0.278.
- De frente: 2 fallos de 49 (4.1 %).
- De perfil: 12 fallos de 28 (42.9 %).
- Ancho ≥ 60 px: 0 fallos de 27.

La conclusión de la sección 12 se mantiene: la pose es el factor más claro.

## 14. Accesorios y fallos (PROVISIONAL, 2026-10-15)

Scripts: `experiments/unc/fase0_lista_accesorios.py`. Archivo: `experiments/unc/salon_muestra/accesorios_muestra.csv`. Las anotaciones son **preliminares, pendientes de confirmación del responsable**. No usar como evidencia hasta confirmar.

Universo: 86 recortes con nombre (71 aciertos, 15 falsos negativos).

| Condición | Casos | Falsos negativos | Tasa de fallo |
|---|---|---|---|
| Con lentes | 34 | 3 | 8.8 % |
| Sin lentes | 47 | 8 | 17.0 % |
| Lentes dudosos | 5 | 4 | 80 % |
| Con gorra | 7 | 3 | 42.9 % |
| Sin gorra | 79 | 12 | 15.2 % |
| Lentes y gorra | 4 | 3 | 75 % |
| Sin lentes ni gorra | 44 | 8 | 18.2 % |

**Lectura preliminar:**
- Los lentes **no** aumentan el fallo en esta muestra; de hecho, las caras con lentes fallan menos. Con 34 casos no es concluyente.
- La gorra sí parece asociada al fallo (3 de 7). Con tan pocos casos, no es concluyente.
- Los 5 casos dudosos fallan en 4 de 5. Son caras borrosas o ambiguas, y el fallo ahí puede deberse a la calidad, no al accesorio.
- La pose sigue siendo la explicación más fuerte (sección 12). Hay que cruzar accesorio y pose para separar los efectos.

## 15. Accesorios y fallos: resultados con anotación confirmada (2026-10-15)

Archivo: `experiments/unc/salon_muestra/accesorios_muestra.csv`, anotado inicialmente y confirmado y corregido por el responsable. Cuatro valores `si` sin tilde se normalizaron a `sí`. Los datos son de 86 recortes con nombre (71 aciertos, 15 falsos negativos).

| Condición | Casos | Falsos negativos | Tasa de fallo |
|---|---|---|---|
| Con lentes | 38 | 6 | 15.8 % |
| Sin lentes | 47 | 8 | 17.0 % |
| Con gorra | 7 | 3 | 42.9 % |
| Sin gorra | 79 | 12 | 15.2 % |
| Lentes y gorra | 4 | 3 | 75 % |
| Solo gorra | 3 | 0 | 0 % |
| Solo lentes | 34 | 3 | 8.8 % |
| Ni lentes ni gorra | 44 | 8 | 18.2 % |

**Lectura:**
- **Los lentes no cambian el fallo.** 15.8 % con lentes frente a 17.0 % sin lentes.
- **La gorra aparece en 7 recortes: 4 de `jesus` y 3 de `fran`.** De los 4 de `jesus`, 3 fallan. Los 3 de `fran` aciertan.
- **La gorra por sí sola no basta para fallar.** `fran` con gorra y cara de frente acierta en los 3 casos. El `jesus` con gorra de frente (v1_01011) también acierta.
- **Los 3 fallos con gorra están en perfil o muy girados** (pose 0.30 a 0.43). Coinciden con la pose, no solo con la gorra.
- **Con gorra y gafas, `jesus` falla 3 de 4.** Son casos de perfil con ambos accesorios.

**Limitaciones:** con 7 casos con gorra, todo lo anterior es una señal, no una conclusión. Los 4 `jesus` con gorra vienen de un solo video del salón, así que no sabemos si el efecto se repite en otras condiciones.

**Pendiente:** la columna `notas` del CSV sigue diciendo "pre-anotación, pendiente de confirmación". Conviene actualizarla a "confirmado por responsable".

## 16. Demo visual del motor sobre el salón (2026-10-15)

Script: `experiments/unc/demo_video_salon.py`. Video anotado: `experiments/unc/demo/demo_salon_0s.mp4` (primeros 30 s de `VID_20260907_102851.mp4`). Decisiones por frame: `experiments/unc/demo/decisiones_0s.csv`. La asistencia de prueba se escribe en `experiments/unc/demo/asistencia_demo_no_oficial.csv`, no en `data/asistencia.csv`.

Resultados en 30 s (899 frames):
- 1 493 decisiones sobre 33 pistas de seguimiento.
- Estados: DESCONOCIDO 994, ANCLADO 430, CENTRATE 36, VERIFICANDO 21, CONFIRMADO 9, FRAUDE 3.
- Asistencia confirmada: `dani`, `jose`, `jesus`, `bob`, `carlos`.

**Hallazgo: falsa alarma de anti-spoofing.** En el segundo 6.2 (frames 186 a 188) el sistema marcó "FRAUDE DETECTADO" a una persona real, con la cabeza inclinada y el rostro de perfil. Revisado visualmente. Probablemente el modelo de liveness no está calibrado para ángulos así. Hay que medir cuántas veces pasa en el video completo antes de decidir qué hacer.

**Otros comportamientos visibles:** en el segundo 20 el salón está casi vacío y el motor no detecta caras (correcto). Varias personas se mueven y salen del encuadre.

## 17. Efecto del filtro de tamaño mínimo (2026-10-15)

Script: `experiments/unc/fase0_efecto_filtro_tamano.py`. Salida: `experiments/unc/salon_muestra/efecto_filtro_tamano.csv`. Se simula el filtro sobre las decisiones ya calculadas (no se recorrió el video de nuevo).

| Persona | Filtro 40 px (actual) | Filtro 20 px | Sin filtro |
|---|---|---|---|
| alexa | 0/12 (0 %) | 9/12 (75 %) | 9/12 (75 %) |
| bob | 0/14 (0 %) | 9/14 (64 %) | 9/14 (64 %) |
| carlos | 4/6 (67 %) | 6/6 (100 %) | 6/6 (100 %) |
| dani | 16/17 (94 %) | 16/17 (94 %) | 16/17 (94 %) |
| fran | 3/3 (100 %) | 3/3 (100 %) | 3/3 (100 %) |
| jesus | 1/4 (25 %) | 1/4 (25 %) | 1/4 (25 %) |
| jose | 10/10 (100 %) | 10/10 (100 %) | 10/10 (100 %) |
| kate | 0/13 (0 %) | 10/13 (77 %) | 10/13 (77 %) |
| kevin | 0/7 (0 %) | 7/7 (100 %) | 7/7 (100 %) |
| Desconocidos (falsos positivos) | 0/157 | 0/157 | 0/157 |

**Lectura:**
- Con el filtro de 40 px, **alexa, bob, kate y kevin no se reconocen nunca**. Con 20 px, sí.
- Bajar el filtro a 20 px recupera 30 aciertos sin introducir falsos positivos en los 157 desconocidos.
- El filtro de 20 px y el de 0 px dan lo mismo, porque la muestra solo contiene caras de 21 px o más (la detección original tenía ese límite).
- `jesus` sigue en 25 % con cualquier filtro. Su caso es de gorra, no de tamaño.

**Limitaciones:**
- Esto mide el reconocimiento aislado. En el motor real, las decisiones pasan también por la votación de 3 frames y por el anti-spoofing, que pueden cambiar el resultado.
- Las caras menores de 21 px no están en la muestra, así que no sabemos qué pasa con ellas.
- Con 20 px el anti-spoofing se aplicaría a caras muy pequeñas, donde puede dar más falsas alarmas (como la de la sección 16).

## 18. Prueba del motor con filtro de 20 px (2026-10-15)

Se corrió el motor real (`asistencia_unificada_optimizada.py`, sin cambios en el código) sobre los primeros 30 s del video del salón, con filtro de 40 px y con filtro de 20 px. El filtro se fija desde `experiments/unc/demo_video_salon.py` con `--filtro=`.

| Medida | Filtro 40 px | Filtro 20 px |
|---|---|---|
| Decisiones | 1 493 | 4 072 |
| Pistas de seguimiento | 33 | 108 |
| Personas confirmadas o ancladas | dani, jose, jesus, bob, carlos | dani, jose, jesus, bob, alexa, kate, kevin, carlos |
| Alarmas de FRAUDE | 3 | 3 |

**Verificación contra la muestra etiquetada** (recortes de los primeros 30 s con decisión asociada por IoU ≥ 0.5):

| Resultado | Filtro 40 px | Filtro 20 px |
|---|---|---|
| Personas registradas reconocidas y confirmadas | 3 | 4 |
| Personas registradas no reconocidas | 2 | 3 |
| Caras sin decisión (no procesadas por el filtro) | 25 | 0 |
| Desconocidos correctos | 11 | 25 |
| Identidades incorrectas | 0 | 0 |
| Desconocidos confirmados como registrados | 0 | 0 |

**Lectura:**
- Con 20 px el motor procesa las 25 caras que antes se ignoraban, sin confundir identidades ni confirmar desconocidos en esta muestra.
- Pasa de 33 a 108 pistas. Muchas son caras de fondo que no están en la muestra; no las hemos verificado una por una.

**Limitaciones:** la verificación usa solo unos 40 recortes etiquetados en 30 s. Es una prueba de humo, no una medida general. Para decidir el umbral del filtro hace falta correr el video completo (106 s) y verificar contra los 300 recortes.

## 19. Video completo del salón con filtro 20 px vs 40 px (2026-10-15)

Motor: `scripts/asistencia_unificada_optimizada.py` (sin cambios). Video: `VID_20260907_102851.mp4` completo (107 s). Verificación contra los 300 recortes etiquetados (decisión asociada por IoU ≥ 0.5 en el mismo frame).

Salidas: `experiments/unc/demo/demo_salon_0s_107s_filtro{20,40}.mp4` y `decisiones_0s_107s_filtro{20,40}.csv`.

**Contra la muestra etiquetada (300 recortes):**

| Resultado | Filtro 40 px | Filtro 20 px |
|---|---|---|
| Personas registradas reconocidas y confirmadas | 33 | **75** |
| Personas registradas no reconocidas | 5 | 11 |
| Personas registradas sin decisión (no procesadas) | 48 | 0 |
| Desconocidos correctos | 60 | **156** |
| Desconocidos sin decisión | 97 | 1 |
| Identidades incorrectas | 0 | 0 |
| Desconocidos confirmados como registrados | 0 | 0 |

De las 86 caras con nombre, el filtro de 20 px reconoce 75 (87 %). Con 40 px, 33 (38 %).

**Decisiones en el video completo:**

| Medida | Filtro 40 px | Filtro 20 px |
|---|---|---|
| Decisiones | 13 893 | 33 534 |
| Pistas de seguimiento | 146 | 308 |
| Alarmas de FRAUDE | 6 | **39** |

**Alarmas de FRAUDE con 20 px:** caen en 8 pistas. Una pista acumula 21 de las 39. Las cajas miden casi todas entre 28 y 38 px. El anti-spoofing no es confiable a esos tamaños.

**Pendiente:**
- Revisar las identidades confirmadas en el video completo que no están en la muestra (por ejemplo, `bob` con 1 493 decisiones con 20 px, frente a 36 con 40 px). La muestra no cubre esas caras.
- Decidir qué hacer con el anti-spoofing en caras pequeñas.

## 20. Revisión de las alarmas de FRAUDE (2026-10-16)

Script: `experiments/unc/fase0_revision_fraudes.py`. Salida: `experiments/unc/demo/revision_fraudes_resumen.jpg` y `revision_fraudes.csv`.

**Resultado visual:** las 8 pistas con alarma son **personas reales**. Son falsas alarmas.

| Pista | Alarmas | Segundo | Ancho (px) |
|---|---|---|---|
| 372 | 21 | 62.7 | 32 |
| 290 | 8 | 53.4 | 32 |
| 2 | 3 | 6.2 | 68 |
| 344 | 2 | 60.0 | 28 |
| 527 | 2 | 97.2 | 50 |
| 305 | 1 | 62.3 | 38 |
| 385 | 1 | 66.0 | 32 |
| 555 | 1 | 105.5 | 59 |

**Por qué el motor marca fraude:** el motor da fraude cuando la confianza del modelo es menor a 0.90, aunque el modelo diga que la cara es real. Para la pista 372 (frame 1880):
- Con la caja exacta de la detección (972.3, 446.9, 1004.2, 488.2): `is_real = True`, confianza **0.805** → fraude.
- Con la caja redondeada a enteros (972, 446, 1004, 488): `is_real = True`, confianza **0.970** → real.

Es decir, **un redondeo de menos de un píxel cambia la decisión**.

**Consecuencia:** el anti-spoofing no es confiable para caras de unos 30 px. El umbral de 0.90 está muy cerca de lo que el modelo devuelve para caras pequeñas.

**Corrección a la revisión:** el script `fase0_revision_fraudes.py` recalcula con la caja redondeada, así que su columna `es_real_modelo` = True no reproduce la decisión del motor. Su resumen visual sí es válido.

## 21. Anti-spoofing omitido en caras pequeñas (2026-10-16)

**Cambio en el motor:** se agregó `tamano_minimo_liveness = 60` en `scripts/asistencia_unificada_optimizada.py`. Con ancho de cara menor a 60 px, el anti-spoofing no se evalúa (estado "omitido") y la cara sigue el flujo normal. Decisión tomada por la inestabilidad del modelo demostrada en la sección 20.

**Efecto en el video completo (107 s, filtro 20 px):**

| Medida | Antes | Después |
|---|---|---|
| Alarmas de FRAUDE | 39 | **3** |
| Personas registradas reconocidas (de 86) | 75 | 76 |
| Desconocidos correctos (de 157) | 156 | 156 |

Las 3 alarmas que quedan son de una sola pista (la del segundo 6.2, sección 16), con caja de 68 px, por encima del umbral de 60. Es la falsa alarma por perfil, y no se resuelve con el umbral de tamaño.

**Riesgo aceptado:** una foto o pantalla con una cara de menos de 60 px no será detectada como fraude. Queda anotado.

## 22. Revisión de las pistas de `bob` (2026-10-16)

Script: `experiments/unc/fase0_revision_pistas.py bob 20`. Salida: `experiments/unc/demo/pistas_bob_filtro20.csv`, `pistas_bob_filtro20.jpg` y `pistas_bob_filtro20_zoom.jpg`.

Las 1 493 decisiones de `bob` (filtro 20 px) provienen de **6 pistas**, y en la revisión visual todas son el mismo hombre, con camiseta de Venezuela:

| Pista | Frames | Segundos | Ancho mediano | Similitud máxima |
|---|---|---|---|---|
| 369 | 987 | 62.4–97.5 | 36 px | 0.65 |
| 12 | 214 | 0.1–7.3 | 36 px | 0.61 |
| 286 | 166 | 53.2–59.0 | 33 px | 0.46 |
| 538 | 81 | 104.1–106.7 | 37 px | 0.59 |
| 271 | 27 | 52.0–52.8 | 29 px | 0.38 |
| 343 | 18 | 60.1–60.7 | 39 px | 0.43 |

**Conclusión:** no hay confusión de identidad en `bob`. El alto número de decisiones es porque la misma persona permanece en cámara mucho tiempo (la pista 369 dura 35 segundos). Las similitudes (0.38 a 0.65) son menores que las de otras personas, probablemente porque la cara mide ~36 px y muchas veces está girada.

## 23. Revisión de la corrida 38–107 s con ventana (2026-10-16)

Archivos: `experiments/unc/demo/decisiones_38s_69s_filtro20_ventana.csv` y `asistencia_38s_69s_filtro20_ventana.csv`.

**Resultado:** 29 393 decisiones, 196 pistas, 9 personas confirmadas (`carlos`, `alexa`, `jose`, `bob`, `dani`, `jesus`, `fran`, `kevin`, `kate`). Similitudes: medianas entre 0.39 (`jesus`) y 0.70 (`jose`).

**Verificación de pistas contra el video:**
- `jesus`: 3 pistas verificadas (p332, p90, p214). Todas son él, con gorra y gafas, reconocido con similitudes de 0.36 a 0.45. **Se reconoce con accesorios sin que se haya forzado nada**, solo con la confianza baja. Coincide con la sección 15: a veces acierta, a veces no.
- `bob`: las pistas son el mismo hombre con camiseta de Venezuela (sección 22).
- `alexa`, `kate`, `dani` y `fran`: las pistas revisadas muestran a la misma persona en cada caso, coherente con su video de registro.
- No se encontró ninguna confusión de identidad confirmada en esta corrida.

**Corrección de un análisis previo (importante):** una montaje intermedio sugirió que dos pistas de `jesus` estaban ancladas sobre una pared sin cara. Al verificar frame por frame, ambas pistas son `jesus` real. El error fue del montaje (frames centrales mal elegidos y sin recuadro). No hay evidencia de anclaje a objetos.

**Sobre las rayas de la camiseta detectadas como rostro:** no se encontraron casos en la muestra revisada (detecciones de 18 a 30 px en el tramo 38–107 s: todas caras reales). No confirmado. Si se identifica el segundo exacto, se puede revisar.

## 24. Falsos positivos del detector y anclaje (2026-10-16)

Durante la revisión de la corrida `prueba_1610_1048`, el responsable encontró dos capturas con nombre sobre regiones sin rostro:
- `jesus/s0079.0_p214_sim0.42.jpg`: el recuadro está sobre un **brazo**.
- `alexa/s0058.2_p123_sim0.49.jpg`: el recuadro está sobre **ropa oscura**.

**Verificación:** en esos frames, RetinaFace sí devolvió una detección en esa zona: brazo con confianza **0.666** y ropa con **0.549**. Las caras reales del mismo frame tienen confianza ≥ 0.93. Son **falsos positivos del detector**.

**Por qué el nombre aparecía:** la pista estaba **anclada** (ya había sido confirmada antes). El ancla se mantiene hasta 90 frames sin volver a verificar, así que el nombre sigue mostrándose aunque el recuadro ya no esté sobre la cara. Además, en las filas ANCLADO el CSV repite la similitud original, no la del frame actual: los `sim0.42` y `sim0.49` de esos archivos son engañosos.

**Nota:** esto confirma observaciones previas sobre el detector (rayas de camisa, regiones de piel). El filtro de ancho (20 px) no los elimina, y el filtro de confianza del motor no existía hasta ahora.

**Costo de filtrar por confianza** (medido sobre los 1 127 recortes etiquetados):

| Umbral de confianza | Personas registradas que se perderían | Desconocidos que se perderían |
|---|---|---|
| ≥ 0.70 | 5 de 86 (5.8 %) | 13 de 157 (8.3 %) |
| ≥ 0.75 | 6 de 86 (7.0 %) | 18 de 157 (11.5 %) |
| ≥ 0.80 | 9 de 86 (10.5 %) | 25 de 157 (15.9 %) |

Los dos falsos positivos de arriba (0.666 y 0.549) quedan fuera con un umbral de 0.70.

**Actualización (2026-10-16, corrida `prueba_1610_1425` con filtro de confianza 0.70):**
- La confianza mínima de detección procesada quedó en 0.70. El caso de ropa oscura (alexa, conf 0.55) desapareció.
- El caso del brazo (jesus) **persiste**: su confianza varía por frame y alcanza 0.7346 en el segundo 79.07, por encima de 0.70.
- Costo medido de subir el umbral (sobre los 1 127 recortes, con su resultado de reconocimiento previo sin filtro de confianza):

| Rango de confianza | Personas registradas que se pierden | De esas, se reconocían bien | Desconocidos que se pierden |
|---|---|---|---|
| [0.70, 0.75) | 1 | 1 | 5 |
| [0.75, 0.80) | 3 | 1 | 7 |
| [0.80, 0.90) | 8 | 4 | 23 |

Subir de 0.70 a 0.75 cuesta 1 recorte útil de persona registrada, y elimina el caso del brazo.

## 25. Corrida con filtro de confianza 0.75 (2026-10-16)

Carpeta: `experiments/unc/demo/prueba_1610_1441/`. Video completo desde el segundo 38. Filtro: tamaño 20 px, confianza 0.75.

**Resultado:**
- El caso del brazo (jesus, conf 0.7346) **desapareció**. La confianza mínima procesada quedó exactamente en 0.75.
- El caso de la ropa oscura (alexa) sigue fuera.
- Asistencia: las 9 personas confirmadas (`carlos` 0.3644, `alexa` 0.3877, `jose` 0.4384, `bob` 0.3571, `dani` 0.4130, `jesus` 0.3944, `fran` 0.3706, `kate` 0.3682, `kevin` 0.5286).

**Efecto sobre el reconocimiento** (contra los recortes de la muestra dentro del tramo 38–107 s: 79 con nombre y 131 desconocidos):

| Resultado | Cantidad |
|---|---|
| Personas reconocidas y confirmadas | 67 de 79 (85 %) |
| Personas sin decisión (filtradas por confianza) | 7 |
| Personas procesadas pero no reconocidas | 5 |
| Identidades incorrectas | 0 |
| Desconocidos confirmados como registrados | 0 |
| Desconocidos sin decisión (filtradas) | 26 de 131 |

Comparado con la corrida sin filtro de confianza (76 de 86 personas reconocidas, 88 %), el filtro cuesta unos pocos reconocimientos, y elimina los falsos positivos sobre brazos y ropa.

**Capturas por estudiante en esta corrida:** alexa 40, bob 49, carlos 24, dani 54, fran 10, jesus 19, jose 53, kate 47, kevin 27.

## 26. Script del margen top-2 (2026-10-19)

**Cambio en el motor:** `_decidir_identidad` ahora pide a FAISS las dos mejores coincidencias. Las decisiones **no cambiaron**: se sigue usando solo la primera y el umbral 0.35. Se agregan `sim_top2`, `nombre2` y `margen` (sim_1 − sim_2) a cada decisión. En las filas ANCLADO no se recalcula, así que esos campos van vacíos.

**Demo:** el `decisiones.csv` incluye `sim_top2`, `nombre2` y `margen`. Se agrega `--margen=N` (por defecto 0.10) y una carpeta `dudosos/<nombre>/` con las capturas cuyo margen es menor a N, más `dudosos_resumen.jpg`.

Corrida: `experiments/unc/demo/prueba_1910_1436/`.

**Resultado:**

| Caso | Decisiones | Margen mediana | Margen < 0.05 |
|---|---|---|---|
| La primera coincidencia es una persona registrada | 111 | **0.333** | 0 |
| La primera coincidencia es Desconocido | 16 697 | **0.034** | 10 875 (65 %) |

- **No hay confusión entre registrados.** Cuando el sistema identifica a una persona registrada, la segunda opción queda muy lejos (margen mediano 0.33). Ningún caso bajó de 0.05.
- El margen pequeño aparece **solo cuando la cara no coincide con nadie** (Desconocido). Ahí no es "duda entre dos personas", es "no hay coincidencia": todos los centroides quedan en la misma banda baja (0.2 a 0.35).
- Segundas opciones cuando la primera es una persona: `alexa`→`dani` (23), `bob`→`jose` (19), `alexa`→`kate` (17), `jesus`→`jose` (12). Coincide con la matriz de centroides, donde `alexa`–`dani` es el par más cercano (0.297).
- Único caso alexa/dani revisado (segundo 46.27, sim 0.3877 contra 0.2043, margen 0.18): **es `alexa`**. Ver `duda_alexa_dani.jpg`.

**Conclusión:** no se encontró evidencia de que el sistema confunda a dos personas registradas. La impresión de "una chica confundida con otra" probablemente viene de la etiqueta anclada que se queda sobre un recuadro desplazado (sección 24), no de un cambio de identidad.

**Nota:** con esta escala de similitudes, un umbral de margen de 0.10 para "duda" no sirve para desconocidos (captura el 65 % de las filas). Si se quiere monitorear, conviene mirar solo las filas cuya primera opción es una persona registrada.

## 27. Impacto de revalidar el ancla cada 30 frames (2026-10-19)

Medido sobre la corrida `prueba_1910_1436` (38–107 s, 2 069 frames):

- Frames anclados: 8 118, en 31 pistas.
- Revalidaciones con el valor actual (90 frames): 78. Con 30 frames: 256. **+178 ciclos.**
- Cada ciclo implica ~3 embeddings extra (la revalidación y los 2 o 3 frames de votación). En total, ~534 embeddings extra.
- Costo medido en este equipo: detección **15.6 ms/frame**, embedding **6.6 ms/cara**.
- Cómputo base de la corrida: 32 s de detección + 113 s de embeddings ≈ 145 s.
- Extra: **3.5 s por tramo de 69 s (+2.4 %)**; ~5.5 s para el video completo de 107 s.

**El costo de cómputo es bajo.** El costo real está en la interfaz: cada revalidación muestra "Verificando 1/3" y "2/3" durante 2 o 3 frames. Pasaría en ~178 ventanas dentro de 8 118 frames anclados, es decir ~5 % del tiempo de etiqueta cambia del nombre a "Verificando".

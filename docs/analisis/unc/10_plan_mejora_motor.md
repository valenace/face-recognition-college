# Plan de perfeccionamiento del motor de reconocimiento

**Contexto:** falso positivo de identidad del 2026-10-21 (ver `reporte_dani_confusion.md`):
una estudiante no registrada confirmada como `dani` con similitud 0.325, mientras la
dani real daba 0.628–0.649.

**Regla que gobierna todo el plan:** el error no es de umbral, es de **comparación de
identidades**. Ningún cambio de parámetros lo arregla; hay que cambiar cómo se decide a
quién pertenece una cara.

---

## 0. Qué NO vamos a hacer (descartado con datos)

| Propuesta | Por qué se descarta |
|---|---|
| Subir el umbral global a 0.35–0.40 | En el tramo medido, `jesus` llega como máximo a 0.388: se perdería **por completo**. También se pierden partes de `alexa` (0.332), `kate` (0.326) y `bob` (0.301). |
| Usar el margen top-2 como regla de rechazo | El falso positivo tenía margen **0.118**, mayor que el de muchos aciertos. El margen no discrimina. |
| Bajar el número de votes | La desconocida es estable: gana los 3 votos sin esfuerzo. |
| Alargar el ancla | Empeora: mantiene visible la etiqueta equivocada más tiempo. |

---

## Fase 1 — Resolver conflictos entre pistas simultáneas  ✅ Hecha

**Estado:** implementada el 2026-10-21. Medida sobre el tramo 36–107 s del video 1:
FP de identidad **2 → 0**, los 9 registrados conservan sus similitudes. Ver
`09_incidente_identidad_dani.md` §9.

**Límite:** no corrige el falso positivo del video 3 (una sola pista reclamando `dani`,
sin conflicto simultáneo). Ese caso requiere la Fase 3.

**Idea:** dos pistas activas en el mismo frame **no pueden ser la misma persona**. Si
ambas reclaman `dani`, una está equivocada.

**Implementación**
- Archivo: `scripts/asistencia_unificada_optimizada.py`, dentro de `procesar_frame`.
- Tras calcular las decisiones, agrupar las que están en `CONFIRMADO`/`ANCLADO` por
  nombre. Si un nombre aparece en más de una pista:
  1. conservar la pista con mayor similitud (desempate: la que lleva más tiempo anclada);
  2. si la diferencia es **< 0.05**, no decidir: dejar ambas etiquetas y marcarlas como
     ambiguas (evitacastigar fragmentos legítimos del tracker);
  3. en los casos claros, **degradar la perdedora a `DESCONOCIDO`**, borrar su ancla y
     reiniciar sus votos.

**Por qué funciona aquí:** pista 70 (0.628) contra pista 117 (0.325): diferencia 0.30,
muy por encima del corte de 0.05. El error desaparece sin tocar a nadie más.

**Riesgos**
- El tracker a veces duplica momentáneamente a la misma persona en dos pistas. El corte
  de 0.05 evita degradar en ese caso, y cuando hay empate se deja el registro de
  ambigüedad para revisarlo.
- Puede dar falsos positivos en escenas con gemelos o personas muy parecidas: es el
  precio de obligar a que las identidades sean exclusivas.

**Cómo medirlo**
- Tramo video 1 (36–107 s): FP de identidad **2 → 0**, y los 9 estudiantes siguen
  confirmándose (cobertura 9/9).
- Video 3 completo (11 partes): el FP de la parte 7 (0.336) → 0, cobertura 8/8.

**Esfuerzo estimado:** 2–3 h.

---

## Fase 2 — Observabilidad de las decisiones  ✅ Hecha (parcial)

**Estado:** el campo `motivo` está implementado (motor, CSV de las corridas y
`ver_flujo`). Queda pendiente mostrar el estado `AMBIGUO` de forma destacada en la vista
web; hoy se registra como motivo (`identidad_ambigua_con_pista_N`).

**Implementación**
1. **Dejar de "subir" la etiqueta anclada.** Hoy, en cada revalidación, el ancla se
   actualiza con la similitud del frame actual, y por eso la etiqueta de la dani real
   subía de 0.41 a 0.55 en pantalla. Fijar el ancla al valor de la confirmación y, si
   quieres, mostrar aparte la similitud actual.
2. **Escribir el motivo de cada degradación** en `decisiones.csv`: nuevo campo
   `motivo` con valores como `conflico_con_pista_70`, `ancla_caida`, `votos_insuficientes`.
3. **Añadir el estado `AMBIGUO`** a la vista en vivo, para que el profesor vea que dos
   caras Tobacco la misma identidad y pueda revisar.

**Por qué:** sin esto, el próximo caso同等 será igual de difícil de diagnosticar.

**Esfuerzo estimado:** 2 h.

---

## Fase 3 — Umbral adaptativo por persona  ❌ Rechazada con datos

**Resultado del cálculo (2026-10-21, `experiments/unc/fase9_umbrales.py`):**

Las distribuciones del **registro** se separan bien: las fotos propias dan similitud
mediana 0.70–0.81 con su centroide, y las ajenas no pasan de 0.34. Los umbrales
propuestos irían de 0.32 (`jesus`) a 0.50 (`kate`).

**Pero no se sostienen en el aula.** Los umbrales calculados sobre el registro se
aplicaron a las confirmaciones reales del video 3:

| Persona | Umbral propuesto | % de sus confirmaciones que lo supera | Resultado |
|---|---|---|---|
| fran | 0.474 | **0.0 %** | **se pierde por completo** |
| kate | 0.503 | **0.0 %** | **se pierde por completo** |
| alexa | 0.449 | 14.9 % | cobertura muy reducida |
| dani | 0.445 | 41.2 % | cobertura reducida |
| carlos | 0.446 | 61.4 % | aceptable |
| kevin | 0.367 | 63.1 % | aceptable |
| bob | 0.473 | 67.7 % | aceptable |
| jesus | 0.324 | 84.0 % | aceptable |

**Por qué falla:** hay un salto de dominio entre el registro (fotos de cerca, 224 px
alineadas) y el aula (caras de 30–80 px, de perfil, con gorra). En el aula `fran` vive
en 0.31–0.34 y `kate` en 0.31–0.43; con sus umbrales de registro no confirmarían nunca.

Además, incluso usando solo las confirmaciones reales (no las ancladas), **todos** los
estudiantes tienen confirmaciones entre 0.30 y 0.34, que es exactamente donde cae el
falso positivo (0.336). **El falso positivo está dentro de la distribución genuina.**

**Conclusión:** un umbral escalar — global o por persona — no puede separar este caso.

---

## Fase 3b — Qué sí muestra la evidencia: el centroide es el problema

Se comparó la cara del falso positivo contra:
- el **centroide** de dani: **0.336**;
- las **50 fotos individuales** de dani: máximo **0.303**, top-3 medio **0.291**, mediana 0.212.

Es decir: el centroide (un promedio) queda más cerca de una cara ajena que cualquiera de
las fotos reales de la persona. Es el conocido efecto *hub* de los centroides.

Y con caras reales de los estudiantes (4 capturas de cada uno, video 3):

| Persona | Sim. media a sus 50 fotos | Máx. | Top-3 |
|---|---|---|---|
| bob | 0.396 | 0.516 | 0.508 |
| jesus | 0.268 | 0.466 | 0.456 |
| kevin | 0.292 | 0.448 | 0.436 |
| carlos | 0.310 | 0.425 | 0.408 |
| dani | 0.289 | 0.407 | 0.399 |
| alexa | 0.263 | 0.388 | 0.382 |
| kate | 0.264 | 0.341 | 0.335 |
| fran | 0.197 | 0.318 | 0.304 |
| **falso positivo** | — | **0.303** | **0.291** |

Con `top-3` sobre las fotos, el falso positivo (0.291) queda **por debajo de todos los
estudiantes reales**, incluido `fran` (0.304). El margen con `fran` es estrecho (0.013),
pero es la primera regla que separa el caso sin sacrificar a nadie.

**Esto convierte la Fase 4 (multi-prototipo) en la solución recomendada**, no en un plan B.

---

## Fase 3c — Alternativa sin tocar el motor: media podada ❌ Rechazada

Se probó si bastaba con **cambiar el vector de referencia** (sin tocar el motor):
descartar las fotos más atípicas y recalcular la media. Resultados sobre la cara del
falso positivo y 6 capturas reales de dani:

| Variante | Fotos | FP | dani (media) | dani (mín) |
|---|---|---|---|---|
| media actual (50) | 50 | **0.302** | 0.437 | 0.282 |
| sin el 25 % más lejano | 38 | 0.310 | 0.428 | 0.256 |
| sin 25 % lejano y 25 % cercano | 26 | 0.301 | 0.409 | 0.252 |
| sin 15 % lejano y 15 % cercano | 36 | 0.309 | 0.419 | 0.261 |

**No hay mejora:** el falso positivo se queda en ~0.30 en todas las variantes, y las
capturas reales incluso bajan. El efecto hub no se corrige recortando fotos: está en la
propia media.

> Nota de método: una primera medición dio 0.270 para la variante "sin 25 % lejano y
> cercano", pero era un error de índice (`np.sort(np.argsort(...)` devuelve el orden
> natural). Repetida correctamente, da 0.301. Se documenta para que no se repita.

**Conclusión:** sin tocar el motor, no queda ninguna variante de vector de referencia que
separe el caso.

---

## Fase 4 — Varios prototipos por persona (recomendada)

**Estado:** siguiente paso. La evidencia de la Fase 3b muestra que el centroide es el
problema: infla la similitud de las caras ajenas (efecto *hub*).

**Contexto:** en la primera fase de este proyecto se decidió usar **un centroide por
persona** y se descartó explícitamente la "galería de prototipos". Los datos de hoy
muestran el costo de esa decisión: la cara de una desconocida queda a 0.336 del
centroide de `dani`, pero solo a 0.303 de la foto individual más parecida de dani.

**Implementación**
- Al registrar, agrupar las 50 fotos en 3–5 clústeres (k-means) y guardar **un centroide
  por clúster** (personas con goggles, gorro, perfil, etc.).
- La búsqueda FAISS devuelve el mejor de todos los prototipos; la etiqueta es la del
  dueño del prototipo.
- La fila por persona pasa a ser una tabla `face_prototype` (o una columna JSON con
  varios vectores).

**Cuándo hacerlo:** si la Fase 3 no basta, o si el grupo real tiene parejas/gemelos
parecidos. Requiere volver a registrar a los 9 (o derivar los prototipos de las fotos que
ya existen, que es viable).

**Esfuerzo estimado:** 1–2 días.

---

## Fase 5 — Evaluación de regresión permanente

**Problema:** los errores de identidad solo aparecen cuando hay un video con personas
reales y alguien mira la pantalla. No hay forma automática de detectarlos.

**Implementación**
1. **Fixture de pistas etiquetadas.** Guardar una tabla pequeña de pistas conocidas:
   `video, segundo, pista, identidad_verificada`. Con lo obtenido hoy ya hay 39 pistas
   etiquetadas del tramo 36–107 s del video 1 (37 correctas, 2 falsas).
2. **Script de regresión** `experiments/unc/regresion_identidad.py` que:
   - procesa los tramos etiquetados de los videos 1 y 3;
   - calcula precisión por pista, cobertura y falsos positivos;
   - falla si el FP supera un máximo acordado.
3. **Ejecutarlo antes de cada cambio** en el motor, como los tests.

**Esfuerzo estimado:** 3 h.

---

## Orden recomendado

| # | Fase | Esfuerzo | Estado |
|---|---|---|---|
| 1 | Conflictos entre pistas | 45–60 min | ✅ Hecha; resuelve el incidente del video 1 |
| 2 | Observabilidad (`motivo`) | 45–60 min | ✅ Hecha |
| 3 | Umbral adaptativo por persona | 1.5–2 h | ❌ Rechazada: pierde a `fran` y `kate` |
| 3c | Media podada (sin tocar el motor) | 45 min | ❌ Rechazada: no mueve el falso positivo |
| 4 | Multi-prototipo (top-3 sobre las fotos) | 3–4 h | ⛔ Fuera del alcance actual: exige cambiar la comparación del motor |
| 5 | Regresión permanente | 1–1.5 h | ✅ Hecha (`regresion_identidad.py`) |

**Restricción vigente:** el motor no se reescribe y no se cambian librerías ni modelos.
Con esa restricción, las mejoras posibles son **ajustes pequeños** (como la Fase 1) y
cambios de presentación, no de algoritmo.

**Implementado además:** la **banda de revisión** (0.30–0.40 se muestra como
`posible <nombre>` en amarillo). No cambia decisiones ni asistencia; solo evita afirmar
una identidad que el sistema no puede garantizar.

**Lo único que queda por mejorar** es el falso positivo del video 3 (parte 7, pista 3),
y la única vía con evidencia es la Fase 4. Queda registrado como límite conocido en
`regresion_pistas.csv` (`baseline=1`).

---

## Criterios de aceptación

- Tramo video 1 (36–107 s): **0 falsos positivos de identidad**, cobertura **9/9**. ✅
  Cumplido con la Fase 1.
- Video 3 (11 partes): **0 falsos positivos**, cobertura **8/8** de los presentes y
  `jose` (ausente) en **0**. ⏳ Pendiente: requiere la Fase 3 (hoy sigue el FP de la
  parte 7 con similitud 0.336).
- `jesus` conserva confirmaciones (su techo real es ~0.39).
- Los tiempos de confirmación no suben de 0.5 s en promedio.
- Los números ya publicados en las fases 4–8 se conservan como histórico; cualquier
  cambio de criterio se documenta como revisión, sin reescribir resultados anteriores.
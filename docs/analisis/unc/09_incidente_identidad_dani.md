# Reporte de incidente: una estudiante no registrada confirmada como `dani`

**Fecha:** 2026-10-21
**Video:** `data/unc/salon/VID_20260907_102851.mp4` (video 1), segundo 36 → 107
**Motor:** `scripts/asistencia_unificada_optimizada.py` (umbral 0.30, 3 votos, ancla 30)
**Base vectorial:** PostgreSQL + pgvector, 9 centroides ArcFace de 512 d
**Detectado por:** el usuario, en la vista anotada (`ver_flujo`)

---

## 1. Resumen

En la vista en vivo del motor aparecieron **dos rostros etiquetados como `dani` al
mismo tiempo**. El usuario verificó que **uno de los dos no es dani**: es una estudiante
**no registrada**.

Al reproducir el tramo con datos, el motor confirma la identidad de la desconocida en
**2 pistas** (117 y 167), con similitudes de **0.325 y 0.301** frente al centroide de
`dani`. La dani real quedó en las pistas 70 y 249, con similitudes de **0.628 y 0.649**.

**Conclusión:** es un **falso positivo de identidad** causado por el modelo de
"un centroide por persona": el embedding de la desconocida cae dentro del vecindario del
centroide de `dani`. No es un error de umbral ni de lógica de votes.

---

## 2. Qué se midió

Tramo procesado: **2 129 frames** (36.0 s → 107.0 s).

| Métrica | Valor |
|---|---|
| Decisiones de rostro | 25 821 |
| Pistas detectadas | 179 |
| Pistas que recibieron nombre | 39 |
| Pistas **incorrectas** (persona no registrada) | **2** (117 y 167) |
| Precisión a nivel de pista | **37/39 = 94.9 %** |

Distribución de las 39 pistas confirmadas por identidad asignada:
`bob` 7, `jesus` 6, `dani` 4 (2 correctas + 2 falsas), `jose` 4, `alexa` 4,
`kate` 3, `kevin` 4, `carlos` 3, `fran` 1.

> Nota: una persona puede generar varias pistas (el tracker se rompe con oclusiones).
> El denominador "pistas" no equivale a "personas"; por eso el error se reporta también
> como "1 persona no registrada confirmada como dani".

---

## 3. Las cuatro pistas etiquetadas como `dani`

| Pista | Frames | Ancho | Intervalo (s) | Confirmada (s) | Sim. al confirmar | Sim. máx. | Frames con nombre | Veredicto |
|---|---|---|---|---|---|---|---|---|
| 70 | 984 | 73 px | 52.74 – 85.81 | 52.97 | 0.413 | **0.628** | 806 | dani real |
| 117 | 133 | 38 px | 57.34 – 63.64 | 61.37 | 0.325 | **0.330** | 30 | **no registrada** |
| 167 | 25 | 41 px | 63.10 – 63.94 | 63.50 | 0.301 | **0.330** | 13 | **no registrada** |
| 249 | 599 | 87 px | 86.77 – 106.77 | 87.57 | 0.367 | **0.649** | 536 | dani real |

**Pistas 70 y 249 no se solapan en el tiempo** (70 termina en 85.81 s, 249 empieza en
86.77 s). Son la misma persona en dos tramos: el tracker perdió la pista y la recuperó
después. La appearance cambió (pelo recogido), lo que explica que la similitud con el
centroide sea 0.649.

**Las pistas 117 y 167 sí se solapan con la 70** (57.3–63.6 s y 63.1–63.9 s dentro de
52.7–85.8 s). Ahí es donde la vista muestra dos cajas con el mismo nombre.

### 3.1 A quién se parece cada cara (top-5 en su mejor frame)

| Pista | 1º | 2º | 3º | 4º | 5º |
|---|---|---|---|---|---|
| 70 (dani real) | **dani 0.627** | alexa 0.187 | carlos 0.077 | bob 0.073 | fran 0.059 |
| 249 (dani real) | **dani 0.649** | alexa 0.092 | jesus 0.075 | carlos 0.073 | fran 0.049 |
| 117 (desconocida) | **dani 0.325** | jesus 0.207 | alexa 0.118 | carlos 0.079 | jose 0.064 |
| 167 (desconocida) | **dani 0.301** | alexa 0.153 | (resto < 0.1) | | |

El margen top-2 (primera menos segunda) es **grande** en el caso falso: 0.118 y 0.153.
Es decir, la desconocida "gana" claramente a `dani` frente a todos los demás. **Por eso
el margen top-2, que hoy solo se registra, no habría detectado este error.**

---

## 4. Mecánica del fallo

1. **Comparación de un solo vector.** El motor compara el embedding de la cara contra
   **un centroide por persona**. La desconocida tiene un embedding a 0.325 del centroide
   de dani; no hay forma de saber que es otra persona.
2. **Umbral 0.30 superado.** 0.325 > 0.30, así que entra en el conteo de votos.
3. **3 votos consecutivos.** La cara es estable en su apariencia: 30 frames seguidos por
   encima de 0.30 → `CONFIRMADO`.
4. **Ancla de 30 frames.** El nombre queda congelado aunque la cara cambie.
5. **Reanclaje.** En cada revalidación el ancla se actualiza con la similitud del frame
   actual. Esto explica que en pantalla la etiqueta de la dani real subiera hasta 0.55
   y la de la desconocida mostrara 0.32.

### 4.1 Por qué la etiqueta se mantuvo ~1.4 s

Las pistas 117 y 167 dijeron "dani" durante 30 + 13 = **43 frames ≈ 1.4 s** de las 2 129
del tramo (2 %).

### 4.2 Por qué la asistencia no se vio afectada

La dani real ya estaba registrada en el mismo tramo (pista 70, s52.97). La asistencia se
registra **una vez por persona y sesión** (`asistencia_hoy`), así que la etiqueta
equivocada **no generó una asistencia adicional ni borró la correcta**.

**El daño real fue de visualización:** una etiqueta incorrecta en pantalla durante 1.4 s,
y un falso positivo a nivel de identidad (2 de 39 pistas confirmadas).

---

## 5. Relación con hallazgos anteriores

| Hallazgo previo | Contexto | Relación con este incidente |
|---|---|---|
| Estudiante no registrada confirmada como `dani` (video 3, parte 7, pista 3) | Fase 8, oct 2026 | **Mismo fenómeno**: una no registrada sobre el centroide de dani. Allí la similitud fue 0.336, aquí 0.325/0.301. |
| Margen top-2 mediano 0.333 en aciertos, ninguno < 0.05 | Fase 8 | El margen **no discrimina**: el falso positivo tenía margen 0.118, bien por encima de los aciertos. |
| `fran`, `kate`, `jesus` con similitudes medias 0.32–0.39 | Fases 3 y 8 | Indican que un global más alto de umbral **no** es la solución: eliminaría a `jesus` entero (su máximo en este tramo es 0.388). |

---

## 6. Qué NO es la causa (descartado)

| Hipótesis | Veredicto |
|---|---|
| Umbral 0.30 demasiado bajo | **Descartado**: la dani real da 0.63–0.65; para rechazar a la desconocida habría que subir a ~0.35–0.40, lo que elimina a `jesus` y parte de `alexa`, `kate` y `bob`. |
| Margen top-2太小 | **Descartado**: el falso tiene margen 0.118, mayor que el de muchos aciertos. |
| Anchura de cara (38–41 px) | **No es la causa**: dani real es de 73–87 px y sí funciona; pero sí explica por qué las pistas falsas son cortas y unstable. |
| Pocos votes | **No**: la desconocida es estable, gana los 3 votos sin esfuerzo. |
| Ancla demasiado larga (30 frames) | **Agravante, no causa**: mantiene la etiqueta visible más tiempo del debido. |

---

## 7. Reproducir

```bash
# Ver el tramo con detalle, cara por cara
venv/bin/python manage.py ver_flujo \
  --video data/unc/salon/VID_20260907_102851.mp4 \
  --desde 36 --hasta 107 --detalle
```

Lascomparison de las caras contra los 9 centroides se puede regenerar con el script
`experiments/unc/fase7_agrupar_desconocidos.py` (misma mecánica de embedding + FAISS).

---

## 8. Recomendación resumida

El arreglo tiene que actuar sobre **la comparación de identidades**, no sobre el umbral.
Dos medidas baratas y una de fondo:

1. **Resolver conflictos entre pistas simultáneas**: dos pistas activas en el mismo frame
   no pueden ser la misma persona. Se conserva la de mayor similitud y la otra vuelve a
   `Desconocido`. En este incidente elimina el error por completo.
2. **Exigir similitud sostenida para mantener el ancla**: si la mediana de los últimos
   N frames no supera el umbral + margen, se cae el ancla. La desconocida se sostiene en
   0.32; la dani real en 0.63.
3. **Umbral adaptativo por persona**, derivado de las estadísticas de cada registro
   (distribución intra vs. inter). Es la solución de fondo y evita repetir el trabajo
   cuando se mezclen personas parecidos.

El detalle está en `10_plan_mejora_motor.md`.

---

## 9. Corrección aplicada (2026-10-21)

Se implementó la **Fase 1** del plan en `scripts/asistencia_unificada_optimizada.py`:

- Nuevo método `_resolver_conflictos`, llamado al final de `procesar_frame`.
- Si dos pistas activas comparten nombre, se conserva la de mayor similitud.
- Si la diferencia es **≥ 0.05**, la perdedora pasa a `DESCONOCIDO`, se le borra el ancla
  y se le reinician los votos.
- Se registra **`motivo`** en cada decisión (`conflicto_con_pista_N` o
  `identidad_ambigua_con_pista_N`), y se añadió como columna en los CSV de las corridas.
- La pista perdedora queda **vetada** 90 frames mientras la ganadora siga anclada, para
  que no vuelva a confirmarse en el frame siguiente.

### 9.1 Resultado medido en el mismo tramo (36–107 s del video 1)

| Métrica | Antes | Después |
|---|---|---|
| Pistas con nombre | 39 | 35 |
| Pistas de la desconocida etiquetadas como dani | **2** (117 y 167) | **0** |
| Falsos positivos de identidad | **2** | **0** |
| Registrados confirmados | 9/9 | **9/9** |
| Similitudes de los 9 registrados | — | idénticas (0.306–0.529) |

Las 4 pistas que perdieron el nombre son: 117 y 167 (la desconocida) y 149 y 169
(duplicados de `jesus` y `jose`, que seguían etiquetados por sus pistas ganadoras).
**Ningún estudiante perdió su registro de asistencia.**

Tests nuevos: `tests/test_registro.py::TestConflictosIdentidad` (3 casos: degradación,
diferencia pequeña y nombres distintos).

## 10. Banda de revisión y regresión (2026-10-21)

### 10.1 Banda de revisión (presentación)

El motor marca `revision=True` cuando una identidad confirmada tiene similitud entre
0.30 y **0.40**. No cambia la decisión ni la asistencia: solo la presentación.

- En pantalla, esas caras se dibujan en **amarillo** con el texto `posible <nombre>`.
- En la vista en vivo hay una leyenda: verde confirmada, amarillo posible, rojo
  desconocido.
- En `ver_flujo` se muestran como `CONFIRMADO (posible)`.

Esto es honesto con el límite medido: en la banda 0.30–0.40 el sistema **no puede
afirmar** la identidad, y es exactamente donde cayó el falso positivo (0.333–0.336).

### 10.2 Regresión de identidad

`experiments/unc/regresion_identidad.py` recorre las pistas etiquetadas de
`regresion_pistas.csv` y compara la identidad asignada con la esperada:

```
Aciertos: 11 | errores: 0 | falsos positivos: 0 | perdidas: 0 | conocidos (baseline): 1
RESULTADO: OK
```

- 9 pistas largas correctas del video 1 (una por estudiante).
- 2 falsos positivos históricos del video 1 (pistas 117 y 167), corregidos por la Fase 1.
- 1 límite conocido: el falso positivo del video 3 (parte 7, pista 3), marcado como
  `baseline=1` porque requiere la Fase 4 (multi-prototipo), fuera del alcance actual.

Si aparece un fallo nuevo, el script sale con código 1. Si el límite conocido mejora,
avisa para actualizar el baseline.

### 9.2 Alcance de la corrección (importante)

El fix resuelve el caso reportado porque había **dos pistas simultáneas** reclamando
`dani`. **No resuelve el falso positivo del video 3** (parte 7, pista 3): allí la
desconocida era la única pista que reclamaba `dani` en ese momento (la dani real no
estaba en cuadro), así que no existe conflicto que resolver.

Verificación: la corrida completa del video 3 con el fix (`prueba_2110_1613_video3`)
produce un `resumen_por_parte.csv` **idéntico** al anterior y no registra ningún motivo
de conflicto.

Ese caso necesita la **Fase 3** (umbral adaptativo por persona): el umbral de `dani`
debe quedar por encima de 0.333, que es donde entra la desconocida.
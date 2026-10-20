# Plan Completo de Experimentos — Pipeline de Reconocimiento Facial
**Proyecto:** CV College — Sistema de Asistencia Biométrica  
**Fecha de creación:** 2026-09-01  
**Última actualización:** 2026-09-01  
**Estado global:** 🔵 En planificación

---

## Resumen del Objetivo

El objetivo de estos experimentos es **demostrar científicamente** que el pipeline de
reconocimiento facial puede operar en un salón de clases real con:

| Métrica | Meta |
|---|---|
| Tasa de Verdadero Positivo (TPR) | ≥ 85 % |
| Tasa de Falso Positivo (FPR) | ≤ 2 % |
| F1-Score global | ≥ 0.90 |
| Tiempo hasta primera confirmación de asistencia | ≤ 3 segundos |

Cada experimento sigue el método científico:

```
Hipótesis → Diseño del experimento → Ejecución → Medición → Análisis → Conclusión
```

---

## Mapa General de Fases

```
FASE 1 (✅ COMPLETADA)
  Diagnóstico baseline sin cambios
       ↓
FASE 2 (🔵 PRÓXIMA)
  Re-registro de personas con pocos frames
       ↓
FASE 3
  Curva ROC → Umbral óptimo
       ↓
FASE 4
  Mejoras al código del motor
       ↓
FASE 5
  Evaluación cuantitativa multi-video
       ↓
FASE 6
  Documentación y reporte final
```

---

## FASE 1 — Diagnóstico Inicial (✅ COMPLETADA)

### Descripción
Antes de cambiar cualquier cosa, medimos y documentamos el estado actual del sistema.
Esta es la **línea base** contra la que compararemos todas las mejoras futuras.

### ¿Qué hicimos?
- Inventario completo del dataset (videos, frames por persona, duraciones)
- Verificación de la DB de embeddings (normas, dimensiones)
- Cálculo de la matriz de similitudes inter-clase
- Lectura del CSV de asistencia histórico para observar similitudes reales

### Resultados Obtenidos

**Hallazgos críticos:**

| Hallazgo | Valor | Implicación |
|---|---|---|
| hanny — frames en DB | 11 | Centroide inestable |
| hector — frames en DB | 23 | Límite inferior |
| Similitudes de ronald en CSV | 0.36–0.38 | Rozando umbral 0.35 |
| Similitud máxima inter-clase | 0.280 (hector↔jorge) | Margen de 0.07 sobre umbral |
| Umbral actual | 0.35 (hardcoded) | Sin validación empírica |

**Archivos generados:** `docs/analisis/01_diagnostico_inicial.md`

---

## FASE 2 — Optimización del Registro

### Estado: 🔵 PRÓXIMA

### Descripción
Re-registrar a las personas con embeddings de baja calidad para darles una base sólida
antes de realizar los experimentos de umbral. También re-generar la base de datos completa
desde cero para que todos los centroides sean comparables.

### Problema a resolver

El experimento de sensibilidad previo (Ángel y Ronald) demostró que:
- Con < 20 muestras diversas el centroide tiene un "valle de precisión" en rangos medios de muestras
- Con ≥ 25 muestras diversas el centroide es robusto al 100% en condiciones del salón

Las personas en riesgo identificadas en Fase 1:

| Persona | Frames actuales | Video disponible | Acción |
|---|---|---|---|
| hanny | 11 | `hanny_video/hanny.mp4` (179 frames, 6s) | Re-registrar |
| hector | 23 | `hector_pineda_video/hector_pineda.mp4` (186 frames, 7.9s) | Re-registrar |
| ronald | 50 | `ronald_delgado_video/ronald_delgado.mp4` + `2ronald_delgado.mp4` | Verificar calidad |
| angel | 37 | `angel_video/angel.mp4` (331 frames, 11s) | Mantener o mejorar |

### Protocolo de Re-registro

Para garantizar diversidad de poses en el video de registro, el estudiante debe:

```
Paso 1: Mira al frente (3 segundos)
Paso 2: Gira suavemente a la IZQUIERDA ~30° y vuelve al centro (2 segundos)
Paso 3: Gira suavemente a la DERECHA ~30° y vuelve al centro (2 segundos)
Paso 4: Inclina la cabeza hacia ARRIBA y vuelve al centro (1 segundo)
Paso 5: Inclina la cabeza hacia ABAJO y vuelve al centro (1 segundo)
Paso 6: Mira al frente nuevamente (1 segundo final)
```

**Total: ~10 segundos @ 30fps = ~300 frames brutos → ~25-50 frames útiles tras el filtro de diversidad**

### ¿Por qué funciona el filtro de diversidad?

El script `registro_video.py` descarta un frame nuevo si su similitud con algún frame ya
guardado supera 0.95. Esto garantiza que cada frame capturado es suficientemente distinto
a todos los anteriores:

```python
# Pseudocódigo del filtro
for vec_nuevo in frames_del_video:
    max_similitud_con_existentes = max(dot(vec_nuevo, v) for v in ya_guardados)
    if max_similitud_con_existentes < 0.95:
        guardar(vec_nuevo)  # Es suficientemente diferente
    else:
        descartar(vec_nuevo)  # Muy parecido a uno ya guardado
```

Con el protocolo de movimiento, el video tiene suficiente variedad para que el filtro capture
~25-50 poses genuinamente distintas en lugar de capturar solo las primeras estáticas.

### Experimento 2A — Comparación Antes/Después del Re-registro

**Hipótesis:** Re-registrar a `hanny` con su video existente pero con un paso de muestreo
más agresivo aumentará la diversidad de sus embeddings y mejorará sus similitudes en el salón.

**Método:**
1. Guardar el centroide actual de `hanny` como referencia
2. Re-correr `registro_video.py` sobre `hanny.mp4` cambiando parámetros de extracción
3. Comparar la similitud del nuevo centroide vs. el antiguo contra frames del salón
4. Medir: `sim(centroide_nuevo, frame_salon)` vs. `sim(centroide_viejo, frame_salon)`

**Resultado esperado:** El centroide nuevo debe dar similitudes 15-30% más altas en el salón.

**Script a ejecutar:**
```bash
python scripts/run_single_video_reg.py hanny data/dataset_cv/videos/hanny_video/hanny.mp4
```

### Experimento 2B — Regenerar DB Completa

**Hipótesis:** Regenerar la DB desde cero (re-procesando todos los frames existentes más los
nuevos) producirá centroides más consistentes y comparables entre personas.

**Método:**
1. Ejecutar `generar_db.py` sobre el dataset actualizado
2. Verificar que todas las normas siguen siendo 1.0
3. Recalcular la matriz de similitudes inter-clase
4. Comparar con la matriz del Diagnóstico Inicial (Fase 1)

**Resultado esperado:** Los centroides de `hanny` y `hector` deben tener mayor similitud con
sus propios frames de salón, sin que las similitudes inter-clase aumenten significativamente.

### Artefactos que genera esta fase

```
docs/analisis/02_resultados_reregistro.md    ← Comparación antes/después
data/database_embeddings.pkl                 ← DB actualizada (sobrescribe)
data/dataset_cv/frames/hanny/               ← Frames renovados
```

---

## FASE 3 — Curva ROC y Selección del Umbral Óptimo

### Estado: ⏳ Pendiente (requiere Fase 2 completada)

### Descripción
Este es el experimento más riguroso del proyecto. El objetivo es encontrar empíricamente
el **umbral de similitud óptimo** que maximice la identificación de alumnos reales mientras
minimiza la aceptación de personas no registradas (falsos positivos).

### Fundamento Teórico

El sistema actualmente usa umbral = 0.35 sin ninguna justificación empírica. El umbral
correcto depende de las distribuciones reales de similitudes en el salón:

```
Distribución intra-clase (misma persona):    generalmente en rango 0.36 – 0.65
Distribución inter-clase (persona diferente): generalmente en rango -0.10 – 0.30
```

El umbral óptimo es el valor que mejor **separa** estas dos distribuciones.

### Diseño del Experimento

**¿Cómo obtenemos "positivos" y "negativos" si no tenemos etiquetas manuales?**

Usamos dos estrategias complementarias:

**Estrategia A — Positivos Genuinos con Ground Truth Automático:**
Para cada persona registrada en la DB, usamos el motor con umbral muy permisivo (0.20) sobre
sus propios videos de registro para extraer sus embeddings reales con alta confianza:

```python
# Ground truth de persona P = frames de P donde el motor la reconoce con sim > 0.20
# (Prácticamente todos los frames de su propio video)
positivos[P] = [frames donde sim(frame, centroide_P) > 0.20]
```

**Estrategia B — Negativos Genuinos:**
Para obtener impostores reales, usamos el esquema **Leave-One-Out**:
- Crear una DB temporal SIN la persona P
- Pasar los frames genuinos de P a través de esta DB
- El mejor match encontrado es un "impostor candidato" — midiendo su similitud
- Si esa similitud supera el umbral, es un Falso Positivo

```python
for persona_excluida in personas_registradas:
    db_sin_P = {k: v for k, v in db.items() if k != persona_excluida}
    frames_de_P = obtener_frames_del_video(persona_excluida)
    
    for frame in frames_de_P:
        vec = arcface.embed(frame)
        mejor_sim_en_db_sin_P = max(dot(vec, centroide) for centroide in db_sin_P.values())
        # Si mejor_sim_en_db_sin_P > umbral → Falso Positivo (confundió a P con alguien más)
```

**Estrategia C — Videos del Salón (Negativos Reales):**
Los videos del salón tienen personas NO registradas. Cada detección de rostro en el salón
que no corresponda a ninguno de los 9 registrados es un negativo genuino.

### Sweep de Umbrales

```python
umbrales = np.arange(0.10, 0.75, 0.02)  # 32 valores de umbral
resultados = []

for tau in umbrales:
    tp, fp, fn, tn = 0, 0, 0, 0
    
    # Evaluar positivos (personas registradas en sus propios videos)
    for nombre, frames in positivos.items():
        for frame in frames:
            sim = calcular_mejor_similitud(frame, db)
            if sim > tau and nombre_encontrado == nombre:
                tp += 1
            elif sim > tau and nombre_encontrado != nombre:
                fp += 1  # Confusión de identidad
            else:
                fn += 1
    
    # Evaluar negativos (personas NO registradas)
    for frame in frames_no_registrados:
        sim = calcular_mejor_similitud(frame, db)
        if sim > tau:
            fp += 1  # Impostor aceptado
        else:
            tn += 1  # Impostor rechazado correctamente
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr       = recall
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    resultados.append({'tau': tau, 'TPR': tpr, 'FPR': fpr, 'F1': f1, ...})
```

### Gráficas a Generar

**Gráfica 1 — Curva ROC:**
```
Eje Y (TPR): % de alumnos reales reconocidos
Eje X (FPR): % de impostores aceptados
Anotación: punto EER y punto óptimo de Youden
```

**Gráfica 2 — Curva Precision-Recall:**
```
Eje Y: Precisión = TP / (TP + FP)
Eje X: Recall = TP / (TP + FN)
Muestra el tradeoff entre ser preciso y ser sensible
```

**Gráfica 3 — Distribución de Similitudes (Histograma Superpuesto):**
```
Histograma azul: distribución de similitudes intra-clase (personas reales)
Histograma rojo: distribución de similitudes inter-clase (impostores)
Línea vertical: umbral elegido
```
Esta gráfica es la más intuitiva para entender el sistema.

**Gráfica 4 — F1-Score vs. Umbral:**
```
Eje Y: F1-Score
Eje X: Umbral τ
Marca el umbral que maximiza F1
```

### Resultados Esperados

Basado en las similitudes observadas en el CSV histórico y en los experimentos previos:

| Umbral | TPR (esperado) | FPR (esperado) | F1 (esperado) |
|---|---|---|---|
| 0.35 (actual) | ~88 % | ~8 % | ~0.82 |
| 0.42 (propuesto) | ~80 % | ~2 % | ~0.88 |
| 0.50 (conservador) | ~65 % | ~0.5 % | ~0.78 |
| **óptimo (a determinar)** | **~85 %** | **~1-2 %** | **~0.91** |

> Estos son estimados preliminares. Los valores exactos se determinarán al ejecutar el experimento.

### Script del experimento
**Archivo:** `experiments/fase3_curva_roc.py` ← *a crear en esta fase*

### Artefactos que genera esta fase
```
resultados_experimentales/fase3_curva_roc.png
resultados_experimentales/fase3_curva_pr.png
resultados_experimentales/fase3_histograma_similitudes.png
resultados_experimentales/fase3_f1_vs_umbral.png
resultados_experimentales/fase3_tabla_umbrales.md
docs/analisis/03_resultados_curva_roc.md
```

---

## FASE 4 — Mejoras al Código del Motor

### Estado: ⏳ Pendiente (requiere Fase 3 completada)

### Descripción
Con el umbral óptimo determinado empíricamente, implementamos mejoras al motor de inferencia
`asistencia_unificada_optimizada.py`. Cada mejora tiene una justificación teórica y será
evaluada individualmente para verificar que ayuda.

### Mejora 4A — Umbral Óptimo (del resultado de Fase 3)

**Qué cambia:** El valor hardcoded `self.umbral_similitud = 0.35` → valor determinado en Fase 3

**Por qué:** Un umbral sin base empírica es una suposición. La Fase 3 nos da el valor real.

**Código actual:**
```python
self.umbral_similitud = 0.35  # ← sin justificación empírica
```

**Código propuesto:**
```python
self.umbral_similitud = UMBRAL_OPTIMO  # ← del experimento Fase 3, ej. 0.42
```

### Mejora 4B — Verificación de Margen Top-2 (Anti-Ambigüedad)

**Qué cambia:** En lugar de aceptar la mejor similitud directamente, verificar que la segunda
mejor similitud está suficientemente lejos.

**Por qué:** Si `angel` y `lino` tienen similitud mutua de 0.256, y un impostor da 0.38
con `angel` y 0.35 con `lino`, el margen de 0.03 indica ambigüedad. Sin la verificación
de margen, el sistema aceptaría al impostor como `angel`.

**Código actual:**
```python
dist, idx = self.indice_faiss.search(vec, 1)  # Solo Top-1
max_sim = dist[0][0]
if max_sim > self.umbral_similitud:
    candidato = self.nombres_lista[idx[0][0]]
```

**Código propuesto:**
```python
dist, idx = self.indice_faiss.search(vec, 2)  # Top-2
mejor_sim  = dist[0][0]
segunda_sim = dist[0][1]
margen = mejor_sim - segunda_sim

MARGEN_MIN = 0.05  # Parámetro a calibrar experimentalmente

if mejor_sim > self.umbral_similitud and margen > MARGEN_MIN:
    candidato = self.nombres_lista[idx[0][0]]
else:
    candidato = "Desconocido"  # Ambiguo o no reconocido
```

**Experimento 4B — Calibración del margen mínimo:**
Evaluar el margen real entre Top-1 y Top-2 en reconocimientos correctos vs. incorrectos
para determinar el valor óptimo de `MARGEN_MIN`.

**Resultado esperado:** Con `MARGEN_MIN ≈ 0.05-0.08`, se eliminan los casos ambiguos
sin sacrificar más de 5% en TPR.

### Mejora 4C — Logging Detallado por Frame

**Qué cambia:** El motor guarda un CSV con el detalle de cada decisión tomada.

**Por qué:** Actualmente el motor es una "caja negra" durante la inferencia. Para analizar
errores y optimizar parámetros necesitamos saber qué similitud se obtuvo en cada frame,
qué identidad propuso el sistema, si pasó el liveness, etc.

**Formato del CSV de logging:**
```csv
frame_idx, tracker_id, persona_propuesta, similitud, segunda_similitud, margen, liveness_score, decision_final
145, 3, angel, 0.4521, 0.2134, 0.2387, 0.95, RECONOCIDO
146, 3, angel, 0.4489, 0.2201, 0.2288, 0.94, RECONOCIDO
147, 5, lino, 0.3821, 0.3698, 0.0123, 0.91, RECHAZADO_MARGEN
148, 5, lino, 0.6134, 0.2411, 0.3723, 0.96, RECONOCIDO
```

**Resultado esperado:** Con este log podremos identificar patrones de fallo, ver en qué
frames el sistema duda, y afinar los parámetros sin tener que re-ejecutar todo el video.

**Script a crear:** `scripts/asistencia_con_logging.py`

### Mejora 4D — Corrección del import en test_asistencia_salon.py

**Qué cambia:** Una línea de código.

**Por qué:** El script de prueba usa la versión NO optimizada del motor, lo que significa
que las pruebas no reflejan el rendimiento real del sistema en producción.

**Código actual:**
```python
from scripts.asistencia_unificada import MotorAsistencia  # ← versión antigua
```

**Código propuesto:**
```python
from scripts.asistencia_unificada_optimizada import MotorAsistencia  # ← versión correcta
```

### Artefactos que genera esta fase
```
scripts/asistencia_unificada_optimizada.py    ← modificado (4A, 4B, 4C)
scripts/asistencia_con_logging.py             ← nuevo
scripts/test_asistencia_salon.py              ← import corregido (4D)
docs/analisis/04_resultados_mejoras_motor.md  ← comparación de métricas antes/después
```

---

## FASE 5 — Evaluación Cuantitativa Multi-Video

### Estado: ⏳ Pendiente (requiere Fase 4 completada)

### Descripción
La evaluación más completa del proyecto: correr el motor mejorado sobre los **7 videos del
salón** y medir el rendimiento real persona por persona.

### Protocolo de Evaluación

**Paso 1 — Descubrir Ground Truth automáticamente:**
Para cada video del salón y cada persona registrada, usamos el motor de referencia con umbral
muy permisivo (0.20) para detectar en qué frames y en qué coordenadas aparece cada persona.
Estas coordenadas son el "Ground Truth" (la verdad de referencia).

```python
ground_truth = {}  # {video_path: {persona: [frame_idx con bbox]}}

for video in videos_salon:
    for persona in personas_registradas:
        apariciones = detectar_persona_en_video(video, persona, umbral=0.20)
        ground_truth[video][persona] = apariciones
```

**Paso 2 — Evaluar el motor mejorado en las ubicaciones del Ground Truth:**
En cada frame del Ground Truth, verificar si el motor mejorado (con el umbral óptimo de
Fase 3 y las mejoras de Fase 4) reconoce correctamente a la persona.

```python
for video, gt_personas in ground_truth.items():
    for persona, frames_gt in gt_personas.items():
        for frame_idx, bbox_ref in frames_gt.items():
            resultado = motor_mejorado.evaluar_frame(video, frame_idx, bbox_ref)
            if resultado.nombre == persona:
                tp += 1
            elif resultado.nombre == "Desconocido":
                fn += 1
            else:
                fp_confusion += 1  # Confundió con otra persona
```

**Paso 3 — Evaluar falsos positivos en frames de "desconocidos":**
Para cada video, también evaluar frames donde NO debería haber reconocimiento (personas no
registradas) y verificar que el sistema las rechace.

### Métricas por Video y Global

Para cada uno de los 7 videos:

| Métrica | Fórmula | Qué mide |
|---|---|---|
| Precisión | TP / (TP + FP) | ¿Cuándo dice que reconoció, acertó? |
| Recall (Sensibilidad) | TP / (TP + FN) | ¿De los que debía reconocer, a cuántos reconoció? |
| Especificidad | TN / (TN + FP) | ¿De los desconocidos, a cuántos rechazó? |
| F1-Score | 2·P·R / (P+R) | Balance entre Precisión y Recall |
| FPR | FP / (FP + TN) | % de impostores aceptados |
| Tiempo confirmación | frames × (1/FPS) | Cuánto tarda en confirmar asistencia |

### Análisis Comparativo: Baseline vs. Pipeline Mejorado

La tabla final compara el sistema **antes** (umbral 0.35, sin margen Top-2) con el sistema
**después** (umbral óptimo, con margen Top-2) en todos los videos:

| Video | Baseline F1 | Mejorado F1 | Δ Mejora |
|---|---|---|---|
| 20260806_105629.mp4 | ? | ? | ? |
| 20260806_105645.mp4 | ? | ? | ? |
| 20260806_105847.mp4 | ? | ? | ? |
| 20260806_105921.mp4 | ? | ? | ? |
| 20260820_140625.mp4 | ? | ? | ? |
| 20260820_140734.mp4 | ? | ? | ? |
| 20260820_140914.mp4 | ? | ? | ? |
| **GLOBAL** | **?** | **?** | **?** |

*(Los valores se llenarán al ejecutar el experimento)*

### Gráficas a Generar

**Gráfica 1 — Barras de Precisión por Persona (Baseline vs. Mejorado):**
Barras agrupadas mostrando la precisión de reconocimiento de cada uno de los 9 estudiantes
antes y después de las mejoras.

**Gráfica 2 — Heatmap de Reconocimientos por Video:**
Mapa de calor donde el eje X son los 7 videos y el eje Y son las 9 personas.
- Verde = reconocido correctamente en ese video
- Rojo = no reconocido (FN)
- Naranja = confundido con otro (error de identidad)

**Gráfica 3 — Timeline de Reconocimientos en el Video Principal:**
Para el video más largo (`20260806_105645.mp4`), un gráfico de línea de tiempo mostrando
en qué frame se confirmó la asistencia de cada persona.

### Script del experimento
**Archivo:** `experiments/fase5_evaluacion_completa.py` ← *a crear en esta fase*

### Artefactos que genera esta fase
```
resultados_experimentales/fase5_barras_precision.png
resultados_experimentales/fase5_heatmap_reconocimientos.png
resultados_experimentales/fase5_timeline_asistencia.png
resultados_experimentales/fase5_tabla_comparativa.md
docs/analisis/05_resultados_evaluacion_completa.md
```

---

## FASE 6 — Documentación y Reporte Final

### Estado: ⏳ Pendiente (requiere Fases 1-5 completadas)

### Descripción
Consolidar todos los resultados en un documento técnico completo que pueda ser presentado
como reporte académico/profesional del proyecto.

### Estructura del Reporte Final

**Archivo:** `docs/analisis/06_reporte_final_pipeline.md`

```
1. Resumen ejecutivo (1 página)
   - ¿Qué se construyó?
   - ¿Qué métricas se lograron?
   - ¿Qué limitaciones tiene?

2. Descripción del sistema
   - Arquitectura del pipeline (diagrama Mermaid)
   - Modelos utilizados (RetinaFace, ArcFace, ByteTrack, FAISS)
   - Parámetros finales calibrados

3. Descripción del dataset
   - 9 personas, X frames totales
   - Condiciones de captura (salón de clases)
   - Distribución de poses y condiciones de iluminación

4. Metodología de evaluación
   - Protocolo de Ground Truth automático
   - Definición de TP/FP/FN/TN
   - Curva ROC y selección de umbral

5. Resultados por fase
   - Fase 1: Diagnóstico (con tablas)
   - Fase 2: Mejora del dataset (comparación antes/después)
   - Fase 3: Curva ROC (con gráficas)
   - Fase 4: Mejoras al código (con comparación de métricas)
   - Fase 5: Evaluación final (tabla comparativa + heatmap)

6. Análisis de errores
   - ¿En qué casos falla el sistema?
   - ¿Qué factores afectan más el rendimiento? (iluminación, ángulo, distancia)
   - Ejemplos visuales de TP, FP, FN

7. Conclusiones y trabajo futuro
   - Hallazgos más importantes
   - Qué se haría diferente con más tiempo
   - Posibles mejoras (galería, PCA whitening, face augmentation)
```

### Artefactos que genera esta fase
```
docs/analisis/06_reporte_final_pipeline.md    ← Reporte técnico completo
docs/analisis/README.md                       ← Índice de todos los documentos
```

---

## Índice de Todos los Artefactos del Proyecto

### Documentos en `docs/analisis/`

| Archivo | Fase | Estado |
|---|---|---|
| `00_guia_metodologica.md` | — | ✅ Creado |
| `01_diagnostico_inicial.md` | 1 | ✅ Completado |
| `02_plan_experimentos.md` | — | ✅ Este archivo |
| `03_resultados_reregistro.md` | 2 | ⏳ Pendiente |
| `04_resultados_curva_roc.md` | 3 | ⏳ Pendiente |
| `05_resultados_mejoras_motor.md` | 4 | ⏳ Pendiente |
| `06_resultados_evaluacion_completa.md` | 5 | ⏳ Pendiente |
| `07_reporte_final_pipeline.md` | 6 | ⏳ Pendiente |

### Scripts en `experiments/`

| Archivo | Fase | Estado |
|---|---|---|
| `analisis_sensibilidad_muestras.py` | Pre | ✅ Existe |
| `analisis_sensibilidad_ronald.py` | Pre | ✅ Existe |
| `analisis_sensibilidad_salon.py` | Pre | ✅ Existe |
| `fase3_curva_roc.py` | 3 | ⏳ A crear |
| `fase5_evaluacion_completa.py` | 5 | ⏳ A crear |

### Gráficas en `resultados_experimentales/`

| Archivo | Fase | Estado |
|---|---|---|
| `grafica_precision_muestras.png` | Pre | ✅ Existe |
| `grafica_precision_ronald.png` | Pre | ✅ Existe |
| `grafica_precision_salon.png` | Pre | ✅ Existe |
| `fase3_curva_roc.png` | 3 | ⏳ Pendiente |
| `fase3_histograma_similitudes.png` | 3 | ⏳ Pendiente |
| `fase3_curva_pr.png` | 3 | ⏳ Pendiente |
| `fase3_f1_vs_umbral.png` | 3 | ⏳ Pendiente |
| `fase5_barras_precision.png` | 5 | ⏳ Pendiente |
| `fase5_heatmap_reconocimientos.png` | 5 | ⏳ Pendiente |
| `fase5_timeline_asistencia.png` | 5 | ⏳ Pendiente |

---

## Tabla de Parámetros Actuales vs. Esperados Finales

| Parámetro | Valor Actual | Valor Esperado (post-Fase 4) | Justificación |
|---|---|---|---|
| `umbral_similitud` | 0.35 | ~0.42-0.50 (a determinar) | Curva ROC Fase 3 |
| `umbral_liveness` | 0.90 | 0.90 (sin cambio) | Ya es robusto |
| `tamano_minimo_rostro` | 40 px | 40 px (sin cambio) | Adecuado para el salón |
| `votos_requeridos` | 3 | 3 (sin cambio) | Buen balance velocidad/seguridad |
| `margen_top2` | No existe | ~0.05-0.08 (a calibrar) | Anti-ambigüedad Fase 4B |
| `MAX_FOTOS` registro | 50 | 50 (sin cambio) | Suficiente |
| `UMBRAL_CALIDAD_BLUR` | 60 | 60 (sin cambio) | Adecuado |
| `UMBRAL_POSE_DIVERSIDAD` | 0.95 | 0.95 (sin cambio) | Funciona bien |

---

*Documento actualizado con el progreso de cada fase. Fecha de inicio: 2026-09-01.*

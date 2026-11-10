# 07 — Reporte final: dataset UNC

**Proyecto:** CV College — reconocimiento facial para asistencia
**Dataset:** `data/unc/` (primera data real de la universidad)
**Fecha del reporte:** 2026-10-20
**Estado:** Fases 0 a 8 completadas.

---

## 1. Resumen ejecutivo

Se construyó y evaluó un sistema de reconocimiento facial para asistencia con **9 estudiantes registrados** y dos videos de salón.

**Resultados principales con la configuración final:**

| Medida | Resultado |
|---|---|
| Cobertura | 9 de 9 estudiantes confirmados |
| Tiempo hasta confirmar a una persona | 0.06 a 0.26 s desde que aparece |
| Personas confirmadas dentro de ±0.5 s (±15 frames) | **76 de 80 (95 %)** |
| Errores de identidad | **0** |
| Desconocidos confirmados por error | **1 de 157 (0.6 %)** |
| Umbral de decisión | 0.30 (calibrado, no heurístico) |
| Configuración | RetinaFace + ArcFace (512 d) + centroide por estudiante + FAISS + ByteTrack |

**Conclusión general:** el sistema **no confunde identidades** y cubre a todos los estudiantes. Su punto débil es el **recall en condiciones difíciles**: caras de perfil, caras muy pequeñas y la combinación de gorra y lentes.

---

## 2. Dataset

### 2.1 Videos de registro (`data/unc/stds/`)

Un video por estudiante, vertical, 1080×1920, entre 100 y 229 frames.

| Persona | Video | Accesorios verificados |
|---|---|---|
| alexa | `VID_20260907_103652.mp4` | Sin lentes en el frame revisado |
| bob | `VID_20260907_103303.mp4` | Sin gorra |
| carlos | `VID_20260907_103851.mp4` | Pendiente de verificar |
| dani | `VID_20260907_103138.mp4` | Pendiente de verificar |
| fran | `VID_20260907_103618.mp4` | — |
| jesus | `VID_20260907_103208.mp4` | Sin lentes ni gorra |
| jose | `VID_20260907_103228.mp4` | — |
| kate | `VID_20260907_103318.mp4` | — |
| kevin | `VID_20260907_103913.mp4` | — |

### 2.2 Videos de salón (`data/unc/salon/`)

| Video | Duración | Uso |
|---|---|---|
| `VID_20260907_102851.mp4` | 106.8 s (3 203 frames) | **Principal y etiquetado.** Estudiantes sentados, salen, regresan y miran a cámara |
| `VID_20260907_105622.mp4` | 50.6 s (1 519 frames) | Generalización, sin etiquetas. Estudiantes interactuando |
| `video5096333890499381162.mp4` (11 partes `lv_0_2026*.mp4`) | 124.0 s (3 710 frames), 128.9 s en las partes | **Revalidación (Fase 8).** Aula distinta, 8 registrados presentes y estudiantes no registrados |

### 2.3 Muestra etiquetada del salón

- 1 frame por segundo → **1 127 recortes**.
- Muestra aleatoria de **300** (semilla 42), etiquetada y **revisada por una segunda persona**.
- Distribución: 160 desconocidos, 44 dudosos, 13 `no_cara` y 83 con nombre antes de las correcciones; tras las revisiones quedaron **86 con nombre** y **157 desconocidos** evaluables.
- Tamaño de cara: mediana **37 px**; 58 % de los recortes por debajo de 40 px.

---

## 3. Pipeline y configuración final

### 3.1 Registro

1. Detección con RetinaFace; se acepta solo el frame con **una sola cara** (mayor caja).
2. Alineación por ojos (224×224).
3. Filtro de nitidez (Laplaciano) y brillo.
4. Embedding ArcFace de 512 dimensiones, con CLAHE.
5. Filtro de poses repetidas (similitud > 0.95 con las ya guardadas).
6. Centroide normalizado de las hasta 50 fotos válidas (mínimo 15).

### 3.2 Inferencia

| Parámetro | Valor final | Origen |
|---|---|---|
| Detector | RetinaFace | — |
| Reconocedor | ArcFace, 512 d | — |
| Búsqueda | FAISS `IndexFlatIP` | — |
| Tracking | ByteTrack | — |
| Umbral de similitud | **0.30** | Fase 4 |
| Votos consecutivos | 3 (ventana de 5 frames) | Diseño |
| Tamaño mínimo de cara | **20 px** | Pruebas de filtro |
| Confianza mínima de detección | **0.75** | Falsos positivos (§24) |
| Anti-spoofing | omitido bajo **60 px** | §21 |
| Revalidación del ancla | cada **30 frames** | §27 |

---

## 4. Resultados por fase

### Fase 0 — Muestra y etiquetas
300 recortes etiquetados y revisados por dos personas. Ver `00_explicacion_sencilla.md` y sección 2.3.

### Fase 1 — Diagnóstico del dataset
Ver `01_diagnostico_dataset_unc.md`.
- Registro: detección 0.98–1.00, útil 0.93–1.00. Las caras miden 450 a 680 px.
- Salón: mediana **13 caras por frame**; ancho mediano **37 px**; 58 % bajo 40 px.
- Las predicciones de nitidez del plan original no se cumplieron; se reemplazaron por mediciones reales.

### Fase 2 — Registro
Ver `02_resultados_registro.md`.
- 9 centroides con norma 1.0 y 50 fotos cada uno.
- Similitud máxima entre personas: **0.297** (`alexa`–`dani`), seguida de 0.295 (`fran`–`jesus`). Ambas por debajo de la alerta de 0.30.
- Dispersión interna: medias de 0.644 (`jesus`) a 0.807 (`kate`). `jesus` tenía 4 fotos atípicas (perfil); se midió el efecto (cos 0.9965) y se decidió no modificarlas.

### Fase 3 — Robustez a accesorios
Ver `03_resultados_robustez_accesorios.md`.
- **Lentes: sin efecto** (mediana 0.496 con lentes contra 0.474 sin accesorios; Mann-Whitney p = 0.512).
- **Gorra sola: sin efecto** en los 3 casos disponibles (`fran`).
- **Gorra + lentes: sí hay efecto.** Los 4 recortes de `jesus` bajan a mediana **0.315**. Con umbral 0.30 pasan 2 de 4; con 0.35 pasaba 1 de 4.

### Fase 4 — Umbral
Ver `04_resultados_curva_roc.md`.
- **AUC 0.962**; AP 0.953; EER 0.080; d' 2.885.
- Mejor F1 en 0.28, pero con umbral 0.30 y la regla de 3 votos: **precisión 1.000, recall 0.849, F1 0.918, 0 falsos positivos** sobre la muestra.
- Se fijó **0.30** en el motor.

### Fase 5 — Mejoras al motor
Aplicadas y verificadas:
- Umbral calibrado 0.30.
- Anti-spoofing omitido en caras menores de 60 px: las alarmas de fraude bajaron de **39 a 3** en el video completo.

> Nota (A6): el "39 → 3" es de la configuración de la Fase 5. Las 8 alarmas de
> `experiments/unc/metricas/salida/asistencia/spoofing.csv` y las 12 filas del
> video 3 son de otras corridas; no se comparan entre sí.
- Filtro de confianza de detección 0.75: eliminó los falsos positivos sobre brazos y ropa.
- Revalidación del ancla cada 30 frames (+2.4 % de cómputo).
- Margen top-2 registrado en cada decisión (sin usarse todavía para decidir).
- Corrección del anti-spoofing de rostros pequeños y del conteo de votos (de acumulados a consecutivos) se hizo antes del registro.

### Fase 6 — Evaluación final
Ver `06_resultados_evaluacion_salon.md`.

| Medida | Resultado |
|---|---|
| Cobertura | 9 de 9 |
| Tiempo de confirmación | 0.06–0.26 s |
| Recorte (instantáneo) | 63 de 86 (73.3 %) |
| Recorte (ventana ±0.5 s) | **76 de 80 (95 %)** |
| Errores de identidad | 0 |
| Falsos positivos | 1 de 157 (0.6 %) |

Video 2 (sin etiquetas): 6 estudiantes confirmados; `dani` y `carlos` revisados visualmente sin errores.

### Fase 7 — Este reporte

### Fase 8 — Revalidación en el video 3 (aula distinta)
Ver `08_resultados_video3.md`.

| Medida | Resultado |
|---|---|
| Frames reales (11 partes) | 3 808 |
| Cobertura de registrados presentes | **8 de 8** |
| Confirmaciones del ausente (`jose`) | 0 |
| Errores de identidad | **1** de 78 pistas (1.3 %) |
| Pistas no registradas (ancho ≥ 40 px) | 74 |
| Estudiantes no registrados en el aula | 21 (más 8 registrados presentes y 1 ausente) |
| FP con 1 voto / con 3 votos a 0.30 | 4 de 74 / **1 de 74** |
| FP por persona no registrada | **1 de 21 (4.8 %)** |
| Re-confirmación tras cada corte | mediana 1.31 s, máx 13.2 s |

El falso positivo fue una estudiante no registrada confirmada como `dani` durante 30
frames (sim 0.336); no alteró la asistencia porque `dani` estaba confirmada en las 11
partes por otras pistas. El umbral 0.30 no se movió: subirlo a 0.34 costaría 4 pistas
correctas de `fran`, `jesus` y `kate`.

---

## 5. Análisis de errores

### 5.1 Pose (la causa principal)
Con caras de frente el fallo es **4.1 %**; de perfil, **42.9 %**. La pose de cada recorte se estimó con la desviación de la nariz respecto al centro de los ojos (landmarks de RetinaFace).

### 5.2 Tamaño
Las caras de 60 px o más casi no fallan (0 a 3.6 %). Por debajo de 60 px la tasa sube a 18–36 %. El orden no es perfectamente monótono por el tamaño muestral.

### 5.3 Accesorios
Ver 5.3 en `03_resultados_robustez_accesorios.md`: lentes sin efecto, gorra sola sin efecto, gorra y lentes juntos con efecto claro en `jesus`.

### 5.4 Falsos positivos del detector
RetinaFace detectó "caras" en un **brazo** (confianza 0.666 a 0.735) y en **ropa oscura** (0.549). El filtro de confianza 0.75 los eliminó. Este fue también el origen visual de las "rayas de la camiseta detectadas como rostro" que se observaron en la demo.

### 5.5 Etiqueta anclada sobre un recuadro desplazado
Cuando una pista ya está anclada, el nombre se muestra aunque el recuadro derive. Se mitigó revalidando cada 30 frames y con el filtro de confianza. No afecta la asistencia (se registra una sola vez), solo la visualización.

### 5.6 Anti-spoofing inestable en caras pequeñas
Para una misma cara, la confianza del modelo cambió de 0.805 a 0.970 solo por redondear la caja a enteros. Con el umbral de 0.90 eso producía falsas alarmas de fraude. Se omitió el anti-spoofing bajo 60 px (riesgo aceptado y documentado).

---

## 6. Limitaciones

1. **Un solo video etiquetado y una sola fecha.** El umbral debe revalidarse con más aulas, más iluminación y otras cámaras.
2. **Cámaras de registro con accesorios no controlados.** Solo se verificaron algunos videos; `carlos` y `dani` quedaron pendientes.
3. **Gorra + lentes con evidencia limitada:** 4 recortes de una sola persona (`jesus`).
4. **Desconocidos sin pista:** 22 de 157 recortes fueron filtrados por confianza, así que el FPR se mide sobre 135 casos.
5. **El video 2 no tiene etiquetas**: su resultado es cualitativo.
6. **La asistencia real no se probó en vivo con cámara**: todas las pruebas son sobre video pregrabado.
7. **El anti-spoofing no se probó con ataques reales** (foto o pantalla).

---

## 7. Trabajo futuro

1. **Revalidar el umbral 0.30** con el tercer video y con más sesiones.
2. **Registrar con accesorios** (gorra y lentes) si la política de un video por estudiante cambia; el caso `jesus` lo justifica.
3. **Evaluar la regla de margen top-2** como criterio de rechazo (hoy solo se registra).
4. **Probar el anti-spoofing** con fotos y pantallas, y calibrar su umbral por tamaño de cara.
5. **Asistencia en vivo** con cámara real, para medir falsos positivos por personas que pasan frente al aula.
6. **Medir el costo de cómputo por aula** (número de caras simultáneas, FPS sostenido).

---

## 8. Artefactos

### Documentos
- `00_explicacion_sencilla.md`: explicación en lenguaje sencillo.
- `00_plan_experimentos_unc.md`: plan con estado de cada fase.
- `01_diagnostico_dataset_unc.md`
- `02_resultados_registro.md`
- `03_resultados_robustez_accesorios.md`
- `04_resultados_curva_roc.md`
- `06_resultados_evaluacion_salon.md`
- `08_resultados_video3.md`: revalidación en el video 3 (11 partes)
- `03_bitacora_calculos.md`: cada cálculo con fórmula, librería y ejemplo.
- `07_reporte_final_unc.md`: este documento.

### Scripts
| Archivo | Función |
|---|---|
| `experiments/unc/fase0_muestreo_salon.py` | Muestreo de caras del salón |
| `experiments/unc/fase0_muestra_anotacion.py` | Muestra aleatoria y cuadrículas |
| `experiments/unc/fase0_etiquetas_visuales.py` | Etiquetas de primera pasada |
| `experiments/unc/fase0_emparejar_iou.py` | Emparejamiento recorte–detección |
| `experiments/unc/fase0_prueba_estres_iou.py` | Prueba de estrés del emparejamiento |
| `experiments/unc/fase0_reconocimiento_salon.py` | Reconocimiento sobre la muestra |
| `experiments/unc/fase0_causas_fallos.py` | Pose, tamaño y nitidez por resultado |
| `experiments/unc/fase0_lista_accesorios.py` | Lista y cuadrículas de accesorios |
| `experiments/unc/fase0_revision_fraudes.py` | Revisión de alarmas de fraude |
| `experiments/unc/fase0_revision_pistas.py` | Pistas confirmadas por persona |
| `experiments/unc/fase1_diagnostico_dataset_unc.py` | Diagnóstico por video |
| `experiments/unc/fase2_dispersion.py` | Dispersión interna |
| `experiments/unc/fase4_similitudes.py` | Similitud contra los 9 centroides |
| `experiments/unc/fase4_curva_roc.py` | ROC, AUC y métricas por umbral |
| `experiments/unc/fase4_multiframe.py` | 1 frame vs 3 frames |
| `experiments/unc/metricas/metricas.py` | Métricas consolidadas (ROC, PR, AUC, AP, EER, d', matrices, asistencia, rendimiento) |
| `experiments/unc/fase6_evaluacion_salon.py` | Evaluación final del sistema |
| `experiments/unc/demo_video_salon.py` | Corridas con video anotado y capturas |
| `experiments/unc/fase7_video3_partes.py` | Corrida sobre las 11 partes del video 3 |
| `experiments/unc/fase7_revision.py` | Hojas de revisión de identidad por pista |
| `experiments/unc/fase7_pistas.py` | Confirmaciones débiles y pistas no registradas |
| `experiments/unc/fase7_clusters.py` | Conteo de personas distintas entre las no registradas |

### Corridas principales
- `experiments/unc/demo/prueba_2010_0935_final/`: evaluación final (video 1).
- `experiments/unc/demo/prueba_1910_1602_video2/`: generalización (video 2).
- `experiments/unc/demo/prueba_2010_1600_video3/`: revalidación (video 3, 11 partes).

### Motor
- `scripts/asistencia_unificada_optimizada.py` (configuración final).
- `scripts/registro_video.py` y `scripts/run_single_video_reg.py`.
- `core/utils_facial.py` (preprocesado compartido).

---

## 9. Reproducción

```bash
# Registro (un video por estudiante)
venv/bin/python scripts/run_single_video_reg.py <nombre> data/unc/stds/<nombre>/<video>.mp4 --sin-ventana

# Evaluación completa del video 1
venv/bin/python experiments/unc/demo_video_salon.py 0 107 --filtro=20 --carpeta=prueba_final
venv/bin/python experiments/unc/fase6_evaluacion_salon.py

# Curva ROC
venv/bin/python experiments/unc/fase4_similitudes.py
venv/bin/python experiments/unc/fase4_curva_roc.py
venv/bin/python experiments/unc/fase4_multiframe.py
```

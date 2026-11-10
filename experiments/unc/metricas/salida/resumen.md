# Métricas consolidadas del pipeline

**Generado por:** `experiments/unc/metricas/metricas.py`  
**Semilla de bootstrap:** `20261105` · **Remuestreos:** 2 000  
**Datos:** CSV de las fases 0, 4, 6 y 7 en `experiments/unc/`

> Este informe se regenera completo con una sola ejecución. Cada número
> tiene su CSV fuente en `salida/`; no se edita a mano.

---

## 1. Qué métricas aplican y cuáles no

El sistema **no es un modelo de regresión**: no predice un valor continuo.
Produce una **clase** (identidad de 9 candidatos, `Desconocido`, `INTRUSO`)
a partir de una similitud coseno. Por eso:

| Métrica | ¿Aplica? | Motivo |
|---|---|---|
| MAE | No | Requiere un valor continuo predicho y uno real. |
| MSE / RMSE | No | Igual que MAE; además penaliza al cuadrado, sin sentido aquí. |
| MAPE | No | Requiere valores positivos y un cociente porcentual. |
| R² | No | No hay varianza explicada de una variable dependiente. |
| Accuracy, Precision, Recall, F1 | **Sí** | Clasificación / verificación. |
| ROC-AUC, PR-AUC, EER, d' | **Sí** | Verificación biométrica con score. |
| Matriz de confusión, MCC, kappa | **Sí** | Matriz 2×2 de identificación. |
| IoU, tasas de detección | **Sí** | Detección de rostros. |
| FPS, latencia, CPU/RAM | **Sí** | Rendimiento del pipeline. |

---

## 2. Identidad (nivel recorte)

**Protocolo.** 300 recortes etiquetados del video 1. Se excluyen 44
`dudoso` y 13 `no_cara`; quedan **86 positivos** (con nombre) y **157
desconocidos**. Para la verificación score-level, cada positivo aporta
además un negativo (la mayor similitud con otra persona): **243 negativos**.

| Métrica | Valor |
|---|---|
| AUC (verificación) | **0.9616** |
| Average Precision (AP) | **0.9535** |
| EER | **0.0798** (umbral 0.239) |
| d' (discriminabilidad) | **2.885** |
| Youden | umbral **0.281** (TPR 0.919, FPR 0.021) |
| Mejor F1 operativo | umbral **0.280** (F1 0.941) |

**Lectura.** El AUC de 0.9616 significa que, en el 96.2 % de los pares
(genuino, impostor), el genuino obtiene mayor similitud. El EER de 8.0 %
es el punto donde FPR y FNR se igualan. La d' de 2.885 confirma una
separación fuerte. La cifra antigua de 0.7439 era un artefacto de una
rejilla de umbrales truncada (0.10–0.705); la ROC exacta está en
`salida/identidad/roc_puntos.csv`.

### 2.1 Umbral operativo 0.30

| Umbral | TP | FP | FN | TN | Precisión | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 0.250 | 79 | 8 | 7 | 149 | 0.908 | 0.919 | 0.913 | 0.938 |
| 0.280 | 79 | 3 | 7 | 154 | 0.963 | 0.919 | 0.941 | 0.959 |
| 0.300 | 75 | 1 | 11 | 156 | 0.987 | 0.872 | 0.926 | 0.951 |
| 0.350 | 71 | 0 | 15 | 157 | 1.000 | 0.826 | 0.904 | 0.938 |
| 0.400 | 61 | 0 | 25 | 157 | 1.000 | 0.709 | 0.830 | 0.897 |
| 0.450 | 50 | 0 | 36 | 157 | 1.000 | 0.581 | 0.735 | 0.852 |
| 0.500 | 38 | 0 | 48 | 157 | 1.000 | 0.442 | 0.613 | 0.802 |

> **Dos protocolos distintos, no comparables.** El recall de 0.872 de
> esta tabla es la simulación a **un frame** sobre 86 recortes. La
> cobertura de asistencia del video 1 (recall 0.733) es la corrida
> **real del motor con 3 votos y ancla**: confirma 63/86 recortes, pero
> la cobertura de personas es 9/9 porque cada estudiante se confirma en
> algún frame. No mezclar ambas cifras en la misma comparación.

**Recomendación.** El umbral 0.30 mantiene precisión 0.987 y recall 0.872
con **1 falso positivo en 157 desconocidos**. Youden cae en 0.281 (F1
0.9405), pero 0.30 prioriza la precisión, que es lo correcto cuando un
falso positivo asigna una asistencia a la persona equivocada.

### 2.2 Métricas por estudiante (umbral 0.30)

| Estudiante | Support | TP | FP | FN | Precisión | Recall | F1 | Falsos desconocidos |
|---|---|---|---|---|---|---|---|---|
| alexa | 12 | 11 | 0 | 1 | 1.000 | 0.917 | 0.957 | 0 |
| bob | 14 | 9 | 0 | 5 | 1.000 | 0.643 | 0.783 | 0 |
| carlos | 6 | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| dani | 17 | 16 | 0 | 1 | 1.000 | 0.941 | 0.970 | 0 |
| fran | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| jesus | 4 | 2 | 0 | 2 | 1.000 | 0.500 | 0.667 | 0 |
| jose | 10 | 10 | 1 | 0 | 0.909 | 1.000 | 0.952 | 1 |
| kate | 13 | 11 | 0 | 2 | 1.000 | 0.846 | 0.917 | 0 |
| kevin | 7 | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |

**Top-k (closed-set, 86 etiquetados):** top-1 0.977, top-2 1.000, top-3 1.000, MRR 0.988. El rango 1 concentra 84/86; los 2 restantes están en rango 2.

### 2.3 Intervalos de confianza (95 %)

| Métrica | Valor | IC 95 % | Método |
|---|---|---|---|
| auc | 0.9616 | [0.9276, 0.9877] | bootstrap 2000 |
| f1_030 | 0.9259 | [0.8809, 0.9665] | bootstrap 2000 |
| accuracy_030 | 0.9506 | [0.9157, 0.9715] | Wilson |
| recall_030 | 0.8721 | [0.7853, 0.9271] | Wilson |
| precision_030 | 0.9868 | [0.9292, 0.9977] | Wilson |

### 2.4 Error por tamaño y accesorios

| Bin de ancho | Etiquetados | Correctos | Accuracy | Desconocidos | Falsos aceptados | FAR |
|---|---|---|---|---|---|---|
| <25 px | 1 | 1 | 1.000 | 12 | 0 | 0.000 |
| 25-29 px | 13 | 10 | 0.769 | 24 | 0 | 0.000 |
| 30-39 px | 34 | 29 | 0.853 | 55 | 0 | 0.000 |
| 40-59 px | 11 | 8 | 0.727 | 51 | 1 | 0.020 |
| 60-79 px | 4 | 4 | 1.000 | 15 | 0 | 0.000 |
| >=80 px | 23 | 23 | 1.000 | 0 | 0 |  |

| Grupo | n | Correctos | Accuracy | Similitud propia media |
|---|---|---|---|---|
| sin accesorios | 44 | 37 | 0.841 | 0.464 |
| lentes | 35 | 33 | 0.943 | 0.481 |
| gorra | 3 | 3 | 1.000 | 0.539 |
| gorra+lentes | 4 | 2 | 0.500 | 0.285 |

### 2.5 Multiframe (1 vs 3 frames)

| Umbral | Protocolo | TP | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.280 | 1 frame | 79 | 3 | 7 | 0.963 | 0.919 | 0.940 |
| 0.280 | 3 frames | 76 | 3 | 10 | 0.962 | 0.884 | 0.921 |
| 0.300 | 1 frame | 75 | 1 | 11 | 0.987 | 0.872 | 0.926 |
| 0.300 | 3 frames | 73 | 0 | 13 | 1.000 | 0.849 | 0.918 |
| 0.350 | 1 frame | 71 | 0 | 15 | 1.000 | 0.826 | 0.904 |
| 0.350 | 3 frames | 67 | 0 | 19 | 1.000 | 0.779 | 0.876 |

---

## 3. Detección

- **9 estudiantes, 388 frames muestreados:**
  detección micro 0.9974; cara única micro 0.9974.
- **Video del salón:** 3203 frames, 1068 muestreados; detección 0.9307, útil 0.0009.
- **Tamaño de cara:** mediana 37.0 px; 58.5 % mide menos de 40 px.
- **Emparejamiento IoU:** 300 recortes; IoU medio 1.000, mediano 1.000, 100.0 % ≥ 0.50.
  Los recortes se extrajeron de las detecciones, por eso el IoU es 1.0;
  la prueba de estrés es la que mide la robustez del emparejamiento.
- **Estrés de emparejamiento:** 100.0 % de aciertos en 6000 perturbaciones.

| Persona | Frames | Detección | Cara única | Útil |
|---|---|---|---|---|
| alexa | 105 | 1.000 | 1.000 | 1.000 |
| bob | 115 | 1.000 | 1.000 | 1.000 |
| carlos | 144 | 1.000 | 1.000 | 1.000 |
| dani | 125 | 1.000 | 1.000 | 1.000 |
| fran | 154 | 1.000 | 1.000 | 1.000 |
| jesus | 130 | 0.977 | 0.977 | 0.932 |
| jose | 116 | 1.000 | 1.000 | 1.000 |
| kate | 163 | 1.000 | 1.000 | 1.000 |
| kevin | 100 | 1.000 | 1.000 | 1.000 |

| Filtro | Etiquetados | Reconocidos | Tasa micro | Tasa macro (9) |
|---|---|---|---|---|
| 40 px | 86 | 34 | 0.395 | 0.429 |
| 20 px | 86 | 71 | 0.826 | 0.817 |
| 0 px | 86 | 71 | 0.826 | 0.817 |

| Resultado | n | Ancho mediano | Nitidez media | Pose media |
|---|---|---|---|---|
| acierto | 71 | 38.0 px | 1530.97 | 0.154 |
| falso_negativo | 15 | 34.0 px | 1982.76 | 0.302 |

---

## 4. Asistencia

| Video | Presentes | Confirmados | Cobertura | Recall recortes | Desconocidos falsos |
|---|---|---|---|---|---|
| Video 1 | 9 | 9 | 1.000 | 0.733 | 1/157 |
| Video 3 | 8 | 8 | 1.000 | — | — |

**Matriz de sesión (9 estudiantes):**

| Video | TP | FN | FP | TN | Precisión | Recall | F1 | Nota |
|---|---|---|---|---|---|---|---|---|
| video 1 | 9 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | sin ausentes: FP/TN no estimables; cobertura 9/9 |
| video 3 | 8 | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 | jose ausente; 8/8 presentes confirmados |

### 4.1 Intrusos

| Video | Reales | Filas | TP | FP | FN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| video 1 (44 s, sin desconocidos verificados) | 0 | 7 | 0 | 7 | 0 | — | — | — |
| video 3 (21 desconocidos reales) | 21 | 29 | 21 | 8 | 0 | 0.724 | 1.000 | 0.840 |

El video 1 no tiene intrusos reales verificados: las 7 filas se reportan
como cota superior de falsos (0.159/s; ver A1). En
el video 3 hay 21 desconocidos reales y 8 filas de registrados; la
precisión de fila es 0.724 y el recall 1.000. La regla está calibrada
para no perder desconocidos; el profesor descarta los falsos desde
`/sesiones/<id>/`.

**Spoofing:** 8 alarmas en 8 pistas, confianza media 0.760 (rango 0.488–0.970). No hay un conjunto etiquetado de ataques, así que no se calcula
precision/recall; solo se reportan las alarmas.

### 4.2 Presencia por parte (video 3)

| Estudiante | Partes presente | Estabilidad (11) |
|---|---|---|
| alexa | 11 | 1.000 |
| bob | 10 | 0.909 |
| carlos | 8 | 0.727 |
| dani | 11 | 1.000 |
| fran | 2 | 0.182 |
| jesus | 6 | 0.545 |
| jose | 0 | 0.000 |
| kate | 4 | 0.364 |
| kevin | 10 | 0.909 |

### 4.3 Fragmentación de pistas (video 3)

| Persona | Pistas confirmadas | Filas confirmadas | Frames/pista medio | Primera confirmación (s) |
|---|---|---|---|---|
| alexa | 13 | 1470 | 113.1 | 5.3 |
| bob | 13 | 1596 | 122.8 | 1.4 |
| carlos | 9 | 870 | 96.7 | 3.77 |
| dani | 11 | 1226 | 111.5 | 0.1 |
| fran | 3 | 87 | 29.0 | 0.5 |
| jesus | 5 | 564 | 112.8 | 0.1 |
| jose | 0 | 0 |  |  |
| kate | 4 | 558 | 139.5 | 0.1 |
| kevin | 13 | 1032 | 79.4 | 7.03 |

---

## 5. Rendimiento

| Equipo | Etapa | Valor | Unidad | Fuente |
|---|---|---|---|---|
| CPU | deteccion | 47 | ms/frame | docs/computo.md §6 |
| CPU | embedding | 24 | ms/cara | docs/computo.md §6 |
| CPU | faiss | 0.02 | ms/cara | docs/computo.md §6 |
| CPU | overhead | 3 | ms/frame | docs/computo.md §6 |
| CPU | pipeline_total | 7.52 | fps | docs/computo.md §6 (200 frames, 26.6 s) |
| CPU | pipeline_rango | 6.6-7.5 | fps | docs/computo.md §6 |
| CPU | mjpeg_entrega | 6.91 | fps | docs/streaming_mjpeg_vs_polling.md §4 |
| CPU | polling_300ms_entrega | 3.17 | fps | docs/streaming_mjpeg_vs_polling.md §4 |
| CPU | mjpeg_time_to_first_frame | 14 | ms | docs/streaming_mjpeg_vs_polling.md §4 |
| CPU | mjpeg_separacion_media | 164 | ms | docs/streaming_mjpeg_vs_polling.md §4 |
| CPU | carga_modelos_cache | 0.1 | s | docs/computo.md §4 |
| CPU | cpu_consumido | 31.3 | cpu/s | docs/streaming_mjpeg_vs_polling.md §4 |
| CPU | ram_total | 62 | GiB | docs/computo.md §1 |
| CPU | ram_libre_medicion | 2.7 | GiB | docs/computo.md §1 |
| GPU | vram_total | 24 | GiB | docs/computo.md §1 |

| Equipo | Proveedor | FPS | Estado | Nota |
|---|---|---|---|---|
| CPU (i9-14900KF) | CPUExecutionProvider | 7.52 | medido | onnxruntime CPU 1.24.4; ~7 fps en 1080p |
| GPU (RTX 4090) | CUDAExecutionProvider | — | pendiente | requiere onnxruntime-gpu y CUDA/cuDNN; no instalado en este entorno |

El pipeline corre en **CPU** (`onnxruntime` build CPU). La RTX 4090 está
presente pero sin `CUDAExecutionProvider`; el benchmark GPU queda
pendiente y se llenará en `salida/rendimiento/cpu_gpu.csv`.

---

## 6. Limitaciones y validez

- **Un solo video etiquetado** para identidad (300 recortes, 86 positivos).
  Los intervalos de confianza reflejan ese tamaño; no sustituyen una
  validación con más aulas y fechas.
- **Los recortes no incluyen fallos del detector**: solo caras detectadas.
  La tasa de detección se reporta aparte, pero no es un recall de detección
  contra ground truth exhaustivo.
- **El AUC score-level mide un frame**, sin los 3 votos ni el ancla. No es
  directamente comparable con la precisión/recall del motor en vivo;
  por eso ambos protocolos se reportan por separado.
- **Asistencia:** la matriz de sesión tiene pocos negativos (0 en video 1,
  1 en video 3); no es una estimación poblacional de precision/recall.
- **Intrusos:** las cifras son del código real y revisión manual; el
  simulador offline se quedó corto por hasta 7 filas (4 vs 7 en el video
  1 y 22 vs 29 en el video 3). No se usa para la decisión.
- **Intrusos, salvedad A1:** las "7 filas falsas" del video 1 asumen que
  ese tramo no tiene desconocidos, pero el mismo video tiene 157 recortes
  `desconocido` y una no registrada documentada en
  `09_incidente_identidad_dani.md`; la cifra es una **cota**, no un conteo
  verificado. Ver `reporte_auditoria_resultados.md`.
- **Spoofing:** no hay conjunto etiquetado de ataques; solo se reportan las
  alarmas, no precision/recall.
- **mAP de detección no calculado:** exige ground truth exhaustivo (la caja
  de todas las caras de los frames evaluados), que no existe; se reportan
  IoU, estrés de emparejamiento y tasas de detección como alternativas.
- **GPU:** no medida; el entorno solo tiene `onnxruntime` CPU.

---

## 7. Archivos

| Carpeta | Contenido |
|---|---|
| `salida/identidad/` | `scores.csv`, `roc_puntos.csv`, `pr_puntos.csv`, `tabla_umbrales.csv`, `matriz_confusion.csv`, `metricas_por_estudiante.csv`, `topk_y_rango.csv`, `open_set.csv`, `intervalos_confianza.csv`, `por_ancho.csv`, `por_accesorio.csv`, `multiframe.csv` y las 6 gráficas |
| `salida/deteccion/` | `resumen_deteccion.csv`, `iou_emparejamiento.csv`, `estres_iou.csv`, `tamanos_cara.csv`, `efecto_filtro.csv`, `fallos_por_causa.csv` |
| `salida/asistencia/` | `cobertura_video1.csv`, `cobertura_video3.csv`, `confusion_sesion.csv`, `por_persona_video1.csv`, `presencia_por_parte.csv`, `intrusiones.csv`, `barrido_intrusos.csv`, `tracking.csv`, `spoofing.csv` |
| `salida/rendimiento/` | `fps_y_latencia.csv`, `cpu_gpu.csv` |

---

## 8. Reproducibilidad

```bash
venv/bin/python experiments/unc/metricas/metricas.py
```

No se necesita scikit-learn ni pandas. Las gráficas usan matplotlib con
backend `Agg`; los CSV se escriben con `utf-8` y punto decimal.

# Métricas del pipeline — carpeta para la tesis

Esta carpeta reúne, en una sola ejecución, **todas las métricas que sí aplican**
al sistema de asistencia por reconocimiento facial, junto con sus CSV, gráficas
y un informe consolidado (`salida/resumen.md`).

No es un cuaderno de experimentos nuevo: lee los CSV que ya produjeron las
fases 0, 4, 6 y 7 en `experiments/unc/` y los normaliza para que cada número del
documento tenga un archivo fuente.

---

## 1. Cómo se ejecuta

```bash
venv/bin/python experiments/unc/metricas/metricas.py
```

Salida:

```
experiments/unc/metricas/salida/
├── identidad/     ROC, PR, AUC, AP, EER, d', umbrales, matriz de confusión,
│                  métricas por estudiante, top-k, intervalos de confianza,
│                  error por ancho y accesorios, multiframe y 6 gráficas.
├── deteccion/     tasas de detección, IoU, estrés, tamaños y filtro.
├── asistencia/    cobertura video 1 y 3, matriz de sesión, intrusos,
│                  barrido de umbrales, fragmentación de pistas, spoofing.
├── rendimiento/   FPS, latencia por etapa y CPU/GPU.
└── resumen.md     informe con todos los números y su fuente.
```

La ejecución es **determinista**: bootstrap con semilla `20261105`, 2 000
remuestreos, orden estable de scores y CSVs en UTF-8 con punto decimal.

Dependencias: `numpy` y `matplotlib` (ambas ya están en `venv/`). **No** se usa
scikit-learn, pandas ni scipy. Así el cálculo se reproduce con el mismo entorno
del motor y no se añaden paquetes a `requirements.txt`.

---

## 2. Qué se mide y qué no

El sistema **no es un modelo de regresión**. No predice un valor continuo:
produce una **clase** (una de las 9 identidades, `Desconocido` o `INTRUSO`) a
partir de una similitud coseno. Por eso:

| Métrica | ¿Aplica? | Motivo |
|---|---|---|
| MAE | **No** | Requiere un valor continuo predicho y uno real. |
| MSE / RMSE | **No** | Igual que MAE; además penaliza al cuadrado. |
| MAPE | **No** | Requiere valores positivos y un cociente porcentual. |
| R² | **No** | No hay variable dependiente ni varianza que explicar. |
| Accuracy | **Sí** | Clasificación / verificación. |
| Precision, Recall, F1 | **Sí** | Clasificación / verificación. |
| Specificity, NPV, MCC, kappa | **Sí** | Matriz 2×2 completa. |
| ROC-AUC, PR-AUC (AP) | **Sí** | Score continuo de verificación. |
| EER, d', Youden | **Sí** | Métricas biométricas estándar. |
| DIR, FAR, FRR | **Sí** | Identificación abierta. |
| IoU, tasas de detección | **Sí** | Detección de rostros. |
| mAP (detección) | **No calculado** | Requiere ground truth exhaustivo de todas las caras de los frames evaluados; no hay esa anotación. Se reportan IoU, estrés y tasas de detección. |
| FPS, latencia, CPU/RAM | **Sí** | Rendimiento del pipeline. |
| MOTA / IDF1 | **No calculadas** | Requieren `motmetrics` y ground truth de cajas por frame; no hay anotación MOT. Se reporta fragmentación de pistas como descriptivo. |

Si el documento de tesis enumera MAE/MSE/RMSE/MAPE/R², la respuesta es que **no
corresponden al problema**; incluirlas daría una apariencia de rigor sin medir
nada del sistema. La sección 1 de `salida/resumen.md` deja la justificación
escrita.

---

## 3. Definiciones

### 3.1 Clasificación binaria

Para una matriz 2×2 con TP, FP, FN y TN:

| Métrica | Fórmula | Lectura |
|---|---|---|
| Accuracy | `(TP+TN)/N` | Aciertos totales. |
| Precision | `TP/(TP+FP)` | De lo que el sistema acepta, cuánto es correcto. |
| Recall / TPR | `TP/(TP+FN)` | De lo que debería aceptar, cuánto acepta. |
| Specificity / TNR | `TN/(TN+FP)` | De lo que debería rechazar, cuánto rechaza. |
| F1 | `2PR/(P+R)` | Media armónica de precisión y recall. |
| NPV | `TN/(TN+FN)` | De lo que rechaza, cuánto era realmente negativo. |
| FPR | `FP/(FP+TN)` | Falsos positivos entre los negativos. |
| FNR | `FN/(FN+TP)` | Falsos negativos entre los positivos. |
| MCC | `(TP·TN−FP·FN)/√(…)` | Correlación 2×2; robusta al desbalance. |
| Kappa de Cohen | `(Po−Pe)/(1−Pe)` | Acuerdo corregido por azar. |
| Youden J | `TPR+TNR−1` | Punto que maximiza la suma de aciertos. |

### 3.2 Verificación biométrica

- **ROC**: TPR frente a FPR al variar el umbral.
- **AUC**: probabilidad de que un genuino aleatorio tenga mayor score que un
  impostor aleatorio. Se calcula con la **ROC exacta** (un punto por score,
  desde `+inf`), no con una rejilla truncada.
- **PR-AUC / AP**: área bajo la curva Precision-Recall. Es más informativa que
  la ROC cuando hay desbalance (aquí 86 positivos frente a 243 negativos).
- **EER**: punto donde `FPR = FNR`; se interpola sobre la ROC exacta.
- **d'**: separación de medias de genuinos e impostores, normalizada por la
  desviación combinada.
- **DIR / FAR / FRR**: identificación abierta. `DIR` es la fracción de
  etiquetados correctamente identificados; `FAR` es la fracción de desconocidos
  aceptados; `FRR` es la fracción de etiquetados rechazados.

### 3.3 Detección

- **IoU**: intersección sobre unión entre la caja del recorte y la detección.
- **Tasa de detección**: proporción de frames muestreados con al menos una
  cara.
- **Tasa de cara única**: proporción de frames con exactamente una cara.
- **Tasa útil**: proporción de frames con una cara que además supera el filtro
  de tamaño.

### 3.4 Rendimiento

- **FPS**: frames procesados por segundo.
- **Latencia por etapa**: milisegundos de detección, embedding, búsqueda FAISS
  y overhead.
- **CPU/s**: CPU consumida por segundo de pared.

---

## 4. Protocolos y advertencias

1. **Score-level vs operativo.** La ROC/AUC usa 86 positivos (similitud con la
   persona etiquetada) y 243 negativos (86 mejores similitudes con otra persona
   + 157 desconocidos). La tabla operativa usa 86 etiquetados y **solo 157
   desconocidos** como negativos, porque modela la decisión real
   `argmax + umbral`. Son dos vistas distintas y ambas se reportan.
2. **Un frame vs 3 votos.** La tabla de identidad es una simulación a **un
   frame**. El motor real exige 3 votos consecutivos y ancla; por eso el recall
   por recorte de la corrida real (0.733) no es comparable con el recall de la
   simulación (0.872). La cobertura de personas sí es 9/9 en el video 1.
3. **Los recortes no incluyen fallos del detector.** Se extrajeron de las
   detecciones; la tasa de detección se reporta aparte, pero no es un recall de
   detección contra ground truth exhaustivo.
4. **Un solo video etiquetado.** Los intervalos de confianza reflejan 86
   positivos; no sustituyen una validación con más aulas, fechas e iluminación.
5. **Asistencia.** La matriz de sesión tiene pocos negativos (0 ausentes en el
   video 1, 1 en el video 3); no es una estimación poblacional. La cobertura
   9/9 y 8/8 es un caso de estudio, no una tasa general.
6. **Intrusos.** Las cifras son del **código real** y revisión manual
   (`docs/analisis/unc/11_intrusos.md`), no del simulador offline. El video 1 no
   tiene intrusos reales; sus 7 filas son falsas. El video 3 tiene 21
   desconocidos reales y 29 filas (21 + 8 de registrados).
7. **Spoofing.** No hay un conjunto etiquetado de ataques; solo se reportan las
   alarmas y su confianza. No se calcula precision/recall.
8. **GPU.** El entorno solo tiene `onnxruntime` CPU. El benchmark GPU queda
   pendiente en `salida/rendimiento/cpu_gpu.csv`.

---

## 5. Fuentes de datos

| Archivo | Uso |
|---|---|
| `experiments/unc/fase4/fase4_similitudes.csv` | 300 recortes × 9 similitudes; identidad. |
| `experiments/unc/fase4/fase4_multiframe.csv` | 1 vs 3 frames. |
| `experiments/unc/salon_muestra/recortes.csv` | Cajas y anchos de los 300 recortes. |
| `experiments/unc/salon_muestra/muestra_anotacion.csv` | Etiqueta manual de los recortes. |
| `experiments/unc/salon_muestra/emparejamiento_iou.csv` | IoU recorte-detección. |
| `experiments/unc/salon_muestra/prueba_estres_iou.csv` | Robustez del emparejamiento. |
| `experiments/unc/salon_muestra/accesorios_muestra.csv` | Lentes / gorra por recorte. |
| `experiments/unc/salon_muestra/causas_fallos.csv` | Ancho, nitidez y pose de aciertos/FN. |
| `experiments/unc/fase1_resumen_videos.csv` | Detección por persona y salón. |
| `experiments/unc/fase1_caras_salon.csv` | Distribución de anchos por frame. |
| `experiments/unc/salon_muestra/efecto_filtro_tamano.csv` | Reconocimiento con filtro 0/20/40 px. |
| `experiments/unc/demo/prueba_2010_0935_final/fase6_por_persona.csv` | Corrida real video 1. |
| `experiments/unc/demo/prueba_2010_0935_final/decisiones.csv` | Decisiones por frame video 1. |
| `experiments/unc/demo/prueba_2110_1613_video3/resumen_personas.csv` | Confirmaciones video 3. |
| `experiments/unc/demo/prueba_2110_1613_video3/resumen_por_parte.csv` | Presencia por parte. |
| `experiments/unc/demo/prueba_2110_1613_video3/decisiones_video3.csv` | Pistas y fragmentación. |
| `experiments/unc/demo/revision_fraudes.csv` | Alarmas de spoofing. |
| `docs/computo.md` | FPS, latencia, hardware. |
| `docs/streaming_mjpeg_vs_polling.md` | MJPEG vs polling. |
| `docs/analisis/unc/11_intrusos.md` | Cifras de intrusos y barrido. |

---

## 6. Mapa de salidas

| Archivo | Contenido |
|---|---|
| `identidad/scores.csv` | AUC, AP, EER, d', Youden, mejor F1, métricas en 0.30. |
| `identidad/roc_puntos.csv` | FPR, TPR y umbral de la ROC exacta. |
| `identidad/pr_puntos.csv` | Recall y precisión de la PR exacta. |
| `identidad/tabla_umbrales.csv` | 181 umbrales con TP/FP/FN/TN, P/R/F1, accuracy, MCC, kappa. |
| `identidad/matriz_confusion.csv` | Matrices en 0.25–0.50, Youden y mejor F1. |
| `identidad/metricas_por_estudiante.csv` | One-vs-rest por las 9 personas. |
| `identidad/topk_y_rango.csv` | Top-1/2/3, MRR y distribución de rangos. |
| `identidad/open_set.csv` | DIR, FAR y FRR por umbral. |
| `identidad/intervalos_confianza.csv` | IC 95 % (bootstrap y Wilson). |
| `identidad/por_ancho.csv` | Aciertos y falsos aceptados por bin de tamaño. |
| `identidad/por_accesorio.csv` | Lentes, gorra y gorra+lentes. |
| `identidad/multiframe.csv` | 1 frame vs 3 frames. |
| `identidad/*.png` | ROC, PR, histograma, F1 vs umbral, matriz y por estudiante. |
| `deteccion/resumen_deteccion.csv` | Detección por persona, salón y agregados. |
| `deteccion/iou_emparejamiento.csv` | IoU medio/mediano y % ≥ 0.50. |
| `deteccion/estres_iou.csv` | Aciertos bajo perturbación de caja. |
| `deteccion/tamanos_cara.csv` | Percentiles y % < 20/< 40/≥ 40 px. |
| `deteccion/efecto_filtro.csv` | Tasa micro/macro con filtro 0/20/40 px. |
| `deteccion/fallos_por_causa.csv` | Ancho, nitidez y pose de aciertos vs FN. |
| `asistencia/cobertura_video1.csv` | 9/9, recall por recorte, falsos desconocidos. |
| `asistencia/cobertura_video3.csv` | 8/8, `jose` ausente, 21 desconocidos. |
| `asistencia/confusion_sesion.csv` | Matriz de sesión de ambos videos. |
| `asistencia/por_persona_video1.csv` | Detalle de la corrida real. |
| `asistencia/presencia_por_parte.csv` | Matriz 11 partes × 9 estudiantes. |
| `asistencia/intrusiones.csv` | TP/FP/FN, precisión y recall de intrusos. |
| `asistencia/barrido_intrusos.csv` | Barrido de umbrales de la regla v2. |
| `asistencia/tracking.csv` | Pistas confirmadas y fragmentación por persona. |
| `asistencia/spoofing.csv` | Alarmas y confianza; sin ground truth. |
| `rendimiento/fps_y_latencia.csv` | Etapas, FPS, MJPEG/polling, RAM. |
| `rendimiento/cpu_gpu.csv` | CPU medido y GPU pendiente. |
| `resumen.md` | Informe consolidado, listo para citar. |

---

## 7. Cómo se cita en el documento

Cada tabla de `salida/resumen.md` indica su archivo fuente. La forma recomendada
de citar es:

> «Las métricas se calcularon con `experiments/unc/metricas/metricas.py` sobre
> los 300 recortes etiquetados del video 1 (86 positivos, 157 desconocidos). El
> AUC de verificación fue 0.9616 (IC 95 % 0.9276–0.9877) y el EER 0.0798. El
> umbral operativo 0.30 dio precisión 0.987 y recall 0.872 con 1 falso positivo
> en 157 desconocidos.»

Para la asistencia:

> «La corrida real del motor confirmó 9/9 estudiantes en el video 1 y 8/8 en el
> video 3 (`jose` ausente), con 0 errores de identidad. La regla de intrusos
> produjo 7 filas falsas en el video 1 (44 s) y 29 filas en el video 3 (21
> desconocidos reales + 8 registrados).»

---

## 8. Reproducibilidad y mantenimiento

- Semilla fija `20261105`; 2 000 remuestreos bootstrap.
- Los CSV se escriben con `utf-8` y `newline=""`.
- Las gráficas usan `matplotlib.use("Agg")`; no abren ventana.
- Si se agrega un video o una corrida nueva, basta con apuntar el script a los
  CSV correspondientes y volver a ejecutar. No hay estado oculto.
- `experiments/` está en `.gitignore`; para versionar esta carpeta hay que
  agregarla con `git add -f experiments/unc/metricas` (decisión del usuario).

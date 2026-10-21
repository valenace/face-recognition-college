# 03 — Bitácora de cálculos y funciones

**Propósito:** registrar cada cálculo matemático que hacemos, con su fórmula, dónde está en el código, qué librería lo ejecuta y un ejemplo con nuestros datos.

**Convención:** cada entrada tiene la misma estructura:
- **Qué calcula** (en palabras sencillas)
- **Fórmula**
- **Dónde** (archivo y función)
- **Librería y función**
- **Ejemplo con nuestros datos**

---

## A. Vectores y comparaciones

### A1. Norma de un vector (tamaño)
- **Qué calcula:** qué tan "grande" es una lista de números.
- **Fórmula:** ‖v‖ = √(v₁² + v₂² + … + v₅₁₂²)
- **Dónde:** verificaciones de `02_resultados_registro.md`; `registro_video.py` la usa al normalizar el centroide.
- **Librería:** `numpy.linalg.norm(v)`
- **Ejemplo:** la norma de cada centroide de la base de datos da **1.000000**.

### A2. Normalizar un vector (dividir por su norma)
- **Qué calcula:** deja el vector con norma 1, para que las comparaciones sean justas.
- **Fórmula:** v̂ = v / ‖v‖
- **Dónde:** `registro_video.py`, al crear el centroide: `centroide / np.linalg.norm(centroide)`.
- **Librería:** `numpy` (división + `np.linalg.norm`).

### A3. Similitud coseno
- **Qué calcula:** qué tan parecidas son dos caras. Va de -1 a 1.
- **Fórmula:** cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
  - Como los vectores ya tienen norma 1, la fórmula se reduce a **a · b**.
- **Dónde:** en todos los scripts; por ejemplo, la matriz de similitud y la decisión del motor.
- **Librería:** `numpy.dot(a, b)`; o `faiss.IndexFlatIP` en el motor (producto interno exacto).
- **Ejemplo:** `alexa` vs `dani` = **0.297**. Demostración de `v1_00020`: `dani` = **0.573**.

### A4. Producto punto (base de la similitud coseno)
- **Qué calcula:** suma de productos término a término.
- **Fórmula:** a · b = Σᵢ aᵢ·bᵢ
- **Librería:** `numpy.dot`.
- **Ejemplo:** con dos vectores de 512 números, se hacen 512 multiplicaciones y una suma.

### A5. Matriz de similitud entre centroides
- **Qué calcula:** la similitud de todas las personas contra todas.
- **Fórmula:** M = V · Vᵀ, donde V es una matriz de 9 filas (una por persona) y 512 columnas.
- **Dónde:** `02_resultados_registro.md`, sección 4.
- **Librería:** `numpy` (`@` o `np.dot`) con `V.T` para la transpuesta.
- **Resultado:** matriz de 9 × 9. La diagonal es 1.000 (cada persona consigo misma).

---

## B. Centroides y dispersión

### B1. Centroide (promedio de las fotos de una persona)
- **Qué calcula:** la "huella promedio" de una persona.
- **Fórmula:** c = (1/n) Σᵢ vᵢ, seguido de normalizar.
  - Con n = 50 fotos.
- **Dónde:** `registro_video.py`, función `registrar_desde_video`.
- **Librería:** `numpy.mean(vectores, axis=0)` y luego `numpy.linalg.norm`.

### B2. Dispersión interna (qué tan parecidas son las fotos de una persona)
- **Qué calcula:** para cada foto, su similitud con el centroide de su persona.
- **Fórmula:** sᵢ = cos(vᵢ, c) = vᵢ · c
- **Resúmenes que reportamos:**
  - **Media:** (1/n) Σ sᵢ
  - **Mediana:** el valor del medio cuando se ordenan los sᵢ
  - **Mínimo y máximo:** el más bajo y el más alto
  - **Desviación estándar:** √( (1/n) Σ (sᵢ − media)² )
- **Dónde:** `experiments/unc/fase2_dispersion.py`.
- **Librerías:** `numpy.dot`, `numpy.mean`, `numpy.median`, `numpy.min`, `numpy.max`, `numpy.std` (por defecto usa ddof = 0, es decir, divide entre n).
- **Ejemplo:** `kate` tiene media **0.807**; `jesus` tiene media **0.644** y mínimo **0.139**.

### B3. Prueba de quitar fotos atípicas (`jesus`, 2026-10-15)
- **Qué calcula:** cuánto cambia el centroide si quitamos 4 fotos.
- **Fórmula:**
  - Centroide nuevo: c' = mean(46 vectores) / ‖mean(46 vectores)‖
  - Cambio: cos(c, c') = c · c'
- **Resultados:**
  - cos(c, c') = **0.9965**: prácticamente el mismo vector.
  - Media de similitud de las 46 fotos buenas con c': **0.681** (antes 0.644).
  - Similitud de `jesus` con los otros 8: cambia como máximo **0.01**.
- **Librerías:** `numpy.mean`, `numpy.linalg.norm`, `numpy.dot`.
- **Decisión:** se mantiene el centroide original (ver `02_resultados_registro.md`).

---

## C. Imagen

### C1. Varianza del Laplaciano (nitidez)
- **Qué calcula:** qué tan enfocada está una imagen. El Laplaciano detecta bordes; si hay muchos bordes marcados, la imagen tiene detalle y está enfocada.
- **Fórmula:**
  - Primero, L = ∇²I ≈ convolución de la imagen en gris con un núcleo de segunda derivada.
  - Luego, nitidez = Var(L) = (1/N) Σ (Lᵢ − media(L))²
- **Dónde:** `core/utils_facial.py`, función `es_imagen_borrosa`.
- **Librerías:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`, `cv2.Laplacian(gray, cv2.CV_64F)`, `.var()` (de numpy).
- **Ejemplo:** `alexa` mediana **741.9**; `jesus` mediana **476.9**. El umbral del registro es **60**, así que nada fue descartado.

### C2. Brillo medio
- **Qué calcula:** qué tan claro es el recorte.
- **Fórmula:** brillo = promedio de todos los valores de gris de los píxeles.
- **Dónde:** `registro_video.py` (`np.mean(gray)`) y el análisis de `02`.
- **Librerías:** `cv2.cvtColor`, `numpy.mean`.
- **Ejemplo:** `dani` = **98.0**; `bob` = **131.9**.

### C3. CLAHE (mejora de contraste local)
- **Qué calcula:** ajusta el contraste por zonas pequeñas de la imagen, sin pasarse en las zonas ya claras.
- **Pasos:**
  1. Convertir de BGR a LAB (separa la luminosidad L de los colores a y b).
  2. Aplicar CLAHE solo a L, con `clipLimit = 2.0` y cuadrícula de `8×8`.
  3. Volver a juntar L, a y b, y convertir de LAB a BGR.
- **Dónde:** `core/utils_facial.py`, función `aplicar_clahe`.
- **Librerías:** `cv2.cvtColor` (BGR2LAB, LAB2BGR), `cv2.split`, `cv2.merge`, `cv2.createCLAHE(clipLimit, tileGridSize)`, `.apply(L)`.

### C4. Alineación de la cara (rotación y recorte)
- **Qué calcula:** endereza la cara usando la línea entre los dos ojos.
- **Fórmulas:**
  - Ángulo: θ = arctan2(Δy, Δx) × 180/π, donde Δx, Δy son la diferencia entre los dos ojos.
  - Distancia entre ojos: d = √(Δx² + Δy²)
  - Tamaño del recorte: ancho = 4.5 · d, alto = 5.5 · d.
  - Punto de inicio vertical: y₀ = centro_y − 0.4 · alto.
- **Dónde:** `core/utils_facial.py`, función `alinear_rostro`.
- **Librerías:** `numpy.arctan2`, `numpy.degrees`, `numpy.sqrt`; `cv2.getRotationMatrix2D(centro, ángulo, 1.0)`, `cv2.warpAffine(...)` para rotar, `cv2.resize(..., (224, 224))` para escalar.
- **Por qué hay esquinas negras:** la rotación deja zonas sin imagen en las esquinas. Lo vimos en las fotos de `jesus`.

### C5. Recorte con margen (muestra del salón)
- **Qué calcula:** la caja de la cara más un 20 % de margen en cada lado.
- **Fórmula:** margen_x = 0.2 · ancho; margen_y = 0.2 · alto. Luego se recorta con esas dimensiones, sin salirse de la imagen.
- **Dónde:** `experiments/unc/fase0_muestreo_salon.py`, función `recortar`.
- **Librerías:** `numpy` no; solo operaciones con enteros y el recorte de arreglos de `cv2`.

---

## D. Geometría de cajas

### D1. IoU (Intersection over Union)
- **Qué calcula:** qué tanto se solapan dos cajas.
- **Fórmula:** IoU = Área(A ∩ B) / Área(A ∪ B) = Área(A ∩ B) / (Área(A) + Área(B) − Área(A ∩ B))
- **Pasos:**
  1. Intersección: x₁ = max(xA₁, xB₁), y₁ = max(yA₁, yB₁), x₂ = min(xA₂, xB₂), y₂ = min(yA₂, yB₂).
  2. Área de intersección: max(0, x₂ − x₁) · max(0, y₂ − y₁).
- **Dónde:** `scripts/asistencia_unificada_optimizada.py`, método `_calcular_iou`.
- **Uso:** emparejar la caja del tracker con la detección (umbral 0.5). Para la evaluación del salón, lo usaremos para emparejar cada recorte etiquetado con la detección del frame.
- **Librería:** aritmética pura (no usa librería).

---

## E. Decisión y búsqueda

### E1. Decisión por umbral
- **Qué calcula:** si la mejor coincidencia es suficiente.
- **Regla:** si sim(mejor) > 0.35, se acepta esa identidad; si no, se dice "Desconocido".
- **Dónde:** `asistencia_unificada_optimizada.py`, método `_decidir_identidad`.

### E2. Búsqueda del mejor centroide
- **Qué calcula:** cuál de los 9 centroides tiene la mayor similitud con la cara.
- **Fórmula:** argmax_k (v · cₖ)
- **Dónde:** el motor usa FAISS con `IndexFlatIP`, que calcula el producto punto contra todos los centroides y devuelve el mayor.
- **Librerías:** `faiss.IndexFlatIP(512)`, `.add(vectores)`, `.search(consulta, 1)`.
- **Ejemplo (v1_00020):** `dani` 0.573, `alexa` 0.189, `carlos` 0.089, … `jesus` 0.004.

---

## F. Estadística de la muestra del salón

### F1. Muestra aleatoria reproducible
- **Qué calcula:** elige 300 recortes al azar, de forma que siempre sea la misma elección.
- **Dónde:** `experiments/unc/fase0_muestra_anotacion.py`, función `generar_muestra`.
- **Librería:** `random.Random(42).sample(lista, k)` (semilla fija 42).

### F2. Mediana y conteo de anchos de cara
- **Qué calcula:** el ancho típico de las caras del salón.
- **Dónde:** análisis de `02`/`00_explicacion_sencilla`. Se calculó con `statistics.median` y un conteo con `sum(1 for x in a if x < 40)`.
- **Resultado:** mediana **37 px**; **638** de **1 127** recortes con ancho menor a 40 px.

---

## G. Librerías usadas

| Librería | Para qué la usamos | Funciones principales |
|---|---|---|
| **numpy** | Vectores, promedios, normas, matrices | `dot`, `mean`, `median`, `std`, `linalg.norm`, `arctan2`, `allclose`, `asarray`, `reshape` |
| **OpenCV (cv2)** | Leer video y frames, imágenes, rotación, filtros | `VideoCapture`, `imread`, `imwrite`, `cvtColor`, `Laplacian`, `createCLAHE`, `getRotationMatrix2D`, `warpAffine`, `resize`, `rectangle` |
| **uniface** | Detectar caras y calcular embeddings | `create_detector("retinaface")`, `.detect(img)` → bbox, landmarks, confidence; `create_recognizer("arcface")`, `.get_normalized_embedding(img, landmarks)` |
| **faiss** | Búsqueda rápida del mejor centroide | `IndexFlatIP`, `add`, `search` |
| **supervision** | Seguimiento de personas en video (ByteTrack) | `sv.ByteTrack`, `update_with_detections`, `sv.Detections.empty()` |
| **matplotlib** | Gráficas | `scatter`, `hlines`, `axhline`, `savefig` |
| **pickle** | Guardar y leer la base de datos | `pickle.dump`, `pickle.load` |
| **csv** | Leer y escribir tablas | `csv.DictReader`, `csv.DictWriter`, `csv.writer` |
| **random** | Muestra reproducible | `random.Random(42).sample` |
| **statistics** | Mediana en el análisis de anchos | `statistics.median` |

---

## H. Registro de cálculos ejecutados

| Fecha | Cálculo | Entrada | Resultado | Documento |
|---|---|---|---|---|
| 2026-10-14 | Norma de los 9 centroides | `database_embeddings.pkl` | 1.000000 en los 9 | `02` §3 |
| 2026-10-14 | Matriz de similitud 9×9 | 9 centroides | máx. fuera de diagonal 0.297 | `02` §4 |
| 2026-10-14 | Nitidez (varianza Laplaciano) de 450 fotos | fotos de registro | 477 a 1 884 (mediana por persona) | `02` §3 |
| 2026-10-14 | Brillo medio por persona | fotos de registro | 98 a 133 | `02` §3 |
| 2026-10-14 | Mediana de ancho de cara del salón | 1 127 recortes | 37 px; 638 < 40 px | `00_plan` Fase 0 |
| 2026-10-14 | Dispersión interna (9 personas) | fotos y centroides | media 0.644 a 0.807 | `02` §7 |
| 2026-10-15 | Efecto de quitar 4 fotos de `jesus` | 50 fotos de `jesus` | cos(c, c') = 0.9965; decisión: no cambiar | `02` §7 |
| 2026-10-15 | IoU de cada recorte etiquetado con la detección de su frame | 300 filas de la muestra | 300/300 emparejadas; IoU = 1.0000 en todas | `experiments/unc/fase0_emparejar_iou.py` |

| 2026-10-15 | Prueba de estrés: caja del recorte perturbada ±10 % (20 veces por recorte, semilla 42) y se mide si el emparejamiento elige la misma detección | 300 recortes, 6 000 perturbaciones | acierto global 1.0000; 0 recortes con fallo | `experiments/unc/fase0_prueba_estres_iou.py` |

| 2026-10-15 | Reconocimiento de los 300 recortes: similitud con los 9 centroides, decisión con umbral 0.35, comparación con la etiqueta | 300 recortes emparejados | 243 evaluables: 223 aciertos, 16 falsos negativos, 4 falsos positivos; 57 no evaluables (`dudoso` 44, `no_cara` 13); 0 errores de identidad | `experiments/unc/fase0_reconocimiento_salon.py` |

**Nota sobre la prueba de estrés:** ±10 % es una perturbación pequeña. El acierto perfecto indica que las caras del salón están lo bastante separadas para que la caja correcta siempre gane con ese nivel de ruido. Falta probar con perturbaciones mayores (por ejemplo ±30 %) para encontrar el punto donde empieza a fallar.

**Nota sobre el último cálculo:** IoU = 1.0 es lo esperado, porque el recorte salió de la misma detección sobre el mismo frame. Confirma que el frame se leyó bien, pero no prueba el emparejamiento en casos difíciles (caras juntas, cajas parecidas). Para eso hay que medir sobre detecciones de otra corrida o con cajas perturbadas.


 Lo que medí fue esto:                                                                                                           
                                                                                                                                     
  1. Tomé las 46 fotos buenas de jesus (sin las 4 atípicas).                                                                      
  2. Calculé su promedio (sumé los 46 vectores y dividí entre 46).                                                                
  3. Lo normalicé: lo dividí entre su norma para que tuviera tamaño 1. Ese es el                                                  
    centroide nuevo.                                                                                                             
  4. Medí qué tanto cambió comparándolo con el centroide original usando el producto                                              
    punto. El resultado fue 0.9965, casi 1. Eso quiere decir que los dos vectores                                                
    apuntan prácticamente igual.                                                                                                 
                                                                                                                                  
  Además comparé la similitud de jesus con las otras 8 personas, antes y después. El                                              
  cambio fue como máximo de 0.01.      
| 2026-10-15 | Revisión visual de los 16 falsos negativos | 16 recortes | confirmados como la persona etiquetada; etiquetas sin cambios | `experiments/unc/salon_muestra/revision_falsos_negativos.jpg` |

| 2026-10-15 | Pose por recorte: |nariz − centro de ojos| / distancia entre ojos (landmarks RetinaFace) | 87 recortes con nombre | frente 4.1 % de fallo; perfil 44.8 %; nitidez no explica los fallos | `experiments/unc/fase0_causas_fallos.py` |

| 2026-10-15 | Corrección de `v1_00789` (jose → desconocido) y recálculo de reconocimiento y causas | 300 recortes | 228 aciertos, 15 falsos negativos, 0 falsos positivos; perfil 42.9 % de fallo | `02_resultados_registro.md` §13 |

| 2026-10-15 | Fase 1: tasas por video (detección, cara única, útil), ancho de caja y frames útiles estimados | 9 videos de registro (paso 3) y 1 068 frames del salón | registro: detección 0.98–1.00, útil 0.93–1.00; salón: mediana 13 caras/frame, 58 % de caras < 40 px | `experiments/unc/fase1_diagnostico_dataset_unc.py`, `docs/analisis/unc/01_diagnostico_dataset_unc.md` |

| 2026-10-15 | Efecto del filtro de tamaño (40 px, 20 px, sin filtro) sobre el reconocimiento ya calculado | 300 recortes | 40 px: alexa, bob, kate, kevin 0 % reconocidos; 20 px: 75 %, 64 %, 77 %, 100 %; falsos positivos 0 en todos | `experiments/unc/fase0_efecto_filtro_tamano.py` |

| 2026-10-15 | Motor real sobre 30 s del salón con filtro 40 px y 20 px; verificación contra recortes etiquetados por IoU | 40 y 4 072 decisiones | filtro 20: 25 caras antes ignoradas ahora procesadas; 0 identidades incorrectas; 0 desconocidos confirmados | `experiments/unc/demo_video_salon.py`, `docs/analisis/unc/02_resultados_registro.md` §18 |

| 2026-10-15 | Video completo (107 s) con filtro 40 px y 20 px; verificación contra 300 recortes | 13 893 y 33 534 decisiones | personas reconocidas: 33 → 75 de 86; desconocidos correctos: 60 → 156 de 157; FRAUDE: 6 → 39 (8 pistas, cajas 28–38 px) | `experiments/unc/demo/`, `02_resultados_registro.md` §19 |

| 2026-10-16 | Revisión de 8 pistas con FRAUDE: score de anti-spoofing con caja exacta vs redondeada | pista 372, frame 1880 | exacta 0.805 (fraude), redondeada 0.970 (real); las 8 pistas son personas reales | `experiments/unc/fase0_revision_fraudes.py`, `02` §20 |

| 2026-10-16 | Anti-spoofing omitido bajo 60 px; recorrida completa de nuevo | 107 s, filtro 20 px | FRAUDE 39 → 3; reconocidas 75 → 76 de 86; desconocidos 156 | `scripts/asistencia_unificada_optimizada.py`, `02` §21 |

| 2026-10-16 | Pistas confirmadas como `bob` en el video completo (filtro 20 px) | 1 493 decisiones | 6 pistas, todas la misma persona; pista 369 dura 35 s | `experiments/unc/fase0_revision_pistas.py`, `02` §22 |

| 2026-10-16 | Verificación de pistas de la corrida 38–107 s (jesus p90/p214/p332, bob, alexa, kate, dani, fran) | 196 pistas, 29 393 decisiones | todas las pistas revisadas son la persona etiquetada; jesus reconocido con gorra y gafas a sim 0.36–0.45 | `experiments/unc/demo/`, `02` §23 |

| 2026-10-16 | Verificación de 2 capturas erróneas + costo de filtrar por confianza de detección | 2 frames puntuales; 1 127 recortes | falsos positivos con conf 0.67 y 0.55; umbral 0.70 perdería 5/86 personas y 13/157 desconocidos | `02` §24 |

| 2026-10-16 | Corrida con filtro de confianza 0.75 y verificación contra la muestra | 79+131 recortes en el tramo | 67/79 personas confirmadas (85 %), 0 identidades incorrectas, 0 falsos positivos | `experiments/unc/demo/prueba_1610_1441/`, `02` §25 |

| 2026-10-19 | Margen top-2 (sim_1 − sim_2) en todas las decisiones de la corrida 38–107 s | 17 094 filas | registrados: margen mediano 0.333, ninguno < 0.05; Desconocido: mediana 0.034; 0 confusiones registrado-registrado | `experiments/unc/demo/prueba_1910_1436/`, `02` §26 |

| 2026-10-19 | Costo de revalidar el ancla cada 30 en vez de 90 frames | corrida 38–107 s (2 069 frames, 8 118 frames anclados en 31 pistas) | revalidaciones 78 → 256 (+178 ciclos, ~534 embeddings extra); detección 15.6 ms/frame y embedding 6.6 ms/cara → +3.5 s sobre ~145 s de cómputo total (**+2.4 %**) | `02` §27 |

| 2026-10-19 | Fase 4: ROC/AUC, Youden y métricas operativas por umbral | 86 positivos + 243 negativos | AUC 0.744; Youden 0.28 (TPR 0.919, FPR 0.021); F1 máximo 0.940 en 0.28; en 0.30 P 0.987 R 0.872 F1 0.926 | `docs/analisis/unc/04_resultados_curva_roc.md` |

| 2026-10-19 | Fase 4b: umbral con 1 frame vs 3 frames consecutivos (seguimiento por IoU) | 243 recortes (231 con 3 frames) | 0.28: 3f P 0.962 R 0.884 F1 0.921 (76/3/10); 0.30: 3f P 1.000 R 0.849 F1 0.918 (73/0/13) | `04_resultados_curva_roc.md` §7 |

| 2026-10-19 | Corrida de generalización en el video 2 (50 s) con umbral 0.30 | 1 519 frames, sin etiquetas | 6 personas confirmadas; dani/carlos revisados visualmente y consistentes; fran en el límite (0.3031) | `experiments/unc/demo/prueba_1910_1602_video2/`, `04` §9 |

| 2026-10-20 | Fase 3: similitud al propio centroide por condición de accesorio + Mann-Whitney lentes vs sin accesorios | 86 recortes (44 sin, 35 lentes, 3 gorra, 4 ambos) | lentes p=0.512 (sin efecto); gorra sola mediana 0.582; gorra+lentes mediana 0.315 (2/4 pasan con 0.30) | `03_resultados_robustez_accesorios.md` |

| 2026-10-20 | Fase 6: evaluación del motor final (umbral 0.30, conf 0.75, ancla 30, 3 votos) sobre el video 1 | 86 recortes con nombre, 157 desconocidos | cobertura 9/9; tiempo de confirmación 0.06–0.26 s; recorte inmediato 63/86 (73.3 %); ventana ±15 frames 76/80 (95 %); 0 errores de identidad; 1 falso positivo de 157 | `06_resultados_evaluacion_salon.md` |

| 2026-10-20 | Fase 7: reporte final con consolidación de resultados y configuración definitiva | fases 0–6 | 9/9 cobertura; 95 % en ventana; 0 errores de identidad; 0.6 % FPR; umbral 0.30 | `docs/analisis/unc/07_reporte_final_unc.md` |

| 2026-10-20 | Fase 8: verificación de continuidad del video 3 partido en 11 archivos (miniaturas 32×18, MAE contra el original) | 11 partes, 3 868 frames vs 3 710 del original | cadena continua del frame 0 al 3 709; MAE 0.06–0.23; 60 frames negros en la parte 1; +98 frames duplicados por re-exportar a 30.00 fps | `08_resultados_video3.md` §2 |

| 2026-10-20 | Fase 8: corrida del motor final sobre las 11 partes, motor reiniciado por parte | 3 808 frames reales, 38 756 decisiones | 8/8 registrados presentes confirmados; `jose` (ausente) 0 confirmaciones; 7 403 confirmaciones; similitud media 0.328–0.497 | `08_resultados_video3.md` §3 |

| 2026-10-20 | Fase 8: verificación de identidad por pista (78 confirmaciones revisadas) | 78 pistas, 1 recorte por pista | 77 correctas, **1 falsa identidad** (`dani`, parte 7 pista 3, sim 0.336, 30 frames) → precisión 0.987 | `08_resultados_video3.md` §4 |

| 2026-10-20 | Fase 8: separación a nivel de pista entre registrados y no registrados | 78 pistas confirmadas vs 74 no confirmadas de ancho ≥ 40 px; 21 estudiantes no registrados | sim_max de las no registradas: p50 0.226, p95 0.306, máx 0.344; con 1 voto a 0.30 pasan 4/74, con 3 votos 1/74; **FP 1/21 personas (4.8 %) y 1/74 pistas (1.4 %)** | `08_resultados_video3.md` §5 |

| 2026-10-20 | Fase 8: costo de re-confirmar tras cada corte (reinicio de estado) | 62 re-confirmaciones en 10 empalmes | mediana 1.31 s, media 2.57 s, máximo 13.2 s | `08_resultados_video3.md` §6 |

| 2026-10-20 | Fase 8: tasa de reconocimiento por tamaño de cara | 37 334 decisiones (sin `CENTRATE`) | 20–30 px: 9.6 %; 30–40 px: 20.0 %; 40–60 px: 31.0 %; total 19.8 % | `08_resultados_video3.md` §7 |

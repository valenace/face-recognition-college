# Guía Metodológica: Cómo Analizar un Pipeline de Reconocimiento Facial
**Proyecto:** CV College — Sistema de Asistencia Biométrica  
**Propósito:** Explicar en detalle todos los cálculos y análisis realizados, desde conceptos básicos hasta avanzados.

---

## Parte 1: Fundamentos — ¿Qué estamos midiendo?

### 1.1 ¿Qué es un embedding?

Un **embedding** es un vector matemático de números reales que representa una cara humana.
ArcFace, la red neuronal que usamos, convierte una imagen de un rostro en un vector de **512 números**.

```
Imagen de cara → ArcFace CNN → [0.021, -0.134, 0.887, ..., 0.043]  ← 512 números
```

La propiedad mágica: **caras de la misma persona producen vectores similares; caras distintas producen vectores alejados**.

### 1.2 ¿Qué es la similitud coseno?

Para medir qué tan "parecidos" son dos vectores, usamos la **similitud coseno**. Imagina los vectores como flechas en el espacio:

```
sim(A, B) = A · B / (||A|| × ||B||)
```

Donde:
- `A · B` = producto punto (multiplicar elemento a elemento y sumar todo)
- `||A||` = norma (longitud de la flecha)

Como nuestros embeddings están **normalizados** (norma = 1.0), la fórmula se simplifica a:

```python
similitud = np.dot(vec_A, vec_B)  # Simplemente el producto punto
```

**Valores posibles:**
- `1.0` = vectores idénticos (misma persona, misma foto)
- `0.0` a `0.5` = personas distintas (lo normal entre caras diferentes)
- `< 0.0` = vectores opuestos (raro, pero posible)

### 1.3 ¿Qué es un centroide?

El **centroide** es el promedio de varios vectores. Si registramos 25 fotos de una persona:

```python
vectores = [vec_1, vec_2, ..., vec_25]          # 25 embeddings de 512 dims
centroide = np.mean(vectores, axis=0)            # Promedio elemento a elemento
centroide_norm = centroide / np.linalg.norm(centroide)  # Normalizar
```

Geométricamente: el centroide es el **centro de masa** de la nube de puntos en el espacio de 512 dimensiones. Es el punto más representativo de una persona.

---

## Parte 2: Análisis Realizados y Cómo Reproducirlos

### 2.1 Inventario del Dataset

**¿Qué hicimos?** Contar frames y duración de cada video de registro.

**¿Cómo lo hicimos?**

```python
import cv2
from pathlib import Path

base = Path('data/dataset_cv/videos')
for persona_dir in sorted(base.iterdir()):
    for vid in sorted(persona_dir.iterdir()):
        if vid.suffix.lower() in ['.mp4', '.avi', '.mov']:
            cap = cv2.VideoCapture(str(vid))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duracion = total_frames / fps  # segundos
            
            cap.release()
            print(f'{vid.name}: {total_frames} frames @ {fps:.1f} fps = {duracion:.1f}s | {width}x{height}')
```

**¿Por qué importa la cantidad de frames?** Más frames del video = más posibilidad de capturar poses diversas. Pero si el video es estático (persona quieta), 400 frames pueden producir los mismos 5 frames útiles que un video de 100 frames con movimiento.

**Comando de consola para reproducirlo:**

```bash
source venv/bin/activate
python3 -c "
import cv2
from pathlib import Path
base = Path('data/dataset_cv/videos')
for d in sorted(base.iterdir()):
    for v in sorted(d.iterdir()):
        if v.suffix == '.mp4':
            cap = cv2.VideoCapture(str(v))
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f'{d.name}/{v.name}: {tf}f @ {fps:.1f}fps = {tf/fps:.1f}s')
"
```

---

### 2.2 Verificación de la Base de Datos de Embeddings

**¿Qué hicimos?** Cargar el archivo `.pkl` y verificar que todos los vectores estén correctamente normalizados.

**¿Cómo lo hicimos?**

```python
import pickle
import numpy as np

# Cargar la base de datos
with open('data/database_embeddings.pkl', 'rb') as f:
    db = pickle.load(f)

# Verificar cada persona
for nombre, vector in db.items():
    norma = np.linalg.norm(vector)
    print(f'{nombre}: dimensión={vector.shape}, norma={norma:.4f}')
```

**¿Por qué verificar la norma?** Un vector normalizado tiene norma = 1.0. Si la norma no es 1.0, la similitud coseno no funcionará correctamente porque la fórmula asume vectores unitarios. En nuestro caso, todos tienen norma 1.0000 ✅.

**¿Qué es un archivo `.pkl`?** Es un archivo binario de Python (formato `pickle`) que puede guardar cualquier objeto Python — en este caso, un diccionario `{nombre: vector_numpy}`.

**Comando para reproducirlo:**

```bash
python3 -c "
import pickle, numpy as np
db = pickle.load(open('data/database_embeddings.pkl', 'rb'))
print('Personas:', list(db.keys()))
for k, v in db.items():
    print(f'  {k}: shape={v.shape}, norm={np.linalg.norm(v):.4f}')
"
```

---

### 2.3 Matriz de Similitudes Inter-Clase

**¿Qué hicimos?** Calcular la similitud coseno entre TODOS los pares posibles de personas.

**¿Por qué?** Para verificar que ningún par de personas distintas tenga similitud alta (lo que indicaría riesgo de confusión).

**¿Cómo lo hicimos?**

```python
import pickle
import numpy as np

db = pickle.load(open('data/database_embeddings.pkl', 'rb'))
nombres = list(db.keys())
vectores = np.array(list(db.values()))  # Matriz de shape (9, 512)

# La multiplicación matricial calcula TODAS las similitudes de una vez
# vectores @ vectores.T  ← esto es una multiplicación de matrices (9x512) × (512×9) = (9×9)
matriz_similitudes = vectores @ vectores.T
# Resultado: una matriz 9×9 donde [i,j] = similitud entre persona i y persona j

# Imprimir solo la triangular superior (evitar duplicados)
for i, n1 in enumerate(nombres):
    for j, n2 in enumerate(nombres):
        if i < j:
            print(f'{n1} vs {n2}: {matriz_similitudes[i,j]:.4f}')
```

**¿Por qué multiplicación matricial?** La similitud coseno entre vectores normalizados es simplemente el producto punto. Y `A @ B.T` calcula el producto punto de TODAS las combinaciones de filas de `A` con columnas de `B` simultáneamente. Es la forma más eficiente de calcular una matriz de similitudes.

**Interpretación de valores:**
- `sim > 0.5` = ⚠️ RIESGO — personas muy similares
- `sim 0.3 – 0.5` = 🟡 Vigilar — pueden confundirse en condiciones adversas
- `sim < 0.3` = ✅ Bien separadas

**Comando para reproducirlo:**

```bash
python3 -c "
import pickle, numpy as np
db = pickle.load(open('data/database_embeddings.pkl', 'rb'))
nombres = list(db.keys())
vecs = np.array(list(db.values()))
sim = vecs @ vecs.T
for i in range(len(nombres)):
    for j in range(i+1, len(nombres)):
        print(f'{nombres[i]} vs {nombres[j]}: {sim[i,j]:.4f}')
" | sort -t: -k2 -rn | head -15
# El | sort ... ordena por similitud descendente para ver los pares más cercanos
```

---

### 2.4 Análisis del CSV de Asistencia

**¿Qué hicimos?** Leer el historial de asistencias registradas para observar el rango de similitudes reales del sistema en operación.

**¿Por qué?** Las similitudes del CSV representan el rendimiento REAL del sistema en sesiones anteriores. Si vemos que `ronald` siempre aparece con similitud 0.36–0.38 (rozando el umbral de 0.35), sabemos que es un candidato frágil.

**¿Cómo lo hicimos?**

```python
import csv
from collections import defaultdict

similitudes_por_persona = defaultdict(list)

with open('data/asistencia.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        nombre = row['Identificador']
        similitud = float(row['Similitud'])
        similitudes_por_persona[nombre].append(similitud)

for nombre, sims in sorted(similitudes_por_persona.items()):
    print(f'{nombre}: min={min(sims):.4f}, max={max(sims):.4f}, mean={sum(sims)/len(sims):.4f}, n={len(sims)}')
```

**Conceptos estadísticos usados:**
- **Mínimo**: la peor condición en que fue reconocido (¿qué tan cerca del umbral estuvo?)
- **Máximo**: la mejor condición (frontal, buena iluminación)
- **Media**: rendimiento típico
- **n**: cuántas veces fue registrado en total

---

### 2.5 Análisis del Umbral de Decisión

**¿Qué es el umbral y por qué importa?**

El umbral `τ` es el valor de similitud mínima para aceptar un reconocimiento:
```
sim(query, centroide) > τ  →  PERSONA RECONOCIDA
sim(query, centroide) ≤ τ  →  DESCONOCIDO
```

**¿Cómo se calcula el umbral óptimo? (Teoría)**

Necesitamos dos distribuciones:
1. **Intra-clase** (misma persona): distribución de similitudes cuando la query es genuina
2. **Inter-clase** (persona diferente): distribución de similitudes cuando la query es un impostor

El umbral ideal es el punto donde estas dos distribuciones se cruzan mínimamente.

**¿Cómo calculamos el umbral preliminar de 0.42?**

Basado en el CSV de asistencia:
```python
todas_similitudes = [0.4018, 0.3747, 0.4006, 0.3557, 0.4005, 0.3807,
                     0.4005, 0.4006, 0.3856, 0.3802, 0.4444, ...]

# Percentil 20 del lado inferior (para que el 80% de reconocimientos genuinos pasen)
umbral_propuesto = np.percentile(todas_similitudes, 20)
# Resultado: ~0.38–0.42 dependiendo de los datos
```

**Nota**: Este es solo un estimado preliminar. La Fase 3 calculará el umbral óptimo de forma rigurosa usando la Curva ROC.

---

## Parte 3: Conceptos Avanzados

### 3.1 La Curva ROC (Receiver Operating Characteristic)

La curva ROC es el análisis más importante para sistemas de reconocimiento. Mide el tradeoff entre:

- **TPR (True Positive Rate / Sensibilidad):**
  ```
  TPR = TP / (TP + FN)
  
  TP = Personas reales correctamente reconocidas
  FN = Personas reales NO reconocidas (falso negativo)
  ```
  → "¿Qué fracción de los alumnos reales identifica el sistema?"

- **FPR (False Positive Rate):**
  ```
  FPR = FP / (FP + TN)
  
  FP = Personas falsas aceptadas (el sistema las reconoce como alguien registrado)
  TN = Personas falsas correctamente rechazadas
  ```
  → "¿Qué fracción de impostores o desconocidos acepta el sistema?"

**¿Cómo se construye la curva ROC?**

```python
umbrales = np.arange(0.1, 0.8, 0.02)  # Probar 35 umbrales distintos
tprs, fprs = [], []

for tau in umbrales:
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for frame_genuino in frames_de_personas_registradas:
        sim = calcular_similitud(frame_genuino)
        if sim > tau:
            tp += 1   # Correcto: persona reconocida
        else:
            fn += 1   # Error: persona NO reconocida (umbral muy alto)
    
    for frame_impostor in frames_de_personas_NO_registradas:
        sim = calcular_similitud(frame_impostor)
        if sim > tau:
            fp += 1   # Error: impostor aceptado (umbral muy bajo)
        else:
            tn += 1   # Correcto: impostor rechazado
    
    tprs.append(tp / (tp + fn))
    fprs.append(fp / (fp + tn))

# Graficar FPR (eje X) vs TPR (eje Y)
```

**¿Qué buscamos en la gráfica?**
```
    TPR
 1.0 ┤    ┌────────────────────
     │   /
 0.5 ┤  /  ← Zona de operación
     │ /      El punto óptimo está
 0.0 ┤/        en la esquina sup-izq
     └──────────────────────────
     0.0  0.5  1.0   FPR
```

- **Diagonal**: Sistema aleatorio (inútil)
- **Esquina superior izquierda (0,1)**: Sistema perfecto
- **Punto óptimo**: El punto de la curva más cercano a (0,1)

### 3.2 El EER (Equal Error Rate)

El EER es el punto donde **FPR = FNR** (tasa de falso negativo).

```
FNR = 1 - TPR   (personas reales rechazadas)
EER = punto donde FPR == FNR
```

Es útil como medida única del rendimiento: un EER de 5% significa que el sistema comete el 5% de errores en ambas direcciones.

**Cálculo:**
```python
fnrs = [1 - tpr for tpr in tprs]
# Encontrar donde FPR ≈ FNR
eer_idx = np.argmin(np.abs(np.array(fprs) - np.array(fnrs)))
eer_value = (fprs[eer_idx] + fnrs[eer_idx]) / 2
eer_threshold = umbrales[eer_idx]
print(f'EER = {eer_value:.1%} en umbral = {eer_threshold:.2f}')
```

### 3.3 El Índice de Youden

Para elegir el "mejor" umbral, usamos el **Índice de Youden (J)**:

```
J = TPR + (1 - FPR) - 1
J = TPR - FPR
```

El umbral óptimo es aquel que **maximiza J**. Esto balancea sensibilidad y especificidad por igual.

```python
J_scores = [tpr - fpr for tpr, fpr in zip(tprs, fprs)]
idx_optimo = np.argmax(J_scores)
umbral_optimo = umbrales[idx_optimo]
print(f'Umbral óptimo (Youden): {umbral_optimo:.2f}')
print(f'  TPR = {tprs[idx_optimo]:.1%}')
print(f'  FPR = {fprs[idx_optimo]:.1%}')
```

**¿Cuándo preferir un umbral más alto que el de Youden?**
Para sistemas de asistencia, el costo de un FP (aceptar un impostor) es mayor que un FN (alumno hace el proceso de nuevo). En ese caso, movemos el umbral hacia arriba hasta que FPR sea < 1-2%.

### 3.4 FAISS: Búsqueda Vectorial Eficiente

**¿Qué es FAISS?** Una biblioteca de Facebook AI Research para buscar vectores similares de forma muy rápida.

**¿Por qué no simplemente comparar contra todos los vectores?**

Con 9 personas: comparar contra 9 vectores es trivial.  
Con 10,000 personas: comparar contra 10,000 vectores por frame × 30 fps = 300,000 operaciones/segundo. FAISS optimiza esto.

**¿Cómo lo usamos?**

```python
import faiss
import numpy as np

# Crear el índice (solo una vez al inicio)
# IndexFlatIP = Producto Interno (Inner Product) = similitud coseno para vectores normalizados
indice = faiss.IndexFlatIP(512)  # 512 = dimensión de ArcFace

# Agregar los vectores de la DB
vectores_db = np.array(list(db.values())).astype('float32')  # FAISS requiere float32
indice.add(vectores_db)

# Buscar el más similar (en cada frame, para cada cara detectada)
vec_query = np.array([embedding_nuevo]).astype('float32')  # Shape: (1, 512)
distancias, indices = indice.search(vec_query, k=1)  # Top-1 más similar

# distancias[0][0] = similitud con el candidato más cercano
# indices[0][0] = índice del candidato en la lista de nombres
candidato = nombres_lista[indices[0][0]]
similitud = distancias[0][0]
```

**Para Top-2 (lo que implementaremos en Fase 4):**

```python
distancias, indices = indice.search(vec_query, k=2)  # Top-2
mejor_sim = distancias[0][0]
segunda_sim = distancias[0][1]
margen = mejor_sim - segunda_sim

# Aceptar solo si el margen es suficientemente grande
if mejor_sim > umbral and margen > 0.05:
    candidato_aceptado = nombres_lista[indices[0][0]]
else:
    candidato_aceptado = "Desconocido"
```

**¿Por qué el margen importa?**
Si `angel` tiene similitud 0.45 y `lino` tiene 0.43, el margen es 0.02 — muy pequeño. El sistema no está seguro. Si exigimos margen > 0.05, este caso se rechaza como "Desconocido" y se evita un posible error.

---

## Parte 4: Herramientas y Comandos de Diagnóstico Rápido

### 4.1 Auditar la DB en cualquier momento

```bash
cd /ruta/al/proyecto
source venv/bin/activate
python3 scripts/auditar_db.py
```

### 4.2 Ver similitudes entre dos personas específicas

```bash
python3 -c "
import pickle, numpy as np
db = pickle.load(open('data/database_embeddings.pkl', 'rb'))
a, b = 'angel', 'lino'
sim = np.dot(db[a], db[b])
print(f'{a} vs {b}: {sim:.4f}')
"
```

### 4.3 Ver todos los registros del CSV ordenados por similitud

```bash
sort -t',' -k4 -n data/asistencia.csv
```

### 4.4 Verificar cuántos frames tiene cada persona

```bash
for d in data/dataset_cv/frames/*/; do
    echo "$(basename $d): $(ls $d | wc -l) frames"
done
```

### 4.5 Ver metadata de un video específico

```python
import cv2
cap = cv2.VideoCapture('data/salon/20260806_105645.mp4')
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
cap.release()
```

---

## Parte 5: Glosario de Términos Clave

| Término | Significado Simple |
|---|---|
| **Embedding** | Un número que resume una cara en 512 dimensiones |
| **Similitud Coseno** | Qué tan "apuntadas en la misma dirección" están dos caras |
| **Centroide** | El promedio de todas las fotos de una persona |
| **Umbral (τ)** | El mínimo de similitud para aceptar un reconocimiento |
| **TPR** | De cada 100 alumnos reales, ¿cuántos reconoce el sistema? |
| **FPR** | De cada 100 impostores, ¿cuántos acepta el sistema? |
| **Falso Positivo (FP)** | El sistema dice "es X" pero NO es X |
| **Falso Negativo (FN)** | El sistema dice "desconocido" pero SÍ estaba registrado |
| **Curva ROC** | Gráfica que muestra el tradeoff FPR-TPR para todos los umbrales |
| **EER** | El umbral donde los errores de FP y FN son iguales |
| **FAISS** | Motor de búsqueda de vectores similares (muy rápido) |
| **ByteTrack** | Algoritmo que sigue personas entre frames de video |
| **ArcFace** | Red neuronal que convierte caras en embeddings |
| **RetinaFace** | Red neuronal que detecta y localiza caras en imágenes |
| **CLAHE** | Técnica de normalización de iluminación en imágenes |
| **Liveness** | Verificar que la cara es real (no foto/pantalla) |

---

*Este documento se actualiza con cada nuevo análisis o experimento realizado en el proyecto.*

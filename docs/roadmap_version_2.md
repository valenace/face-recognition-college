# Roadmap y Plan de Mejoras — Versión 2 (Futura)

Este documento recopila las características, optimizaciones y extensiones arquitectónicas reservadas para la **Versión 2** del sistema, manteniendo la Versión 1 acotada a los requerimientos formales del proyecto (un centroide medio por estudiante, motor robusto y panel funcional).

---

## 1. Motor Biométrico (Versión 2)

### 1.1 Multi-Prototipo por Estudiante (Fase 4 del Motor)
- **Motivación:** En la Versión 1, el estudiante se modela con un único vector medio (centroide de 50 fotos). Debido a la geometría de alta dimensión (512-D), promediar fotos con distintas poses traslada el centroide hacia zonas de mayor densidad global (efecto *hub*), atrayendo similitudes espurias de caras no registradas (ej. falso positivo residual de Dani a 0.336 en video 3).
- **Diseño propuesto:**
  - Clusterizar las 50 fotos del registro en **3 a 5 prototipos** usando k-Means esférico (representando frontal, perfiles leves, inclinaciones y expresiones).
  - Regla de similitud **Top-3 k-NN**: comparar la cara detectada contra todos los prototipos y promediar los 3 más cercanos del mismo estudiante.
  - Resultados medidos en laboratorio: el falso positivo baja de 0.336 a **0.291** (quedando por debajo del umbral de corte 0.30 y por debajo de todos los estudiantes reales).
  - Adaptación de base de datos: tabla `FacePrototype` en PostgreSQL con `pgvector` o múltiples vectores por estudiante.

### 1.2 Procesamiento por Lotes (*Batching*) en ArcFace
- **Motivación:** En la Versión 1, cuando RetinaFace detecta varias caras en un frame, se extrae el embedding de cada rostro uno por uno en un bucle secuencial en Python (`for rostro in rostros`).
- **Diseño propuesto:** Apilar todos los rostros recortados y alineados por RetinaFace en un único tensor de NumPy `[N, 3, 112, 112]` para procesarlos en una sola llamada vectorizada a ArcFace y FAISS, reduciendo latencia sin alterar los modelos.

### 1.3 Módulo de Re-Identificación (Re-ID) de Apariencia
- **Motivación:** ByteTrack asocia únicamente por IoU espacial. Giros bruscos u oclusiones momentáneas crean un nuevo `track_id`, obligando a acumular 3 votos desde cero y generando pistas efímeras en estado desconocido.
- **Diseño propuesto:** Memoria de corto plazo (3–5 segundos) que enlace pistas recién aparecidas con pistas perdidas si el descriptor visual de baja dimensión coincide espacial y temporalmente.

### 1.4 Aceleración con GPU (NVIDIA TensorRT / CUDA)
- **Motivación:** La Versión 1 procesa en CPU a ~7 FPS.
- **Diseño propuesto:** Compilar el runtime con `onnxruntime-gpu` y `CUDAExecutionProvider` para aprovechar la tarjeta gráfica RTX 4090, permitiendo procesar múltiples flujos de cámara concurrentes a >60 FPS.

---

## 2. Panel Web y Gestión Académica (Versión 2)

### 2.1 Exportación de Reportes a Excel / CSV y PDF
- **Motivación:** Facilitar el envío de actas de asistencia física a secretaría y coordinaciones académicas.
- **Diseño propuesto:**
  - Botón en el detalle de sesión (`/sesiones/<id>/`) para descargar `.xlsx` (mediante `openpyxl` o similar) con columnas: Carnet, Apellidos y Nombres, Estado (Asistió, Falta, Justificada), Hora de Entrada, Última vez visto, y Porcentaje de Permanencia.
  - Generador de actas imprimibles en PDF con encabezado institucional.

### 2.2 Analítica Académica y Métricas Consolidadas
- **Motivación:** Seguimiento longitudinal de ausentismo y deserción estudiantil.
- **Diseño propuesto:**
  - Reactivación de `MetricasAlumnoView` con gráficos interactivos (asistencia acumulada en el semestre, alertas de inasistencia > 20%).
  - Dashboard para directores y coordinadores con estadísticas globales por materia, profesor y salón.

### 2.3 Panel Lateral Interactivo en Vivo (Presentes vs. Ausentes)
- **Motivación:** Permitir que el docente vea en una columna lateral la nómina completa del curso en tiempo real conforme entran al aula.
- **Diseño propuesto:** Lista conectada vía WebSocket o eventos de streaming que actualice en tiempo real avatares en verde (presentes) y gris (pendientes).

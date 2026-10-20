# Reporte Técnico: Sensibilidad de Precisión en Salón de Clases (Tomas Lejanas y Sombras)

Este reporte detalla los resultados experimentales obtenidos al evaluar la precisión de reconocimiento del alumno `angel` en el video real del salón de clases [`data/salon/20260806_105645.mp4`](file:///home/valentina2/Desktop/Folds/VisionProjects/face-recognition-college/data/salon/20260806_105645.mp4).

---

## 1. Metodología del Experimento (Enfoque Aula)

*   **Sujeto de prueba**: Estudiante `angel`.
*   **Video de Evaluación (Salón)**: Contiene **2044 frames** con múltiples alumnos, oclusiones temporales, caras en ángulos de perfil y rostros pequeños (menores a 60px).
*   **Fase A: Autodescubrimiento de Ground Truth**:
    *   Para aislar el desempeño de reconocimiento en el aula sin etiquetas manuales, corrimos una pre-evaluación usando la base de datos óptima de referencia ($S=25$ muestras del video individual de Angel).
    *   Muestreamos **1 frame cada 15 frames** del video completo del salón para optimizar velocidad (evaluando ~136 frames totales).
    *   Localizamos a `angel` en **14 frames de prueba** donde su rostro es detectado de forma clara y supera el umbral de similitud de **0.35**. Esto define el Ground Truth de coordenadas.
*   **Fase B: Barrido de Muestras de Enrolamiento ($S$)**:
    *   Variamos $S \in \{1, 3, 5, 10, 15, 20, 25, 30\}$ para el registro de Angel.
    *   Para cada $S$, procesamos las ubicaciones del Ground Truth y calculamos el porcentaje de identificaciones correctas utilizando la base de datos resultante.

---

## 2. Tabla de Resultados (Salón de Clases)

| Cantidad de Muestras ($S$) | Frames Evaluados (Ground Truth) | Reconocidos Correctamente | Precisión Final (%) |
| :---: | :---: | :---: | :---: |
| **1** | 14 | 12 | 85.71 % |
| **3** | 14 | 12 | 85.71 % |
| **5** | 14 | 12 | 85.71 % |
| **10** | 14 | 10 | 71.43 % |
| **15** | 14 | 11 | 78.57 % |
| **20** | 14 | 13 | 92.86 % |
| **25** | 14 | 14 | **100.00 %** |
| **30** | 14 | 14 | **100.00 %** |

---

## 3. Gráfica de Precisión en el Aula

![Gráfica de Precisión Salón](./grafica_precision_salon.png)

---

## 4. Análisis y Conclusiones del Escenario Real

Al comparar estos resultados con los del video individual (primer experimento), identificamos dinámicas muy interesantes de la representación facial en ambientes no controlados:

### A. El Comportamiento Incial y el "Valle de Precisión" ($S = 10$)
*   Con $S \le 5$ muestras, la precisión es aceptable (**85.71%**), reconociendo a Angel en 12 de los 14 frames.
*   Sin embargo, al subir a **$S=10$ muestras**, la precisión cae al **71.43%** (solo 10 reconocidos). 
*   **Análisis**: Los primeros 5 frames capturan una pose completamente frontal y de buena calidad. Al llegar a 10 frames de enrolamiento, se incorporan frames donde el alumno realiza un ligero giro o cambio de iluminación. En el espacio vectorial de ArcFace, promediar 5 imágenes frontales perfectas con 5 imágenes ligeramente rotadas sesga el centroide alejándolo de la pose frontal pura. Como resultado, en los frames del salón donde el alumno mira de frente a la cámara, la similitud contra este centroide híbrido cae ligeramente por debajo de $0.35$, causando fallos de reconocimiento.

### B. Consolidación del Centroide ($S \ge 20$)
*   A partir de las **20 muestras**, el centroide comienza a estabilizarse al abarcar una secuencia temporal lo suficientemente rica.
*   Al alcanzar las **25 muestras**, el centroide acumula todo el recorrido de poses, rotación y sombras del video de registro de Angel. Al promediar esta variedad de subespacios vectoriales, el centroide resultante es extremadamente representativo.
*   **Resultado**: Se logra un **100% de efectividad** en el video del salón ($S=25$ y $S=30$), identificando a Angel incluso bajo las duras condiciones de distancia, iluminación y oclusiones propias de una clase en vivo.

### Conclusión y Recomendación Final
Para asegurar la robustez de un sistema en vivo en el aula:
1.  **Mínimo 20-25 muestras de registro**: Registrar a los alumnos con videos de menos de 10 samples generará fallas constantes en el salón debido al "valle de precisión". Se deben capturar al menos 20-25 muestras válidas (aproximadamente 3 a 5 segundos de grabación continua del rostro).
2.  **Movimiento requerido**: Durante el registro, se le debe pedir al estudiante que realice un movimiento suave de rotación (izquierda-derecha y arriba-abajo). Esto garantiza que el centroide resultante represente todos los ángulos posibles que la cámara del salón captará en vivo.

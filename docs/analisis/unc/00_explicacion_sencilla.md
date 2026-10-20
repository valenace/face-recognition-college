# Explicación sencilla de lo que hacemos

**Para:** cualquier persona que quiera entender el proyecto desde cero.
**Fecha:** 2026-10-15 (actualizado 2026-10-20)
**Qué NO es:** no es un informe técnico. Es para entender qué hacemos, por qué, y qué sigue.

---

> [!IMPORTANT]
> **Actualización 2026-10-20 — cifras finales.** Este documento se escribió a mitad del trabajo. Los números definitivos están en `07_reporte_final_unc.md`. En resumen:
> - El umbral quedó en **0.30** (no 0.35), calibrado con la muestra etiquetada.
> - El filtro de tamaño quedó en **20 px** (no 40), la confianza mínima de detección en **0.75** y el ancla se revalida cada **30 frames**.
> - Resultado final: **9 de 9 estudiantes confirmados**, **0 errores de identidad**, **1 falso positivo de 157** desconocidos y **95 %** de caras con pista confirmadas dentro de ±0.5 s.
> - Los falsos positivos de anti-spoofing bajaron de 39 a 3 en el video completo.
> - Las etiquetas fueron revisadas por una segunda persona.

---

## 1. ¿Qué estamos construyendo?

Imagina que quieres que una cámara en un salón de clases **reconozca quién está sentado**, como cuando un profesor pasa lista pero sin tener que leer nombres.

Para eso hacemos dos cosas:

1. **Registro:** le enseñamos al sistema cómo se ve cada estudiante. Usamos un video corto de cada persona.
2. **Asistencia:** cuando la cámara mira el salón, el sistema compara cada cara que ve con los estudiantes que ya conoce, y decide quién es.

Todo esto ocurre en el computador, sin internet, con modelos de inteligencia artificial que ya existen.

---

## 2. Las palabras que vamos a usar

### Cara, rostro, detección
- **Cara o rostro:** la parte de la persona que vemos en la imagen.
- **Detección:** que el programa diga "aquí hay una cara" y dibuje una caja alrededor. Usamos un modelo llamado **RetinaFace**. Una caja así se llama **bounding box** o **bbox**: son cuatro números (esquina izquierda arriba, esquina derecha abajo).

### Frame
- **Frame:** una foto dentro de un video. Un video de 30 segundos a 30 frames por segundo tiene 900 frames.

### Recorte
- **Recorte:** una imagen pequeña que sacamos de un frame, solo con la cara. Son los archivos `.jpg` que guardamos.

### Alineación
- **Alineación:** girar y acomodar la cara para que los ojos queden horizontales y en la misma posición en todas las fotos. Así es más fácil compararlas.

### Embedding
- **Embedding:** convertir una cara en una lista de **512 números**. Es como una huella digital hecha de números. Dos fotos de la misma persona dan listas parecidas; dos personas distintas dan listas diferentes.
- Usamos un modelo llamado **ArcFace** para esto.

### Norma
- **Norma:** el "tamaño" de la lista de números. Hacemos que todas tengan tamaño 1, para que las comparaciones sean justas. Por eso verificamos que la norma sea **1.0**.

### Similitud coseno
- **Similitud coseno:** un número entre -1 y 1 que dice qué tan parecidas son dos listas de números.
  - Cerca de **1**: muy parecidas (probablemente la misma persona).
  - Cerca de **0**: no tienen nada que ver.
  - **Negativa**: en direcciones opuestas.

### Centroide
- **Centroide:** el promedio de las 50 listas de números de una persona. Es la "huella promedio" de esa persona. Lo guardamos en la base de datos.
- Cuando llega una cara nueva, la comparamos contra los 9 centroides y vemos cuál es el más parecido.

### Umbral
- **Umbral:** la línea que decide. Usamos **0.35**. Si la mejor coincidencia supera 0.35, decimos "es esta persona". Si no, decimos "Desconocido".
- El 0.35 es una **regla de cálculo heurística**: lo elegimos pensando, no lo medimos todavía. Calibrarlo es parte de lo que falta.

### Base de datos
- **Base de datos:** el archivo `data/database_embeddings.pkl`. Guarda los 9 centroides, uno por persona.

### Nitidez
- **Nitidez:** qué tan enfocada está una imagen. Medimos un número llamado **varianza del Laplaciano**. Más alto = más nítido.
- Cuidado: los valores dependen de dónde se mide. Por ahora medimos siempre sobre el **recorte alineado de 224 píxeles**, como en el registro.

### Brillo
- **Brillo:** qué tan claro u oscuro es el recorte. Promedio de la luminosidad de todos sus píxeles.

### CLAHE
- **CLAHE:** técnica para mejorar el contraste de una imagen, especialmente en zonas oscuras. Se aplica igual al registrar y al reconocer.

### Liveness (anti-spoofing)
- **Liveness:** verificar que la cara es de una persona real y no una foto o una pantalla. Por decisión del responsable, **está desactivado en el registro**. En la asistencia sí se usa.

### Tracking (seguimiento)
- **Tracking:** seguir a la misma persona de un frame a otro, dándole un número (ID). Usamos **ByteTrack**. Sirve para no tener que reconocer a alguien en cada frame.

### FAISS
- **FAISS:** una biblioteca que compara rápidamente una cara nueva contra muchos centroides. Con 9 personas no hace falta, pero funciona.

### IoU (Intersection over Union)
- **IoU:** mide qué tanto se solapan dos cajas.
  - Se calcula dividiendo el **área que comparten** entre el **área total que ocupan las dos**.
  - Va de 0 (no se tocan) a 1 (son idénticas).
  - Ejemplo: dos cajas casi iguales, desplazadas 10 píxeles, dan un IoU cercano a 0.8.
- Lo usamos para decir "esta caja de la cámara corresponde a esta cara etiquetada".

### Dispersión interna
- **Dispersión interna:** qué tan parecidas son las 50 fotos de una misma persona entre sí. Si son muy distintas, el centroide representa mal a esa persona.

### Muestra y etiqueta
- **Muestra:** un grupo de recortes que escogemos al azar para revisar.
- **Etiqueta:** la respuesta correcta que alguien escribe para cada recorte. Puede ser:
  - el nombre de una de las 9 personas,
  - `desconocido` (una cara clara de alguien que no está registrado),
  - `dudoso` (no se puede decidir con seguridad),
  - `no_cara` (la imagen no tiene una cara, por ejemplo una espalda).

### Ground truth (verdad de referencia)
- **Ground truth:** la respuesta correcta que usamos para medir si el sistema acierta. Nuestras etiquetas son una aproximación, no una verdad absoluta.

### Falso positivo y falso negativo
- **Falso positivo:** el sistema dice "es Ana" cuando en realidad es alguien que no conoce.
- **Falso negativo:** el sistema dice "Desconocido" cuando sí es Ana.

---

## 3. ¿Qué datos tenemos?

Los videos están en `data/unc/`:

- **`stds/`**: un video corto por cada estudiante, para registrarlo. Son **9 personas**: alexa, bob, carlos, dani, fran, jesus, jose, kate y kevin.
- **`salon/`**: videos del salón, donde los estudiantes aparecen sentados.
  - Usamos **solo** `VID_20260907_102851.mp4` (106 segundos). Este video muestra a los estudiantes sentados, saliendo, regresando, sentándose y mirando a la cámara.

---

## 4. Lo que ya hicimos, paso a paso

### Paso 1: Reinicio limpio
Borramos los registros anteriores que había en `data/` (que venían de un intento previo), **después de guardar una copia** en `data/_respaldo_2026-10-14/`. Así empezamos de cero, sin mezclar datos viejos con nuevos.

### Paso 2: Corregir el código
Antes de registrar, mejoramos el código del registro:
- Limpiar la carpeta de fotos antes de cada nuevo registro, para no mezclar fotos viejas.
- Aceptar solo frames con **una sola cara**, para no confundir a la persona con alguien del fondo.
- Repartir las fotos a lo largo de **todo** el video, no solo al principio.
- Exigir un **mínimo de 15 fotos**. Si no se llega, no se guarda nada.
- Usar el **mismo preprocesado** (CLAHE y alineación) al registrar y al reconocer, para que las comparaciones sean justas.
- Poder elegir la base de datos y la carpeta de fotos desde la línea de comandos.

### Paso 3: Registro de los 9 estudiantes
Cada estudiante se registró con **un solo video**, generando **50 fotos** y **un centroide**.

Cuando se registraron:
- `carlos` y `dani` los registró el responsable del proyecto.
- `alexa`, `bob`, `fran`, `jesus`, `jose`, `kate` y `kevin` se registraron uno por uno, revisando el resultado.

### Paso 4: Verificar el registro
Revisamos cada centroide:
- **Dimensión:** 512 números, como debe ser.
- **Norma:** 1.0 en los 9.
- **Fotos:** 50 en cada uno.

Luego comparamos los centroides entre sí. La similitud más alta entre dos personas distintas es **0.297** (`alexa` con `dani`). Está justo debajo de la alerta de 0.30. Esto quiere decir que esas dos personas son las más parecidas entre sí según el sistema, y hay que vigilarlas.

### Paso 5: Muestra del salón y etiquetas
Esta parte no estaba en el plan original, pero la agregamos para tener con qué medir:
1. Tomamos **1 frame por segundo** del video del salón (106 segundos).
2. Detectamos las caras en cada frame: salieron **1 127 recortes**.
3. Tomamos al azar **300** de esos recortes.
4. Alguien etiquetó cada uno. El responsable reemplazó las etiquetas automáticas que yo había puesto.

Resultado de las etiquetas en esos 300: 160 desconocidos, 44 dudosos, 17 de `dani`, 14 de `bob`, 13 `no_cara`, 13 `kate`, 12 `alexa`, 9 `jose`, 7 `kevin`, 4 `carlos`, 4 `jesus`, 3 `fran`.

**Dato importante:** las caras del salón miden en mediana **37 píxeles** de ancho. Son muy pequeñas. Por eso muchas etiquetas quedaron como `dudoso`, y las etiquetas no tienen segunda revisión.

### Paso 6: Demostración con un recorte
Para entender el proceso, seguimos **una sola cara** paso a paso (`v1_00020`, etiquetada como `dani`):
1. Leemos el frame en el segundo 1.0.
2. Detectamos **18 caras** en el frame completo (muchas son del fondo).
3. Identificamos la cara que corresponde al recorte.
4. Calculamos su embedding.
5. La comparamos con los 9 centroides: **dani 0.573**, alexa 0.189, carlos 0.089, y así sucesivamente.
6. Con el umbral 0.35, la decisión es **dani**, lo que coincide con la etiqueta.

Imagen anotada: `experiments/unc/salon_muestra/demo_v1_00020.jpg`.

Este caso fue fácil porque el margen entre el primero y el segundo lugar es grande (0.38). Los casos que importan son los que quedan cerca del umbral.

### Paso 7: Dispersión interna
Medimos qué tan parecidas son las 50 fotos de cada persona a su propio centroide:

| Persona | Media | Mínimo |
|---|---|---|
| kate | 0.807 | 0.706 |
| carlos | 0.775 | 0.568 |
| jose | 0.777 | 0.653 |
| bob | 0.757 | 0.617 |
| fran | 0.742 | 0.396 |
| alexa | 0.725 | 0.579 |
| kevin | 0.665 | 0.369 |
| dani | 0.692 | 0.543 |
| jesus | 0.644 | 0.139 |

Lo más importante:
- **`jesus`** tiene 4 fotos por debajo del umbral (0.139, 0.243, 0.255 y 0.335). Son consecutivas y muestran la cabeza muy inclinada hacia abajo. Esas fotos están en su centroide, así que lo desplazan.
- **`kevin`** tiene valores bajos (mínimo 0.369), aunque ninguna foto baja del umbral.
- **`fran`** tiene una foto en 0.396.

Gráfica: `experiments/unc/fase2_dispersion.png`.

### Paso 8: Documentar
Guardamos todo en:
- `docs/analisis/unc/02_resultados_registro.md`: resultados del registro, con tablas.
- `docs/analisis/unc/00_plan_experimentos_unc.md`: el plan, con el estado de cada fase.

---

## 5. ¿Qué es cada "fase" del plan?

El plan está dividido en fases. Así está el estado hoy:

| Fase | Qué hace | Estado |
|---|---|---|
| **Fase 0** | Muestra del salón y etiquetas | Hecha (agregada por nosotros) |
| **Fase 1** | Diagnóstico del dataset: qué tan bien detecta cada video | **Pendiente** |
| **Fase 2** | Registro de los 9 estudiantes | **Hecha** |
| **Fase 3** | Efecto de gorra y lentes | Pendiente |
| **Fase 4** | Encontrar el umbral correcto con una curva ROC | Pendiente |
| **Fase 5** | Mejorar el motor de asistencia | Pendiente |
| **Fase 6** | Evaluar en el salón | Pendiente |
| **Fase 7** | Reporte final | Pendiente |

**Por qué saltamos la Fase 1:** el plan decía que había que diagnosticar antes de registrar. Decidimos registrar primero y medir después. La Fase 1 sigue siendo necesaria para justificar los resultados, así que la haremos.

---

## 6. ¿Qué falta por hacer?

En orden de importancia:

1. **Decidir qué hacer con las fotos atípicas de `jesus`.** Hay que medir cuánto cambia su centroide si las quitamos, y decidir si las excluimos.
2. **Reconocer las caras del salón.** Tomar las 300 caras etiquetadas, pasar el reconocimiento y comparar con las etiquetas. Para emparejar cada etiqueta con la cara detectada, usar **IoU**.
3. **Revisar casos uno por uno.** Elegir unos cuantos casos útiles (cara grande de alguien registrado, cara grande de un desconocido, cara pequeña, personas parecidas) y mirar qué decide el sistema, paso por paso.
4. **Verificar accesorios** (lentes, gorra) en los videos de `carlos` y `dani`. Ya verificamos `jesus` (sin accesorios), `bob` (sin gorra) y `alexa` (sin lentes en el frame que revisamos).
5. **Corregir el documento del plan** para que refleje todo lo anterior. En particular, la observación de que `alexa` y `dani` usan lentes: solo `dani` lo confirmamos.
6. **Hacer la Fase 1 bien**, con el mismo criterio de nitidez (recorte alineado de 224 px).
7. **Calibrar el umbral** (Fase 4), en vez de dejarlo en 0.35 a ojo.

---

## 7. Lo que NO sabemos todavía

Es importante ser honestos:

- **No sabemos qué tan bien reconoce el sistema en el salón.** Hasta ahora solo medimos los centroides entre sí, y una demostración con un recorte.
- **El umbral 0.35 no está calibrado.** Es una elección inicial.
- **Las etiquetas del salón tienen limitaciones**: las caras son pequeñas y no hubo segunda revisión.
- **El liveness no está probado** contra fotos o pantallas.
- **No sabemos cuánto afectan los accesorios** (gorra, lentes) al reconocimiento.

---

## 8. Glosario rápido

| Palabra | Significado sencillo |
|---|---|
| Frame | Una foto dentro de un video |
| Recorte | Imagen pequeña con solo la cara |
| Bbox | Caja que marca dónde está la cara |
| Alineación | Girar la cara para que quede derecha |
| Embedding | Lista de 512 números que describe una cara |
| Norma | "Tamaño" de esa lista. Queremos que sea 1 |
| Similitud coseno | Número de -1 a 1. Más cerca de 1 = más parecidas |
| Centroide | Promedio de las listas de una persona |
| Umbral | Línea de decisión (0.35) |
| Base de datos | Archivo con los 9 centroides |
| Nitidez | Qué tan enfocada está una imagen |
| Brillo | Qué tan clara u oscura es una imagen |
| CLAHE | Mejora de contraste |
| Liveness | Comprobar que es una persona real y no una foto |
| Tracking | Seguir a la misma persona de un frame al siguiente |
| FAISS | Biblioteca para comparar rápido |
| IoU | Qué tanto se solapan dos cajas (0 a 1) |
| Dispersión interna | Qué tan parecidas son las fotos de una persona |
| Etiqueta | La respuesta correcta anotada a mano |
| Ground truth | La verdad de referencia para medir |
| Falso positivo | Dice que es alguien que no es |
| Falso negativo | No reconoce a alguien que sí es |
| Fase | Una etapa del plan de experimentos |

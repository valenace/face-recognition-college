# 04 — Curva ROC y umbral óptimo (Fase 4)

**Fecha:** 2026-10-19 · **AUC corregida:** 2026-11-05
**Datos:** 300 recortes etiquetados del video `VID_20260907_102851.mp4` (86 con nombre, 157 desconocidos; se excluyen `dudoso` y `no_cara`).
**Scripts:** `experiments/unc/fase4_similitudes.py` y `experiments/unc/fase4_curva_roc.py`.

---

## 1. Qué se midió

Para cada recorte se calculó su embedding con el mismo preprocesado del motor y la **similitud contra los 9 centroides**. Luego se separaron dos grupos:

- **Positivos (86):** la similitud con la persona que indica la etiqueta.
- **Negativos (243):** la mayor similitud con cualquier otra persona; para los desconocidos, la mayor contra los 9.

Se calculan dos lecturas:
1. **Score:** curva ROC y AUC usando solo similitudes.
2. **Operativa:** se simula la decisión real del motor (argmax + umbral) y se calculan precisión, recall y F1 por umbral.

## 2. Resultados

- **AUC (verificación, score-level): 0.9616.**
- **Average Precision (AP): 0.9535.**
- **EER: 0.0798** (umbral 0.239).
- **d' (discriminabilidad): 2.885.**
- **Youden:** umbral **0.281** (TPR 0.919, FPR 0.021).
- **Mejor F1 operativo:** umbral **0.28** (precisión 0.963, recall 0.919, F1 0.940).

> **Corrección del 2026-11-05.** La cifra anterior (**0.7439**) se calculó
> evaluando la ROC solo en umbrales 0.10–0.705. Esa rejilla no incluye los
> extremos: en 0.10 la FPR ya era 0.37 y en 0.705 la TPR era 0.07, así que el
> área quedaba truncada. La ROC exacta —un punto por cada score, desde `+inf`
> hasta el mínimo— da **AUC 0.9616**. La tabla operativa por umbral y la
> recomendación de 0.30 **no cambian**: solo se corrige el área y su
> interpretación. La comprobación por rangos (Mann-Whitney) coincide:
> 0.96162.

| Umbral | TP | FP | FN | TN | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.25 | 79 | 8 | 7 | 149 | 0.908 | 0.919 | 0.913 |
| **0.28** | **79** | **3** | **7** | **154** | **0.963** | **0.919** | **0.940** |
| 0.30 | 75 | 1 | 11 | 156 | 0.987 | 0.872 | 0.926 |
| **0.35 (actual)** | **71** | **0** | **15** | **157** | **1.000** | **0.826** | **0.904** |
| 0.40 | 61 | 0 | 25 | 157 | 1.000 | 0.709 | 0.830 |
| 0.45 | 50 | 0 | 36 | 157 | 1.000 | 0.581 | 0.735 |

> La tabla muestra 3 decimales; el valor exacto del mejor F1 es **0.9405** y el
> de EER **0.0798**. Las cifras de 4 decimales están en
> `experiments/unc/metricas/salida/identidad/scores.csv`.

## 3. Interpretación

- **El umbral actual (0.35) es demasiado alto.** Con él se reconocen 71 de 86 personas (82.6 %) sin ningún falso positivo. Bajando a 0.30 se reconocen 75 (87.2 %) con **un solo falso positivo**.
- **En 0.28 se gana un poco más de recall** (79 de 86) a cambio de 3 falsos positivos.
- **El AUC de 0.962 es alto.** En el 96.2 % de los pares (genuino, impostor) el
  genuino obtiene mayor similitud. El histograma muestra una cola de positivos
  desde 0.10 y negativos hasta 0.312; ese solapamiento explica los falsos
  positivos a umbrales bajos, pero la mayor parte de los negativos queda muy por
  debajo de los positivos. El EER de 8.0 % y la d' de 2.885 confirman una
  separación fuerte, no un modelo débil.
- **Esto es con decisiones de un solo frame.** El motor exige 3 votos consecutivos, lo que reduce aún más los falsos positivos. Un umbral algo más permisivo es razonable en la práctica.

## 4. Recomendación

**Umbral 0.30.** Precisión 0.987, recall 0.872, F1 0.926. Cuatro reconocimientos más que con 0.35, a cambio de 1 falso positivo en 157 desconocidos. Es prudente para asistencia, donde un falso positivo es peor que un falso negativo.

**Alternativa:** 0.28 si se prioriza el recall. F1 máximo, pero triplica los falsos positivos (3 en lugar de 1).

## 5. Limitaciones

- Las etiquetas son de una sola persona y con caras pequeñas (37 px de mediana).
- Un solo video y una sola fecha.
- Los recortes son de caras detectadas; no incluyen casos donde el detector no encontró cara.
- El umbral debe revalidarse con más datos (otras aulas, otras fechas) antes de fijarlo en producción.

## 6. Archivos

- `experiments/unc/fase4/fase4_similitudes.csv`: similitud de cada recorte contra los 9.
- `experiments/unc/fase4/fase4_tabla_umbrales.csv`: tabla completa por umbral.
- `experiments/unc/metricas/`: carpeta consolidada con ROC/PR exactas, AUC,
  AP, EER, d', matrices de confusión, métricas por estudiante y `resumen.md`.
- `fase4_histograma.png`, `fase4_curva_roc.png`, `fase4_f1_vs_umbral.png`, `fase4_curva_pr.png`.

---

## 7. Análisis con varios frames (2026-10-19)

**Actualización de etiquetas:** una segunda persona revisó las 300 etiquetas y las confirmó. Ya no aplica la limitación de "una sola persona".

Script: `experiments/unc/fase4_multiframe.py`. Para cada recorte se toman 3 frames consecutivos, se sigue la cara por IoU y se calcula su similitud con los 9 centroides. Se comparan dos reglas:

- **1 frame:** argmax + umbral.
- **3 frames:** se acepta solo si el mismo candidato gana y supera el umbral en los 3 frames consecutivos (la regla real de votos del motor).

| Umbral | 1 frame: P / R / F1 | TP / FP / FN | 3 frames: P / R / F1 | TP / FP / FN |
|---|---|---|---|---|
| 0.28 | 0.963 / 0.919 / 0.940 | 79 / 3 / 7 | 0.962 / 0.884 / **0.921** | 76 / 3 / 10 |
| 0.30 | 0.987 / 0.872 / 0.926 | 75 / 1 / 11 | 1.000 / 0.849 / **0.918** | 73 / 0 / 13 |
| 0.35 | 1.000 / 0.826 / 0.904 | 71 / 0 / 15 | 1.000 / 0.779 / 0.876 | 67 / 0 / 19 |

231 de 243 recortes tuvieron los 3 frames; el resto se perdió por cortes en el seguimiento o por estar al final del video.

**Lectura:**
- Exigir 3 frames **sube la precisión de 0.987 a 1.000 en 0.30** (elimina el único falso positivo), a cambio de 2 reconocimientos menos.
- En 0.28 los 3 falsos positivos **persisten** con 3 frames: son desconocidos que se parecen de forma sostenida a una persona, no ruido momentáneo.
- Con 3 frames, **0.30 y 0.28 quedan prácticamente empatados en F1** (0.918 contra 0.921), pero 0.30 tiene **cero falsos positivos**.

**Recomendación actualizada: umbral 0.30.** Con la regla de votos, da precisión perfecta y reconoce 73 de 86 caras (85 %). Diferencia con 0.28: 3 reconocimientos más, pero 3 personas no registradas aparecerían como presentes.

---

## 8. Decisión: umbral 0.30 (2026-10-19)

Se fija **`umbral_similitud = 0.30`** en `scripts/asistencia_unificada_optimizada.py`. Con la regla de 3 votos: precisión 1.000, recall 0.849, F1 0.918, 0 falsos positivos sobre la muestra.

## 9. Prueba de generalización en el segundo video (2026-10-19)

**Video:** `data/unc/salon/VID_20260907_105622.mp4` (50.6 s, estudiantes interactuando). Sin etiquetas, así que es una prueba **cualitativa** con revisión visual.

Corrida: `experiments/unc/demo/prueba_1910_1602_video2/`. Ajustes: umbral 0.30, filtro de tamaño 20 px, confianza 0.75, anti-spoofing omitido bajo 60 px, anclas revalidadas cada 30 frames.

**Asistencia confirmada (6 estudiantes):** `dani` (0.4472), `carlos` (0.5464), `fran` (0.3031), `jesus` (0.4019), `kevin` (0.4459), `jose` (0.3803).

**Capturas:** carlos 49, dani 47, kevin 15, fran 13, jose 5, jesus 4. No aparecen `alexa`, `bob` ni `kate`: puede ser que no estén en ese video o que no se reconocieran. No se puede afirmar sin etiquetas.

**Revisión visual inicial:** las cuadrículas de `dani` y `carlos` muestran la misma persona en todas las capturas. Pendiente la revisión del responsable en las demás.

**Nota:** `fran` pasó con 0.3031, muy cerca del umbral. Es el caso más frágil de esta corrida.

# Informe de prueba de liveness

**Fecha:** 2026-11-10 09:34:14 (reanálisis)  
**Fuente:** `data/unc/stds/val/video5158889925956863954.mp4` · **Umbral:** 0.9 · **Votos estables:** 3 · **Tamaño mínimo:** 60 px  
**Duración:** 9.3 s · **Frames:** 114 · **Con cara evaluada:** 114 · **Escenarios:** 1  
**Modelo:** MiniFASNet (uniface 3.3.0), el mismo del motor. **Identidad:** ArcFace + pgvector

> `score_real` es continuo: 1 = cara en vivo, 0 = foto. La decisión
> binaria replica el motor: `real` solo si el modelo confirma y
> `confidence >= 0.9`. `omitido` = cara menor de
> 60 px o perfil extremo; `sin_cara` = sin detección.

## 1. Resultados por escenario

| # | Escenario | GT | Duración | Frames | Con cara | Omitidos | Sin cara | Frames real | Frames foto | % real | Score mediana | Decisión | s voto estable | ID | Sim media |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | real | real | 9.27 | 114 | 114 | 0 | 0 | 114 | 0 | 100.0 % | 0.999 | real | 0.00 | val | 0.554 |

`% real` se calcula sobre frames con cara evaluada, no sobre el total.

> Si durante un ataque el operador queda en cuadro, el `ID` dominante del
> escenario puede ser el operador y no el estudiante mostrado. La tabla
> 2.1 separa ambos casos por identidad detectada.

## 2. Matriz y métricas

| Nivel | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Por frame (todos) | 114 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Por escenario (todos) | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Por sujeto mostrado** | 114 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

Positivo = cara real. TP: real aceptada; FN: real rechazada; FP: ataque aceptado; TN: ataque rechazado.

> **Por sujeto mostrado** es la lectura limpia: toma como positivos los
> frames de los escenarios `real` (operador: `val`) y como
> negativos solo los frames de los ataques cuya identidad detectada es el
> sujeto mostrado (la foto del estudiante), no el operador que quedó en
> cuadro. Los frames del operador dentro de un escenario de ataque se
> excluyen de esta matriz. Las dos primeras filas sí los incluyen, por eso
> pueden mostrar falsos aceptados que en realidad son caras reales.

La tabla de umbrales usa la regla del motor (`is_real` y `confidence >= t`),
no solo el score continuo. El AUC usa el score continuo.

> Sin frames de ataque: especificidad y FPR no son estimables; la
> accuracy de 1.000 solo refleja que todas las caras reales se
> aceptaron.

### 2.1 Decisión por identidad detectada

| ID detectada | Frames | Frames real | Frames foto | % foto | Score mediana | Segmentos |
|---|---|---|---|---|---|---|
| val | 114 | 114 | 0 | 0.0 % | 0.999 | 1 |

Esta tabla separa automáticamente a quien opera la cámara del estudiante
mostrado en el teléfono o papel: un ataque correcto aparece como
`ID del estudiante` con 100 % de frames `foto`.

## 3. Sensibilidad del umbral (por frame)

| Umbral | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 114 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.75 | 114 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.90 | 114 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

AUC del score continuo (crudo): **no estimable** (faltan frames reales o de ataque).

> No hubo frames de ataque en esta corrida: precision, especificidad y AUC no son estimables. Repite la prueba con papel y teléfono.

## 4. Lectura

- **Falsos aceptados (sujeto mostrado):** 0 frames.
- **Falsos rechazados (operador):** 0 frames.
- Del operador se acepta el 100.0 % de los frames (114/114).
- En la matriz cruda (todas las filas) aparecen 0 frames de ataque aceptados y 0 reales rechazados; esa matriz mezcla los frames del operador que quedaron en cuadro durante el ataque. La fila **Por sujeto mostrado** los excluye.


## 5. Identidad reconocida

La identidad es solo informativa: el anti-spoofing decide si la cara
está viva; ArcFace dice a quién se parece. En un ataque con la foto de
un estudiante, lo esperable es que la identidad coincida con ese
estudiante y la decisión sea `foto`.

| # | Escenario | Operador | Sujeto mostrado |
|---|---|---|---|
| 1 | real | val | — |

Los conteos por identidad y el score de liveness de cada una están en
la tabla 2.1 y en `por_identidad.csv`.

## 6. Capturas y archivos

- `liveness_frames.csv`: una fila por frame con la decisión y el score.
- `liveness_segmentos.csv`: resumen por escenario.
- `umbrales.csv`: precisión/recall/F1 en 0.50, 0.75 y 0.90.
- `umbrales_sujeto.csv`: lo mismo, restringido a operador vs sujeto
  mostrado (lectura limpia).
- `matriz_confusion.csv`: matrices por frame, por escenario y **por
  sujeto mostrado**.
- `por_identidad.csv`: decisión por identidad detectada (sección 2.1).
- `capturas/`: un frame anotado por segundo (si no se usó `--sin-capturas`).

## 7. Validez

- Es una prueba **controlada de demostración**, no una estimación
  poblacional: pocos escenarios, un operador y una sola sesión.
- **Matriz por sujeto mostrado:** asume que el operador es la identidad
  dominante de los segmentos `real` y que el sujeto mostrado es la
  identidad no-operador dominante de cada ataque. Si el ataque no se
  reconoce o el operador no aparece, esa fila queda vacía.
- Para que la matriz cruda del escenario sea limpia, el teléfono debe
  cubrir el cuadro; si el operador queda visible, usar la fila **Por
  sujeto mostrado**.
- El motor real exige **3 votos** y omite caras <60 px; aquí se reporta
  el frame suelto y la mayoría por escenario, que es más estricta.
- El brillo del teléfono, el reflejo de la pantalla y la distancia
  influyen en MiniFASNet. Repite cada escenario 2-3 veces y con luz
  estable antes de sacar conclusiones.
- La identidad depende de que el estudiante esté registrado en pgvector
  y de que la foto del teléfono conserve rasgos suficientes.

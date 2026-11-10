# Informe de prueba de liveness

**Fecha:** 2026-11-10 09:34:13 (reanálisis)  
**Fuente:** `camara:0` · **Umbral:** 0.9 · **Votos estables:** 3 · **Tamaño mínimo:** 60 px  
**Duración:** 116.5 s · **Frames:** 1247 · **Con cara evaluada:** 1247 · **Escenarios:** 2  
**Modelo:** MiniFASNet (uniface 3.3.0), el mismo del motor. **Identidad:** ArcFace + pgvector

> `score_real` es continuo: 1 = cara en vivo, 0 = foto. La decisión
> binaria replica el motor: `real` solo si el modelo confirma y
> `confidence >= 0.9`. `omitido` = cara menor de
> 60 px o perfil extremo; `sin_cara` = sin detección.

## 1. Resultados por escenario

| # | Escenario | GT | Duración | Frames | Con cara | Omitidos | Sin cara | Frames real | Frames foto | % real | Score mediana | Decisión | s voto estable | ID | Sim media |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | real | real | 54.08 | 617 | 617 | 0 | 0 | 617 | 0 | 100.0 % | 1.000 | real | 0.00 | val | 0.400 |
| 2 | telefono | ataque | 56.64 | 630 | 630 | 0 | 0 | 425 | 205 | 67.5 % | 0.985 | real | 0.00 | val | 0.420 |

`% real` se calcula sobre frames con cara evaluada, no sobre el total.

> Si durante un ataque el operador queda en cuadro, el `ID` dominante del
> escenario puede ser el operador y no el estudiante mostrado. La tabla
> 2.1 separa ambos casos por identidad detectada.

## 2. Matriz y métricas

| Nivel | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Por frame (todos) | 617 | 425 | 0 | 205 | 0.592 | 1.000 | 0.744 | 0.659 |
| Por escenario (todos) | 1 | 1 | 0 | 0 | 0.500 | 1.000 | 0.667 | 0.500 |
| **Por sujeto mostrado** | 617 | 0 | 0 | 119 | 1.000 | 1.000 | 1.000 | 1.000 |

Positivo = cara real. TP: real aceptada; FN: real rechazada; FP: ataque aceptado; TN: ataque rechazado.

> **Por sujeto mostrado** es la lectura limpia: toma como positivos los
> frames de los escenarios `real` (operador: `val`) y como
> negativos solo los frames de los ataques cuya identidad detectada es el
> sujeto mostrado (la foto del estudiante), no el operador que quedó en
> cuadro. Los frames del operador dentro de un escenario de ataque se
> excluyen de esta matriz. Las dos primeras filas sí los incluyen, por eso
> pueden mostrar falsos aceptados que en realidad son caras reales.
>
> Sujetos mostrados: segmento 2 → `kevin`.

La tabla de umbrales usa la regla del motor (`is_real` y `confidence >= t`),
no solo el score continuo. El AUC usa el score continuo.

### 2.1 Decisión por identidad detectada

| ID detectada | Frames | Frames real | Frames foto | % foto | Score mediana | Segmentos |
|---|---|---|---|---|---|---|
| val | 1126 | 1041 | 85 | 7.5 % | 0.999 | 1 2 |
| kevin | 119 | 0 | 119 | 100.0 % | 0.000 | 2 |
| bob | 1 | 0 | 1 | 100.0 % | 0.308 | 2 |
| dani | 1 | 1 | 0 | 0.0 % | 0.976 | 2 |

Esta tabla separa automáticamente a quien opera la cámara del estudiante
mostrado en el teléfono o papel: un ataque correcto aparece como
`ID del estudiante` con 100 % de frames `foto`.

## 3. Sensibilidad del umbral (por frame)

| Umbral | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 617 | 503 | 0 | 127 | 0.551 | 1.000 | 0.710 | 0.597 |
| 0.75 | 617 | 484 | 0 | 146 | 0.560 | 1.000 | 0.718 | 0.612 |
| 0.90 | 617 | 425 | 0 | 205 | 0.592 | 1.000 | 0.744 | 0.659 |

AUC del score continuo (crudo, reales vs todos los frames de ataque): **0.8806** (617 reales, 630 de ataque).

### 3.1 Sensibilidad por sujeto mostrado

| Umbral | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 617 | 0 | 0 | 119 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.75 | 617 | 0 | 0 | 119 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.90 | 617 | 0 | 0 | 119 | 1.000 | 1.000 | 1.000 | 1.000 |

AUC del score continuo (por sujeto mostrado): **1.0000** (617 reales, 119 del sujeto mostrado).

## 4. Lectura

- **Falsos aceptados (sujeto mostrado):** 0 frames.
- **Falsos rechazados (operador):** 0 frames.
- Del sujeto mostrado se rechaza el 100.0 % de los frames (119/119).
- Del operador se acepta el 100.0 % de los frames (617/617).
- En la matriz cruda (todas las filas) aparecen 425 frames de ataque aceptados y 0 reales rechazados; esa matriz mezcla los frames del operador que quedaron en cuadro durante el ataque. La fila **Por sujeto mostrado** los excluye.

La decisión por escenario (mayoría de frames) es la que se acerca al uso real con votos; la tabla de umbrales muestra el compromiso precisión/recall si se mueve el corte.

## 5. Identidad reconocida

La identidad es solo informativa: el anti-spoofing decide si la cara
está viva; ArcFace dice a quién se parece. En un ataque con la foto de
un estudiante, lo esperable es que la identidad coincida con ese
estudiante y la decisión sea `foto`.

| # | Escenario | Operador | Sujeto mostrado |
|---|---|---|---|
| 1 | real | val | — |
| 2 | telefono | val | kevin |

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

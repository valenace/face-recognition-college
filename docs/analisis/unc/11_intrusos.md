# 11 — Intrusos: separar desconocidos reales de registrados perdidos

Fecha: 2026-10-23
Estado: implementado y medido

---

## 1. El problema

El motor devuelve `DESCONOCIDO` en **cada frame** donde la similitud no alcanza el
umbral 0.30. Un estudiante registrado que se mueve, gira o queda pequeño produce
muchos de esos frames, y ByteTrack los reparte en **pistas distintas**.

La capa de escritura creaba **una fila `INTRUSO` por pista**. Resultado: sesiones
llenas de intrusos que en realidad eran los mismos estudiantes. En la sesión 94
(13 minutos de captura) había **24 filas de intruso** para 9 estudiantes, y en
44 segundos del video 1 —donde **no hay ningún desconocido**— el motor generó
**130 pistas desconocidas**.

El profesor quedaba obligado a identificar manualmente a gente que ya estaba
registrada.

---

## 2. Qué se midió

### Video 1 (todos registrados, 44 s, 1 319 frames)

| Métrica | Valor |
|---|---|
| Pistas confirmadas | 25 |
| Pistas con al menos un frame desconocido | **130** |
| Nunca confirmadas | 111 |
| Persistentes (≥ 15 frames seguidos) | 53 |

Mejor similitud de las 53 persistentes nunca confirmadas:

| Rango | Pistas |
|---|---|
| 0.10 – 0.15 | 2 |
| 0.15 – 0.18 | 7 |
| 0.18 – 0.20 | 2 |
| 0.20 – 0.22 | 9 |
| 0.22 – 0.25 | 11 |
| 0.25 – 0.28 | 9 |
| 0.28 – 0.31 | 11 |

### Video 3 (8 registrados presentes + 21 desconocidos, 4 partes)

| Métrica | Valor |
|---|---|
| Pistas desconocidas nunca confirmadas | 118 |
| Persistentes con mejor score < 0.18 | **18** |
| Persistentes con mejor score < 0.20 | 26 |
| Persistentes con mejor score < 0.22 | 43 |
| Persistentes con mejor score < 0.25 | 63 |

El 97 % de las decisiones desconocidas del video 3 está por debajo de 0.25. Es
decir: **mirar el frame suelto no separa nada**; lo que separa es el **mejor score
de toda la pista**.

---

## 3. La regla

Se aplica en `core/intrusos.py::SeguimientoIntrusos`, no en el motor.

1. Se sigue el **mejor score** de cada pista (`best`).
2. Si `best >= 0.18`, la pista se parece demasiado a un registrado: **no es
   intruso**. Si ya se había creado una fila, se borra (y su captura).
3. Una pista que perdió un **conflicto de identidad** nunca es intruso.
4. Se exigen **15 frames consecutivos** de la misma pista para crear la fila.
5. La nota de la fila guarda la **mejor coincidencia**:
   `Persona no registrada. Pista 12. Mejor coincidencia: dani (0.14).`

El umbral **0.18** se eligió con los dos videos:

| Corte | Falsos en video 1 (todos registrados) | Detectados en video 3 (21 desconocidos) |
|---|---|---|
| 0.15 | 2 | ~14–16 |
| **0.18** | **9** | **18** |
| 0.20 | 11 | 26 |
| 0.25 | 31 | 63 |

`0.18` reduce los falsos del video 1 de 130 pistas a 9 filas y conserva 18 de los
21 desconocidos del video 3.

---

## 4. Resultado medido

Mismo tramo del video 1 (36–80 s), con la regla nueva:

| Concepto | Antes | Después |
|---|---|---|
| Filas `INTRUSO` | 24 (sesión de 13 min) | **6** (44 s) |
| Pistas desconocidas | 130 | 6 filas finales |
| Estudiantes confirmados | 8 | 8 |
| Spoofing | 2 | 1 |

Las 8 confirmaciones del tramo se mantienen: la regla **no toca la asistencia**,
solo lo que se escribe como intruso.

---

## 5. Límites y trabajo futuro

- Quedan **9 falsos** en 44 s del video 1: son estudiantes cuya pista nunca se
  acerca a 0.18 (caras pequeñas u ocluidas). Sin re-identificación no se pueden
  separar de un desconocido real.
- En el video 3 se pierden **3 de 21** desconocidos por el corte.
- El profesor puede **descartar** una fila desde `/sesiones/<id>/` (marca
  `alerta_revisada`), para limpiar los falsos que queden.
- La solución definitiva sigue siendo **multi-prototipo** (top-3 sobre las 50
  fotos), fuera del alcance acordado.

---

## 6. Archivos

| Archivo | Cambio |
|---|---|
| `core/intrusos.py` | Clasificador puro (sin Django) y umbrales |
| `core/registro_asistencia.py` | `aplicar_decisiones` usa el clasificador; nota con la mejor coincidencia |
| `core/servicio_en_vivo.py`, `ver_flujo.py` | Crean el `SeguimientoIntrusos` por captura |
| `apps/attendance/views.py` | Vista de sesiones y acción de descartar |
| `tests/test_registro.py` | 6 pruebas unitarias del clasificador |

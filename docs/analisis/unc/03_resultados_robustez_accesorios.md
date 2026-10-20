# 03 — Robustez a accesorios: gorra y lentes (Fase 3)

**Fecha:** 2026-10-20
**Estado:** Cerrada con limitaciones.
**Datos:** 86 recortes con nombre del video `VID_20260907_102851.mp4`, con anotación de accesorios confirmada por el responsable (`experiments/unc/salon_muestra/accesorios_muestra.csv`).

---

## 1. Protocolo adaptado

El plan original de la Fase 3 proponía tres condiciones (A: sin accesorio en registro y evaluación; B: con accesorio solo en evaluación; C: registro multi-video). **No es posible con el esquema actual**, porque cada estudiante se registra con **un solo video**.

Protocolo que sí se puede hacer:

1. Cada recorte etiquetado tiene su anotación de `lentes` y `gorra`.
2. Se calcula la similitud de cada recorte **con su propio centroide** (`fase4_similitudes.csv`).
3. Se comparan las distribuciones por condición de accesorio.
4. Se mide cuántos superan los umbrales candidatos 0.28, 0.30 y 0.35.

Grupos: **sin accesorios** 44, **solo lentes** 35, **solo gorra** 3 (todas de `fran`), **gorra y lentes** 4 (todas de `jesus`).

## 2. Resultados

| Condición | n | Media | Mediana | Mínimo | Máximo | ≥ 0.28 | ≥ 0.30 | ≥ 0.35 |
|---|---|---|---|---|---|---|---|---|
| Sin accesorios | 44 | 0.464 | 0.474 | 0.102 | 0.758 | 39 | 37 | 36 |
| Solo lentes | 35 | 0.481 | 0.496 | 0.202 | 0.621 | 34 | 33 | 31 |
| Solo gorra (`fran`) | 3 | 0.539 | 0.582 | 0.451 | 0.585 | 3 | 3 | 3 |
| Gorra y lentes (`jesus`) | 4 | 0.285 | **0.315** | 0.120 | 0.391 | 3 | **2** | 1 |

## 3. Interpretación

**Los lentes no afectan el reconocimiento.** La mediana con lentes (0.496) es incluso algo mayor que sin accesorios (0.474). La prueba de Mann-Whitney no encuentra diferencia: **p = 0.512**.

**La gorra sola tampoco.** Los 3 recortes de `fran` con gorra tienen similitud alta (0.45 a 0.59) y pasan cualquier umbral. Es coherente con la observación de la sección 15: `fran` con gorra acertó en 3 de 3.

**La combinación gorra y lentes sí afecta.** Los 4 recortes de `jesus` con ambos accesorios bajan su similitud (mediana 0.315, frente a 0.474 sin accesorios). El peor caso es 0.120.

**Con el umbral elegido (0.30), 2 de esos 4 recortes se reconocen.** Con el umbral anterior (0.35) solo se reconocía 1 de 4. Bajar el umbral a 0.30 recupera la mitad de los casos con doble accesorio, sin falsos positivos en la muestra.

## 4. Limitaciones

- **Solo 7 recortes con gorra**, y todos de 2 personas (`fran`, `jesus`).
- Los 4 de `jesus` son del **mismo día y el mismo video**, y además están en perfil. **No se puede separar el efecto del accesorio del efecto de la pose**: en la sección 12 se mostró que el perfil por sí solo explica el 43 % de los fallos.
- No hay condición A/B controlada (la misma persona con y sin accesorio en el mismo momento).
- La similitud se mide sobre recortes de caras detectadas; no incluye casos sin detección.
- Registro con un solo video por persona, por decisión del proyecto.

## 5. Conclusión

- **Lentes: sin impacto medible.**
- **Gorra sola: sin impacto medible** (evidencia limitada a 3 casos).
- **Gorra y lentes juntos: sí hay impacto**, con similitud mediana 0.315 y un mínimo de 0.120. Con el umbral 0.30 se reconocen 2 de 4 casos; con 0.35, 1 de 4. El margen es frágil.

**Recomendación para el futuro:** si en producción los estudiantes pueden llevar gorra y lentes, conviene registrar **al menos un video adicional con esos accesorios** (el esquema multi-video que el plan original llamaba condición C). Hoy no está permitido por la decisión de usar un solo video por estudiante, así que queda como trabajo futuro.

## 6. Archivos

- `experiments/unc/salon_muestra/accesorios_muestra.csv`: anotación por recorte.
- `experiments/unc/fase4/fase4_similitudes.csv`: similitud de cada recorte contra los 9.
- Secciones relacionadas: `02_resultados_registro.md` §12 (pose) y §15 (accesorios y fallos).

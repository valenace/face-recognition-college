# Diagnóstico Inicial del Pipeline de Reconocimiento Facial
**Proyecto:** CV College — Sistema de Asistencia Biométrica  
**Fecha:** 2026-09-01  
**Autor del análisis:** Antigravity (IA) + Valentina  
**Estado:** ✅ Completado

---

## 1. Inventario Completo del Dataset

### 1.1 Videos de Registro (`data/dataset_cv/videos/`)

| Persona | Archivo(s) | Frames | FPS | Duración | Resolución |
|---|---|---|---|---|---|
| angel | `angel.mp4` | 331 | 30.0 | 11.0 s | 464×848 |
| cristobal | `cristobal.mp4` | 447 | 29.9 | 14.9 s | 720×1280 |
| gabriela | `gabriela_perez.mp4` + `gabriela_perez_1.mp4` | 133+127 | ~24.4 | 5.3+5.4 s | 720×1280 |
| hanny | `hanny.mp4` | 179 | 30.0 | 6.0 s | 720×1280 |
| hector | `hector_pineda.mp4` | 186 | 23.7 | 7.9 s | 720×1280 |
| jorge | `jorge.mp4` | 483 | 28.7 | 16.8 s | 360×640 |
| lino | `lino.mp4` | 978 | 30.0 | 32.6 s | 720×1280 |
| maria | `maria_paula.mp4` | 195 | 30.1 | 6.5 s | 720×1280 |
| ronald | `ronald_delgado.mp4` + `2ronald_delgado.mp4` | 214+206 | 30.0 | 7.1+6.9 s | 720×1280 |

> **Nota**: `lino` tiene el video más largo (32.6 s, 978 frames), lo que explica por qué sus
> similitudes en el CSV son las más altas (0.62–0.63). Más diversidad de poses → centroide más representativo.

### 1.2 Frames Extraídos (`data/dataset_cv/frames/`)

Tras aplicar el filtro de `registro_video.py` (diversidad de pose, blur, brillo):

| Persona | Frames en DB | ¿Suficiente? | Riesgo |
|---|---|---|---|
| angel | 37 | ✅ | Bajo |
| cristobal | 50 | ✅ | Bajo |
| gabriela | 50 | ✅ | Bajo |
| **hanny** | **11** | ❌ | **ALTO** |
| hector | 23 | ⚠️ | Medio |
| jorge | 50 | ✅ | Bajo |
| lino | 50 | ✅ | Bajo |
| **maria** | **22** | ⚠️ | Medio |
| ronald | 50 | ✅ | Bajo |

> **¿Por qué hanny tiene solo 11 frames si su video tiene 179 frames?**
> El filtro de diversidad de pose (umbral > 0.95) descartó la mayoría porque la persona no movió la
> cabeza durante el registro. El video de 6 s tenía poca variedad de poses. El script funciona bien
> — el problema es el protocolo de grabación.

### 1.3 Videos del Salón (`data/salon/`)

| Archivo | Tamaño | Observaciones |
|---|---|---|
| `20260806_105629.mp4` | 83 MB | Sesión 06-Aug, toma corta |
| `20260806_105645.mp4` | 411 MB | **Video principal** — más largo |
| `20260806_105847.mp4` | 185 MB | Sesión 06-Aug, toma media |
| `20260806_105921.mp4` | 24 MB | Sesión 06-Aug, toma muy corta |
| `20260820_140625.mp4` | 41 MB | Sesión 20-Aug |
| `20260820_140734.mp4` | 38 MB | Sesión 20-Aug |
| `20260820_140914.mp4` | 31 MB | Sesión 20-Aug |

---

## 2. Estado de la Base de Datos de Embeddings

### 2.1 Verificación de la DB

- **Archivo:** `data/database_embeddings.pkl`
- **Personas:** 9 (`maria, angel, hanny, gabriela, hector, cristobal, lino, ronald, jorge`)
- **Dimensión de cada vector:** 512 (ArcFace estándar)
- **Norma de cada vector:** 1.0000 exacta ✅ (correctamente normalizados)
- **Formato:** Un único centroide por persona

### 2.2 Similitudes Observadas en el CSV de Asistencia Real

Del archivo `data/asistencia.csv` (sesiones del 07-Aug al 21-Ago):

| Persona | Similitudes registradas | Observación |
|---|---|---|
| angel | 0.3717 – 0.4444 | Rango aceptable |
| maria | 0.3747 – 0.4018 | Cerca del límite |
| hanny | 0.4006 (constante) | Sospechosamente fijo — posible anclaje único |
| gabriela | 0.3557 – 0.6081 | Rango amplio — buena diversidad de poses capturadas |
| hector | 0.3896 – 0.4319 | Aceptable |
| cristobal | 0.3807 – 0.4526 | Aceptable |
| **ronald** | **0.3643 – 0.3797** | **MUY BAJO — riesgo de fallo** |
| lino | 0.6130 – 0.6262 | **Excelente — mejor registrado** |
| jorge | 0.4119 | Solo un registro disponible |

---

## 3. Matriz de Similitudes Entre Centroides (Inter-Clase)

Un valor alto entre dos personas distintas indicaría riesgo de confusión.

| Par | Similitud | Riesgo |
|---|---|---|
| hector ↔ jorge | 0.2798 | ⚠️ El más alto del dataset |
| angel ↔ lino | 0.2562 | ⚠️ Vigilar |
| hanny ↔ lino | 0.2451 | ⚠️ Vigilar |
| lino ↔ ronald | 0.2090 | 🟡 Monitorear |
| hector ↔ cristobal | 0.1930 | ✅ Aceptable |
| hanny ↔ gabriela | 0.2142 | ✅ Aceptable |
| *todos los demás* | < 0.18 | ✅ Bien separados |

**Conclusión**: Ningún par supera 0.30. El sistema tiene buena separabilidad natural.

---

## 4. Análisis del Umbral Actual (0.35)

### El problema

```
Similitud máxima entre pares distintos: 0.280 (hector-jorge)
Umbral actual:                          0.350
Margen de seguridad =                   0.070
```

Este margen parece suficiente, pero es riesgoso porque:
- Las similitudes de personas registradas **caen** en el salón (distancia, ángulo, iluminación)
- Las similitudes de personas NO registradas pueden **subir** accidentalmente por textura y pose similar

**Umbral propuesto preliminar (antes de la Curva ROC):** `0.42`
- El 80% de los reconocimientos exitosos están por encima de 0.38
- La media de reconocimientos exitosos es ~0.42
- Será validado formalmente en Fase 3

---

## 5. Discusión: ¿Centroide Único vs. Galería Completa?

**Veredicto: Mantener el centroide.** Pero optimizar el protocolo de grabación.

### ¿Por qué el centroide es suficiente?

1. `registro_video.py` ya garantiza diversidad al descartar frames con similitud > 0.95
2. El centroide de 25-50 poses diversas es el **punto de máxima verosimilitud** de la distribución
3. La galería completa no agrega precisión si el registro ya es diverso, pero sí agrega latencia

### ¿Cuándo sería útil la galería?

Solo si el centroide "promedia demasiado" y se aleja de todas las poses posibles. Esto ocurre con:
- Pocas muestras (<10)
- Poses muy extremas mezcladas con frontales

### Protocolo de Registro Óptimo (10 segundos)

```
1. Mirar al frente           →  3 segundos (captura 5-8 frames frontales)
2. Girar izquierda ~30°     →  2 segundos (captura ángulo lateral izq.)
3. Girar derecha ~30°       →  2 segundos (captura ángulo lateral der.)
4. Inclinar arriba y abajo  →  2 segundos (captura variación vertical)
5. Volver al frente         →  1 segundo  (frame final de calidad)
```

Con este protocolo el filtro automático extrae los 25-50 mejores frames.

---

## 6. Hallazgos Críticos y Plan de Acción

| # | Hallazgo | Severidad | Acción |
|---|---|---|---|
| 1 | `hanny` tiene 11 frames → centroide inestable | 🔴 Alta | Re-registrar con protocolo de movimiento |
| 2 | `hector` tiene 23 frames → límite inferior | 🟡 Media | Re-registrar si hay tiempo |
| 3 | `ronald` similitudes 0.36–0.38 → rozando umbral | 🟡 Media | Re-registrar con video más diverso |
| 4 | Umbral 0.35 sin validación empírica | 🟡 Media | Ejecutar Fase 3 (Curva ROC) |
| 5 | No existe verificación de margen Top-2 | 🟡 Media | Implementar en Fase 4 |
| 6 | `test_asistencia_salon.py` importa versión no optimizada | 🟢 Baja | Corregir import |
| 7 | Personas en salón no registradas en DB | 🟢 Info | Documentar para análisis FPR |

---

## 7. Próximos Pasos

- [ ] **Fase 2**: Re-registrar a `hanny` y `hector` usando ambos videos disponibles
- [ ] **Fase 3**: Calcular Curva ROC empírica para umbral óptimo
- [ ] **Fase 4**: Implementar Top-2 margin check y logging detallado
- [ ] **Fase 5**: Evaluación cuantitativa en todos los videos del salón

---

*Documento vivo — actualizar tras completar cada fase.*

# Documentación de Análisis — CV College Face Recognition

Este directorio contiene toda la documentación técnica, metodológica, arquitectónica y experimental del
pipeline de reconocimiento facial para el sistema de asistencia universitaria.

## Orden de Lectura Recomendado

| # | Archivo | Descripción | Estado |
|---|---|---|---|
| 0 | [`00_guia_metodologica.md`](./00_guia_metodologica.md) | Cómo funcionan los análisis — de conceptos básicos a avanzados | ✅ Listo |
| 1 | [`01_diagnostico_inicial.md`](./01_diagnostico_inicial.md) | Estado actual del sistema: dataset, DB, similitudes, hallazgos | ✅ Listo |
| 2 | [`02_plan_experimentos.md`](./02_plan_experimentos.md) | Plan detallado de todas las fases y experimentos | ✅ Listo |
| 3 | [`03_descripcion_proyecto_librerias_modelos.md`](./03_descripcion_proyecto_librerias_modelos.md) | Qué hace el proyecto, modelos neuronales (RetinaFace, ArcFace, etc.) y todas las librerías | ✅ Listo |
| 4 | `04_resultados_reregistro.md` | Resultados de re-registrar personas con pocos frames (Fase 2) | ⏳ Pendiente |
| 5 | `05_resultados_curva_roc.md` | Curva ROC, umbral óptimo, distribución de similitudes (Fase 3) | ⏳ Pendiente |
| 6 | `06_resultados_mejoras_motor.md` | Métricas antes/después de aplicar mejoras al motor (Fase 4) | ⏳ Pendiente |
| 7 | `07_resultados_evaluacion_completa.md` | Evaluación en los 7 videos del salón (Fase 5) | ⏳ Pendiente |
| 8 | `08_reporte_final_pipeline.md` | Reporte técnico final consolidado (Fase 6) | ⏳ Pendiente |

## ¿Dónde están los scripts de los experimentos?

En la carpeta [`experiments/`](../../experiments/) del proyecto raíz.

## ¿Dónde están las gráficas?

En la carpeta [`resultados_experimentales/`](../../resultados_experimentales/) del proyecto raíz.

# Documentación de Análisis — CV College Face Recognition

## Documentación vigente (dataset UNC)

Los análisis del proyecto actual están en [`unc/`](./unc/):

| # | Archivo | Contenido |
|---|---|---|
| 0 | [`unc/00_explicacion_sencilla.md`](./unc/00_explicacion_sencilla.md) | Explicación del sistema en lenguaje sencillo |
| 0 | [`unc/00_plan_experimentos_unc.md`](./unc/00_plan_experimentos_unc.md) | Plan y estado de las fases 0–8 |
| 1 | [`unc/01_diagnostico_dataset_unc.md`](./unc/01_diagnostico_dataset_unc.md) | Diagnóstico del dataset UNC |
| 2 | [`unc/02_resultados_registro.md`](./unc/02_resultados_registro.md) | Registro de los 9 estudiantes |
| 3 | [`unc/03_bitacora_calculos.md`](./unc/03_bitacora_calculos.md) | Bitácora de cada cálculo con fórmula y resultado |
| 3b | [`unc/03_resultados_robustez_accesorios.md`](./unc/03_resultados_robustez_accesorios.md) | Robustez a lentes, gorra y gorra+lentes |
| 4 | [`unc/04_resultados_curva_roc.md`](./unc/04_resultados_curva_roc.md) | ROC, AUC, umbral y métricas por umbral |
| 6 | [`unc/06_resultados_evaluacion_salon.md`](./unc/06_resultados_evaluacion_salon.md) | Evaluación en el salón |
| 7 | [`unc/07_reporte_final_unc.md`](./unc/07_reporte_final_unc.md) | Reporte consolidado (videos 1 y 2) |
| 8 | [`unc/08_resultados_video3.md`](./unc/08_resultados_video3.md) | Revalidación en el video 3 |
| 9 | [`unc/09_incidente_identidad_dani.md`](./unc/09_incidente_identidad_dani.md) | Incidente de identidad y corrección |
| 10 | [`unc/10_plan_mejora_motor.md`](./unc/10_plan_mejora_motor.md) | Plan de mejora y fases rechazadas con datos |
| 11 | [`unc/11_intrusos.md`](./unc/11_intrusos.md) | Separación de desconocidos reales y registrados |

Las métricas consolidadas para la tesis están en
[`experiments/unc/metricas/`](../../experiments/unc/metricas/), con su
`salida/resumen.md`. La auditoría independiente de esos resultados está en
[`reporte_auditoria_resultados.md`](../../reporte_auditoria_resultados.md).

## Referencia metodológica y matemática

| Archivo | Contenido |
|---|---|
| [`00_guia_metodologica.md`](./00_guia_metodologica.md) | Cómo se plantean los análisis |
| [`03_descripcion_proyecto_librerias_modelos.md`](./03_descripcion_proyecto_librerias_modelos.md) | Modelos (RetinaFace, ArcFace, MiniFASNet) y librerías |
| [`04_fundamentos_matematicos_pipeline.md`](./04_fundamentos_matematicos_pipeline.md) | Fundamentos matemáticos del pipeline |
| [`05_temario_matematico_completo.md`](./05_temario_matematico_completo.md) | Temario matemático completo |

## Histórico (dataset anterior)

[`01_diagnostico_inicial.md`](./01_diagnostico_inicial.md) y
[`02_plan_experimentos.md`](./02_plan_experimentos.md) corresponden al dataset
anterior (hanny/lino/ronald). Se conservan como referencia; el plan vigente es
`unc/00_plan_experimentos_unc.md`.

## Scripts y resultados

- `experiments/unc/`: scripts de las fases y la regresión de identidad.
- `experiments/unc/metricas/`: script maestro, README y `salida/` con CSV y figuras.
- `experiments/unc/salida_liveness/`: corridas de la prueba de liveness.
- `reporte_auditoria_resultados.md`: auditoría de los resultados publicados.

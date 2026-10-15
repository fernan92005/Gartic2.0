# Diagnóstico de falsas alarmas

Las cifras 39 FP por matices, 70 FP sin sospecha concreta y 2 FN de sobregeneralización son afirmaciones del encargo. No hay un informe/manifiesto con esos IDs en el checkout: fase4_diagnostico corresponde a joseca_v1. No se atribuyen sus errores a v3.

Se reproduce el protocolo original de v3: cinco folds por ID base, semillas 0,1,2, parámetros congelados y corte publicado 0,16. Se guarda cada predicción en oof_protocolo_original.json. El corte original optimizado sobre esos mismos OOF es retrospectivo y no es el criterio principal de comparación.

| Semilla original | FP H | FP H con matiz | FP H sin sospecha y NLI citado<0,3 | FN H sobregeneralización |
|---|---:|---:|---:|---:|
| 0 | 147 | 41 | 48 | 3 |
| 1 | 140 | 40 | 45 | 3 |
| 2 | 146 | 39 | 45 | 4 |

Las cohortes son operacionales, no causas anotadas. Matiz = cualquier matiz_perdido_* positivo. Sin sospecha = todas las oraciones MiniMax respaldadas, ninguna sospecha numérica/entidad/nombre/fecha/hora/matiz/alcance/causa y respaldo NLI citado mínimo<0,3; la definición exacta está en cohort_masks(). Los patrones pueden solaparse con síntesis y no se suman como errores distintos. No se cambian filtros para forzar 39 o 70.

## Cohortes de B0 en el protocolo nuevo

| Variante | FP matiz B0 corregidos | FP sin sospecha B0 corregidos | FP multifuente B0 corregidos | FP H nuevos | FN H nuevos |
|---|---:|---:|---:|---:|---:|
| B0 | 0 / 32 | 0 / 42 | 0 / 53 | 0 | 0 |
| B1 | 1 / 32 | 3 / 42 | 1 / 53 | 2 | 1 |
| B2 | 2 / 32 | 4 / 42 | 1 / 53 | 10 | 2 |
| B3 | 7 / 32 | 9 / 42 | 8 / 53 | 4 | 6 |
| B4 | 0 / 32 | 4 / 42 | 1 / 53 | 2 | 1 |
| B5 | 3 / 32 | 6 / 42 | 2 / 53 | 9 | 1 |
| B6 | 8 / 32 | 10 / 42 | 8 / 53 | 7 | 7 |
| B7 | 8 / 32 | 10 / 42 | 8 / 53 | 8 | 4 |
| B1_norm | 1 / 32 | 3 / 42 | 1 / 53 | 2 | 2 |
| B0_sin_NLI | 2 / 32 | 18 / 42 | 10 / 53 | 4 | 13 |
| B3_sin_NLI | 1 / 32 | 11 / 42 | 6 / 53 | 32 | 12 |
| B5_sustituye_matices | 10 / 32 | 3 / 42 | 4 / 53 | 11 | 7 |

## Seguimiento de la cohorte histórica reconstruida (semilla 0)

Cambiar de particiones cambia modelos y errores. Solo se puede hablar de corregir un FP si B0 vuelve a fallar en ese mismo caso en los folds nuevos. Se informa de la intersección, sin presentar los denominadores 39/70 como confirmados.

| Cohorte histórica | Tamaño original reconstruido | Presente en desarrollo | Sigue siendo FP B0 nuevo |
|---|---:|---:|---:|
| matices | 41 | 32 | 29 |
| sin_sospecha | 48 | 34 | 30 |

| Variante | FP históricos matiz, confirmados por B0 nuevo, corregidos | FP históricos sin sospecha, confirmados, corregidos |
|---|---:|---:|
| B0 | 0 | 0 |
| B1 | 1 | 0 |
| B2 | 2 | 1 |
| B3 | 6 | 6 |
| B4 | 0 | 1 |
| B5 | 2 | 1 |
| B6 | 7 | 4 |
| B7 | 7 | 4 |
| B1_norm | 1 | 0 |
| B0_sin_NLI | 1 | 8 |
| B3_sin_NLI | 1 | 6 |
| B5_sustituye_matices | 9 | 0 |

La cobertura léxica es un predictor, nunca una prueba de veracidad ni un descuento del score. Los controles sintéticos incluyen intercambio de entidades con cobertura 1. Las correcciones y regresiones completas se conservan por ID en cambios_por_caso.csv.

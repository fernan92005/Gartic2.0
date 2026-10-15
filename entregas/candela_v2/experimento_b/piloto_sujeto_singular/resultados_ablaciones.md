# Resultados de ablaciones B

Todas las variantes se reentrenan con GB de 300 árboles, profundidad 3, learning rate 0,03, subsample 0,8, mínimo 20 y peso H=5. Se mantiene la configuración de v3 para aislar las características. El corte se selecciona por score oficial H dentro de tres folds internos; cuatro folds externos evalúan. No se usa el leaderboard.

| Variante | Score H | Δ H | AUROC H | FP H | FN H | Score global | AUROC global | Score multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.3494 | +0.0000 | 0.8989 | 130 | 8 | 0.4714 | 0.9566 | 0.5000 |
| B1 | 0.3695 | +0.0201 | 0.8957 | 121 | 9 | 0.5139 | 0.9555 | 0.5625 |
| B2 | 0.3454 | -0.0040 | 0.8998 | 127 | 9 | 0.4894 | 0.9574 | 0.5417 |
| B3 | 0.3454 | -0.0040 | 0.8937 | 111 | 13 | 0.5368 | 0.9581 | 0.6458 |
| B4 | 0.3534 | +0.0040 | 0.8985 | 129 | 8 | 0.4681 | 0.9565 | 0.4792 |
| B5 | 0.3574 | +0.0080 | 0.8992 | 124 | 9 | 0.5057 | 0.9570 | 0.5833 |
| B6 | 0.3454 | -0.0040 | 0.8939 | 107 | 14 | 0.5286 | 0.9579 | 0.6667 |
| B7 | 0.3695 | +0.0201 | 0.8954 | 105 | 13 | 0.5466 | 0.9588 | 0.6667 |
| B1_norm | 0.3534 | +0.0040 | 0.8971 | 121 | 10 | 0.5074 | 0.9561 | 0.5625 |
| B0_sin_NLI | 0.3133 | -0.0361 | 0.8910 | 95 | 19 | 0.4763 | 0.9505 | 0.6250 |
| B3_sin_NLI | 0.1647 | -0.1847 | 0.8867 | 136 | 18 | 0.4141 | 0.9504 | 0.6667 |
| B5_sustituye_matices | 0.2932 | -0.0562 | 0.8925 | 120 | 14 | 0.4615 | 0.9557 | 0.5625 |

## Transiciones frente a B0 — H

| Variante | FP eliminados | FP nuevos | FN eliminados | FN nuevos | ΔCoste |
|---|---:|---:|---:|---:|---:|
| B0 | 0 | 0 | 0 | 0 | 0 |
| B1 | 11 | 2 | 0 | 1 | -5 |
| B2 | 13 | 10 | 1 | 2 | 1 |
| B3 | 23 | 4 | 1 | 6 | 1 |
| B4 | 1 | 0 | 0 | 0 | -1 |
| B5 | 15 | 9 | 0 | 1 | -2 |
| B6 | 30 | 7 | 1 | 7 | 1 |
| B7 | 32 | 7 | 1 | 6 | -5 |
| B1_norm | 11 | 2 | 0 | 2 | -1 |
| B0_sin_NLI | 39 | 4 | 2 | 13 | 9 |
| B3_sin_NLI | 26 | 32 | 2 | 12 | 46 |
| B5_sustituye_matices | 21 | 11 | 1 | 7 | 14 |

Las transiciones globales están en transiciones.csv; cada caso que cambia está en cambios_por_caso.csv. El coste se comprueba como ΔFP + 4 ΔFN. Las etiquetas de tipo nunca entran en el extractor.

## Segunda partición agrupada de desarrollo

| Variante | Score H | Δ H frente a B0 | AUROC H | ΔAUROC H |
|---|---:|---:|---:|---:|
| B0 | 0.3133 | +0.0000 | 0.8920 | +0.0000 |
| B1 | 0.3414 | +0.0281 | 0.8901 | -0.0018 |
| B2 | 0.3213 | +0.0080 | 0.8966 | +0.0047 |
| B3 | 0.3173 | +0.0040 | 0.8939 | +0.0019 |
| B4 | 0.3092 | -0.0040 | 0.8911 | -0.0009 |
| B5 | 0.3012 | -0.0120 | 0.8958 | +0.0039 |
| B6 | 0.3173 | +0.0040 | 0.8991 | +0.0071 |
| B7 | 0.3494 | +0.0361 | 0.8996 | +0.0077 |

## Estabilidad, incertidumbre y decisión

| Variante | Δ H por fold principal | IC95% ΔH | IC95% simultáneo (7) | Supera controles |
|---|---|---|---|---|
| B1 | +0.000, +0.081, -0.016, +0.016 | [-0.0194, 0.053] | [-0.0344, 0.0675] | False |
| B2 | +0.000, +0.145, -0.078, -0.082 | [-0.0721, 0.062] | [-0.0979, 0.0887] | False |
| B3 | +0.032, +0.081, -0.156, +0.033 | [-0.0985, 0.0791] | [-0.1361, 0.1015] | False |
| B4 | +0.000, +0.000, +0.016, +0.000 | [0.0, 0.0137] | [0.0, 0.019] | False |
| B5 | +0.048, +0.145, -0.141, -0.016 | [-0.0422, 0.0535] | [-0.0645, 0.0675] | False |
| B6 | -0.016, +0.177, -0.234, +0.066 | [-0.1055, 0.0881] | [-0.1405, 0.1193] | False |
| B7 | -0.016, +0.177, -0.156, +0.082 | [-0.0766, 0.1079] | [-0.1145, 0.1337] | False |

Criterios fijados en protocolo.json: Δscore H>0 en al menos 3/4 folds principales; IC95% pareado Bonferroni7 H con límite inferior>0; Δscore H repetición 2>0; ΔAUROC H>=0 en ambas repeticiones; coste global y multi no aumentan; FN sobregeneralización H no aumentan. Sin cambiar condiciones tras ver resultados.

Bootstrap pareado de 3.000 muestras de grupos completos (semilla 20261015), condicionado a estas predicciones. Los intervalos Bonferroni son un control aproximado por las siete comparaciones principales; las cuatro ablaciones auxiliares son exploratorias y no compiten por exportación. La segunda partición reutiliza casos y no es una prueba independiente.

## Ablaciones auxiliares

- B1_norm sustituye únicamente los cuatro recuentos B1 por proporciones sobre las comprobaciones de matiz; las características v3 se conservan.
- B0_sin_NLI retira las características NLI y sus cruces (incluido llm_contra_sin_apoyo); B3_sin_NLI añade cobertura a ese mismo control. Comparar su diferencia con B3−B0 mide complementariedad, sin atribuir causalidad a una sola variable.
- B5_sustituye_matices retira explícitamente matiz_perdido_* y llmok_matiz_* originales y añade B1+B2. El resto de variantes conserva todas las características v3.

Los resultados al corte fijo original 0,16 también figuran en metricas.csv. Permiten separar sensibilidad al umbral de cambios de ordenación, sin seleccionar con etiquetas externas.

## Recomendación

**Conservar B0/v3. Ninguna variante supera todos los controles; no se genera mi_bot_candidato_b.py.**

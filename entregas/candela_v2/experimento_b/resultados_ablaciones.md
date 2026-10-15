# Resultados de ablaciones B

Todas las variantes se reentrenan con GB de 300 árboles, profundidad 3, learning rate 0,03, subsample 0,8, mínimo 20 y peso H=5. Se mantiene la configuración de v3 para aislar las características. El corte se selecciona por score oficial H dentro de tres folds internos; cuatro folds externos evalúan. No se usa el leaderboard.

| Variante | Score H | Δ H | AUROC H | FP H | FN H | Score global | AUROC global | Score multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.3494 | +0.0000 | 0.8989 | 130 | 8 | 0.4714 | 0.9566 | 0.5000 |
| B1 | 0.3695 | +0.0201 | 0.8957 | 121 | 9 | 0.5139 | 0.9555 | 0.5625 |
| B2 | 0.3454 | -0.0040 | 0.8998 | 127 | 9 | 0.4894 | 0.9574 | 0.5417 |
| B3 | 0.3454 | -0.0040 | 0.8937 | 111 | 13 | 0.5368 | 0.9581 | 0.6458 |
| B4 | 0.3614 | +0.0120 | 0.8988 | 123 | 9 | 0.5155 | 0.9565 | 0.5625 |
| B5 | 0.3574 | +0.0080 | 0.8992 | 124 | 9 | 0.5057 | 0.9570 | 0.5833 |
| B6 | 0.3454 | -0.0040 | 0.8939 | 107 | 14 | 0.5286 | 0.9579 | 0.6667 |
| B7 | 0.3896 | +0.0402 | 0.8955 | 108 | 11 | 0.5466 | 0.9587 | 0.6667 |
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
| B4 | 9 | 2 | 0 | 1 | -3 |
| B5 | 15 | 9 | 0 | 1 | -2 |
| B6 | 30 | 7 | 1 | 7 | 1 |
| B7 | 30 | 8 | 1 | 4 | -10 |
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
| B4 | 0.2972 | -0.0161 | 0.8922 | +0.0002 |
| B5 | 0.3012 | -0.0120 | 0.8958 | +0.0039 |
| B6 | 0.3173 | +0.0040 | 0.8991 | +0.0071 |
| B7 | 0.3333 | +0.0201 | 0.8997 | +0.0078 |

## Estabilidad, incertidumbre y decisión

| Variante | Δ H por fold principal | IC95% ΔH | IC95% simultáneo (7) | Supera controles |
|---|---|---|---|---|
| B1 | +0.000, +0.081, -0.016, +0.016 | [-0.0194, 0.053] | [-0.0344, 0.0675] | False |
| B2 | +0.000, +0.145, -0.078, -0.082 | [-0.0721, 0.062] | [-0.0979, 0.0887] | False |
| B3 | +0.032, +0.081, -0.156, +0.033 | [-0.0985, 0.0791] | [-0.1361, 0.1015] | False |
| B4 | +0.000, +0.065, -0.016, +0.000 | [-0.0324, 0.0472] | [-0.0553, 0.0576] | False |
| B5 | +0.048, +0.145, -0.141, -0.016 | [-0.0422, 0.0535] | [-0.0645, 0.0675] | False |
| B6 | -0.016, +0.177, -0.234, +0.066 | [-0.1055, 0.0881] | [-0.1405, 0.1193] | False |
| B7 | -0.016, +0.177, -0.078, +0.082 | [-0.0474, 0.121] | [-0.0843, 0.1459] | False |

Criterios fijados en protocolo.json: Δscore H>0 en al menos 3/4 folds principales; IC95% pareado Bonferroni7 H con límite inferior>0; Δscore H repetición 2>0; ΔAUROC H>=0 en ambas repeticiones; coste global y multi no aumentan; FN sobregeneralización H no aumentan. Sin cambiar condiciones tras ver resultados.

Bootstrap pareado de 3.000 muestras de grupos completos (semilla 20261015), condicionado a estas predicciones. Los intervalos Bonferroni son un control aproximado por las siete comparaciones principales; las cuatro ablaciones auxiliares son exploratorias y no compiten por exportación. La segunda partición reutiliza casos y no es una prueba independiente.

## Ablaciones auxiliares

- B1_norm sustituye únicamente los cuatro recuentos B1 por proporciones sobre las comprobaciones de matiz; las características v3 se conservan.
- B0_sin_NLI retira las características NLI y sus cruces (incluido llm_contra_sin_apoyo); B3_sin_NLI añade cobertura a ese mismo control. Comparar su diferencia con B3−B0 mide complementariedad, sin atribuir causalidad a una sola variable.
- B5_sustituye_matices retira explícitamente matiz_perdido_* y llmok_matiz_* originales y añade B1+B2. El resto de variantes conserva todas las características v3.

Los resultados al corte fijo original 0,16 también figuran en metricas.csv. Permiten separar sensibilidad al umbral de cambios de ordenación, sin seleccionar con etiquetas externas.

## Recomendación

**Conservar B0/v3. Ninguna variante supera todos los controles; no se genera mi_bot_candidato_b.py.**

## Revisión de cobertura B4 y control de columnas constantes

La primera ejecución se conserva en piloto_sujeto_singular/. B4 no se activaba en ningún caso de desarrollo: cubría singular→colectivo, pero no subconjunto numerado→grupo completo. Sus variaciones de score y las diferencias B6/B7 de aquella ejecución no demostraban valor de B4: añadir columnas constantes cambia la permutación de variables/desempates del algoritmo.

La revisión añade únicamente el mecanismo de subconjuntos cuantificados, ya solicitado en el encargo, y anclaje de confianza con la pregunta; seis controles sintéticos adicionales prueban fracción estricta, fracción completa, propiedad distinta, restricción preservada y evidencia completa. Se han repetido todas las variantes con el mismo protocolo y criterios. El cambio ocurrió después de inspeccionar el piloto: las conclusiones son exploratorias y los IC simultáneos no corrigen toda esta adaptación. No se ha buscado otra configuración o corte sobre folds externos.

## Qué aporta valor y qué añade ruido

B1: ΔH +0.0201; normalizar sus recuentos produce ΔH +0.0040. B2: ΔH -0.0040; B5 con ambas familias: +0.0080. No se declara utilidad estable solo por una mejora puntual.

B3 con NLI: ΔH -0.0040; sin NLI, el incremento B3_sin_NLI−B0_sin_NLI es -0.1486. La cobertura no demuestra sustituir las señales NLI. Retirar los matices originales (B5_sustituye_matices) produce ΔH -0.0562: no se recomienda reemplazarlos sin evidencia adicional.

Se reconstruyen los 32 modelos externos principales sin cambiar configuración y se comprueba igualdad de sus predicciones OOF. uso_caracteristicas.csv enumera activaciones, divisiones e importancia por variable. Esta importancia es descriptiva y no prueba valor causal.

| Variante | Familia nueva | Divisiones en cuatro modelos externos |
|---|---|---:|
| B1 | matiz_clausula_restrictivo, matiz_clausula_condicional, matiz_clausula_aproximado, matiz_clausula_parcial | 45 |
| B2 | matiz_parafrasis_preservado, matiz_parafrasis_perdido, matiz_equivalencia_incierta | 130 |
| B3 | cobertura_mejor_fuente, cobertura_union, ganancia_cobertura_union, numero_fuentes_complementarias | 915 |
| B4 | sujeto_generalizado, sujeto_generalizado_confianza | 0 |

**B4 no se utiliza en ninguna división.** La revisión detecta los dos escapes como señal, pero la activación es demasiado escasa para min_samples_leaf=20. Añadirla no demuestra aportar información al clasificador: su variación de score procede de la dinámica de entrenamiento/desempates al cambiar el número de columnas. La comparación B7/B6 tampoco acredita una ganancia de la señal B4. No se reduce min_samples_leaf tras observar estos casos.

matiz_equivalencia_incierta tampoco se utiliza en las divisiones de B2: su cobertura es demasiado escasa para demostrar valor. B1 conserva la señal más prometedora para una investigación futura, pero su AUROC H cae en ambas particiones y sus IC incluyen cero. B2/B3 aportan cambios mixtos; la combinación B7 mejora el promedio de score H, con regresiones de FN y estabilidad insuficiente.

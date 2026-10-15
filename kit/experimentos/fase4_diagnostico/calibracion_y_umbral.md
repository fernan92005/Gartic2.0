# Calibración y umbral — diagnóstico sin cambios

Corte del bot Joseca: p ≥0,15, reescalado monótonamente a score ≥0,5. Se conserva sin cambios. Los coeficientes y la normalización se reajustan dentro de cada fold; no se aplican los pesos aprendidos con todo train para construir OOF.

El código de entrenamiento original eligió configuración y un único corte sobre OOF H repetido con semillas 0,1,2. Ese proceso tiene optimismo de selección. Aquí se identifica la configuración ya existente mediante igualdad de parámetros, sin elegir una nueva. Para evaluar otro corte de forma diagnóstica se usa calibración anidada: cuatro folds externos y tres internos, con el mismo agrupamiento.

| Arquitectura / dominio | N | Score | AUROC | FP | FN | Coste | Matriz [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---|
| Corte actual 0,15 / H | 473 | 0.2651 | 0.8598 | 135 | 12 | 183 | [[114,135],[12,212]] |
| Corte actual 0,15 / no-H | 807 | 0.3867 | 0.9661 | 198 | 6 | 222 | [[164,198],[6,439]] |
| Corte actual 0,15 / todos | 1280 | 0.3372 | 0.9342 | 333 | 18 | 405 | [[278,333],[18,651]] |
| Corte interno, score oficial global / H | 473 | 0.1205 | 0.8598 | 95 | 31 | 219 | [[154,95],[31,193]] |
| Corte interno, score oficial global / no-H | 807 | 0.5414 | 0.9661 | 106 | 15 | 166 | [[256,106],[15,430]] |
| Corte interno, score oficial global / todos | 1280 | 0.3699 | 0.9342 | 201 | 46 | 385 | [[410,201],[46,623]] |
| Corte interno, objetivo 0,7/0,3 / H | 473 | 0.1968 | 0.8598 | 116 | 21 | 200 | [[133,116],[21,203]] |
| Corte interno, objetivo 0,7/0,3 / no-H | 807 | 0.4890 | 0.9661 | 149 | 9 | 185 | [[213,149],[9,436]] |
| Corte interno, objetivo 0,7/0,3 / todos | 1280 | 0.3699 | 0.9342 | 265 | 30 | 385 | [[346,265],[30,639]] |
| Corte interno, criterio H original / H | 473 | 0.1807 | 0.8598 | 128 | 19 | 204 | [[121,128],[19,205]] |
| Corte interno, criterio H original / no-H | 807 | 0.4171 | 0.9661 | 179 | 8 | 211 | [[183,179],[8,437]] |
| Corte interno, criterio H original / todos | 1280 | 0.3208 | 0.9342 | 307 | 27 | 415 | [[304,307],[27,642]] |

## Cortes por fold y rendimiento externo

| Fold | Corte actual | Corte interno 0,7/0,3 | Corte interno H | Objetivo al corte actual | Objetivo al corte interno |
|---|---:|---:|---:|---:|---:|
| 0 | 0.150 | 0.180 | 0.180 | 0.3831 | 0.4527 |
| 1 | 0.150 | 0.200 | 0.180 | 0.3191 | 0.2044 |
| 2 | 0.150 | 0.240 | 0.180 | 0.3616 | 0.3148 |
| 3 | 0.150 | 0.170 | 0.130 | 0.1381 | 0.1644 |

| Fold | Corte elegido solo con score oficial interno | Score global externo |
|---|---:|---:|
| 0 | 0.250 | 0.3595 |
| 1 | 0.280 | 0.3618 |
| 2 | 0.240 | 0.4516 |
| 3 | 0.200 | 0.3046 |

Se muestran tres criterios de selección interna: score oficial global, criterio histórico H y objetivo interno 0,7 H + 0,3 no-H. El último no pertenece a la métrica oficial. Los tres usan exactamente los mismos modelos y predicciones internas; se informa de todos, sin elegir retrospectivamente el que gane en los folds externos.

## Incertidumbre por escenarios

| Protocolo | IC95% score H | IC95% score no-H | IC95% objetivo |
|---|---|---|---|
| Corte actual | [0.1336,0.3860] | [0.3178,0.4507] | [0.2075,0.3879] |
| Corte interno oficial | [-0.0467,0.2885] | [0.4503,0.6290] | [0.1230,0.3720] |
| Corte interno ponderado | [0.0536,0.3390] | [0.4070,0.5665] | [0.1830,0.3878] |

La mejor rejilla retrospectiva en el OOF completo es 0.16 para el objetivo ponderado y 0.16 para H. Son valores descriptivos optimistas, no una recomendación ni cortes adoptados. La calibración anidada cambia el objetivo en -0.0171 respecto al corte actual.

Curva completa en `curva_coste_oof.csv`: contiene FP, FN y coste H/no-H/global para todos los cortes de la rejilla original 0,02–0,79. La figura `curva_coste.svg` muestra coste normalizado ponderado y el corte actual. No se ajusta con leaderboard, test o giro.

Interpretación: comparar AUROC con score permite separar ordenación y corte; el AUROC no cambia al recolocar un único corte global. Los cortes anidados variables por fold se comparan como decisiones externas, no se confunden con una transformación monótona global. La diferencia actual/anidado no aísla causalmente toda la selección previa de variables o hiperparámetros. El bot ya utilizó todo train al entrenarse; ninguna sección de este fichero constituye una nueva reserva ciega.

## Valoración del corte actual

No hay evidencia para sustituir 0,15 por un corte mayor: la calibración anidada reduce el coste global de 405 a 385, pero empeora H (183 a 200) y el objetivo interno ponderado de 0,3016 a 0,2844. La mejor rejilla retrospectiva queda cerca, en 0,16, y no es validación independiente. El resultado sugiere estudiar las falsas alarmas sistemáticas antes de atribuir el problema exclusivamente al umbral. No demuestra que 0,15 sea óptimo en el test público.

| Diferencia de score anidado − actual | IC95% bootstrap pareado por grupos |
|---|---|
| H | [-0.1775,+0.0195] |
| no-H | [+0.0484,+0.1532] |
| todos | [-0.0229,+0.0810] |
| objective | [-0.0937,+0.0456] |

Con selección interna de la métrica oficial global: score 0.3372 → 0.3699; coste 405 → 385; FP 333 → 201; FN 18 → 46. Diferencia global +0.0327, IC95% pareado [-0.0457,+0.1089]. Score H 0.2651 → 0.1205; no-H 0.3867 → 0.5414.

El balance depende de la mezcla de dominios. Estos son cortes variables elegidos dentro de cada entrenamiento y evaluados fuera; no identifican un único nuevo corte para el bot. El umbral actual es defendible para H frente a los criterios evaluados, pero no se ha demostrado óptimo. La calidad de ordenación en H (AUROC 0,860 frente a 0,966 no-H) y los patrones de falsas alarmas también limitan la puntuación.

2.000 remuestreos de grupos completos con las mismas selecciones para ambos métodos. Los intervalos condicionan las predicciones OOF existentes y no incorporan toda la incertidumbre de selección de arquitectura. No se afirma superioridad estadística por comparar solamente los promedios.

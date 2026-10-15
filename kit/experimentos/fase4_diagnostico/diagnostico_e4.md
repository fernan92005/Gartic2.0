# Diagnóstico E4: referencia separada del bot Joseca

**Joseca no contiene pair_bad, source_math_bad, binding_bad ni max(HGB,relacional).** Esos componentes pertenecen a `entregas/candela_v1/mi_bot.py`, cuyo SHA-256 coincide exactamente con el candidato E4 corregido de fase 3. No se le atribuyen sus mejoras al bot solicitado. En Joseca las comprobaciones numéricas/matices son características aprendidas de una logística, no un veto max.

Se reentrena el HGB de Candela dentro de cada fold de desarrollo, con sus parámetros originales (120 árboles, 7 hojas, mínimo 20, learning rate 0,06, L2=5, peso no-H=0,35). Después se aplica el verificador fijo a validación. El HGB reproduce score H OOF 0,3092369478, comprobación adicional de la reconstrucción de particiones.

| Dominio | Verificador | Activa | Alucinadas | Fieles | FP nuevos | FN corregidos | ΔCoste | ΔScore |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| todos | pair_bad | 5 | 5 | 0 | 0 | 2 | -8 | +0.0131 |
| todos | source_math_bad | 5 | 5 | 0 | 0 | 2 | -8 | +0.0131 |
| todos | binding_bad | 0 | 0 | 0 | 0 | 0 | +0 | +0.0000 |
| todos | union_max | 10 | 10 | 0 | 0 | 4 | -16 | +0.0262 |
| H | pair_bad | 5 | 5 | 0 | 0 | 2 | -8 | +0.0321 |
| H | source_math_bad | 3 | 3 | 0 | 0 | 2 | -8 | +0.0321 |
| H | binding_bad | 0 | 0 | 0 | 0 | 0 | +0 | +0.0000 |
| H | union_max | 8 | 8 | 0 | 0 | 4 | -16 | +0.0643 |
| no-H | pair_bad | 0 | 0 | 0 | 0 | 0 | +0 | +0.0000 |
| no-H | source_math_bad | 2 | 2 | 0 | 0 | 0 | +0 | +0.0000 |
| no-H | binding_bad | 0 | 0 | 0 | 0 | 0 | +0 | +0.0000 |
| no-H | union_max | 2 | 2 | 0 | 0 | 0 | +0 | +0.0000 |

Las filas individuales se solapan potencialmente: el efecto conjunto se calcula por la unión de casos, nunca sumando alarmas repetidas. Las categorías detectadas por cada verificador están en contribucion_e4.csv; las que tienen cero activaciones son cobertura no demostrada, no tipos que el HGB deje necesariamente de detectar.

## Solapamientos

| Par de verificadores | Intersección | Unión | Jaccard |
|---|---:|---:|---:|
| pair_bad / source_math_bad | 0 | 10 | 0.000 |
| pair_bad / binding_bad | 0 | 5 | 0.000 |
| source_math_bad / binding_bad | 0 | 5 | 0.000 |

## Cobertura de comprobaciones y unidades

| Verificador | Casos examinados | Casos con fallo |
|---|---:|---:|
| pair | 5 | 5 |
| source_math | 9 | 5 |
| binding | 41 | 0 |

Unidades extraídas en respuestas (conteos de cantidades, no de casos): {'length': 128, 'year': 404, 'number': 912, 'percent': 303, 'area': 36, 'amount': 418, 'months': 42, 'people': 161, 'time': 129, 'mass': 49, 'amount_per_month': 10}. Casos con algún factor de normalización distinto de uno: 238.

De los FN reales del Joseca OOF, **0** coinciden con alguna alerta de E4. Es solo cobertura potencial: no se implementó ni evaluó una nueva combinación Joseca+E4 y no se asume que el balance global de FP sería favorable.

La normalización convierte céntimos, escalas, distancias/masas y separa meses/tasas; se abstiene ante números escritos no interpretados. No se declara verificada una conversión solamente porque su unidad se haya normalizado. No se desactiva ni modifica la normalización del bot para presentar una ganancia nueva. Los falsos positivos atribuibles al max son exactamente los FP nuevos de la tabla. La ausencia de activaciones en binding implica que su utilidad empírica en este lote es nula.

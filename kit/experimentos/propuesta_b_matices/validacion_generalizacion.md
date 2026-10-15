# Validación y generalización

Desarrollo: 1.280 casos, 473 H, 252 componentes de escenario; reserva histórica: 427, excluida de la comparación B. Se reutilizan exactamente los cuatro folds externos del diagnóstico fase 4. Los grupos conectan ID base, pregunta, fuentes idénticas y solape >=80%; así se evita separar escenarios duplicados con distinto ID. Los manifiestos originales de fases 1–3 faltaban y la reconstrucción previa queda como limitación heredada.

La repetición de estabilidad usa semilla 20261018 y cuatro folds sobre los mismos grupos de desarrollo. Cada fold externo tiene tres internos, semilla 20261017+fold (+100 en la repetición 2). IDs, grupos y tipos solo se usan para particiones, pesos históricos y diagnóstico, nunca como características.

## Cohortes estructurales

| Cohorte | N | Score B0 | AUROC B0 | FP B0 | FN B0 |
|---|---:|---:|---:|---:|---:|
| una_oracion | 1131 | 0.4689 | 0.9528 | 255 | 11 |
| varias_oraciones | 149 | 0.5000 | 0.9851 | 24 | 0 |
| fuentes_2_3 | 15 | 0.8000 | 1.0000 | 1 | 0 |
| muchas_fuentes_6mas | 945 | 0.3537 | 0.9331 | 256 | 10 |
| complementarias_2 | 347 | 0.2657 | 0.9203 | 85 | 5 |
| complementarias_3 | 217 | 0.3119 | 0.9718 | 75 | 0 |
| enumeraciones | 661 | 0.3302 | 0.9380 | 175 | 10 |
| comparaciones | 274 | 0.4780 | 0.9494 | 75 | 2 |
| condicion_otra_oracion_proxy | 46 | 0.5789 | 0.9883 | 8 | 0 |
| muchos_numeros_10mas | 666 | 0.3780 | 0.9200 | 173 | 9 |
| respuesta_larga_60mas | 57 | 0.4231 | 0.9752 | 15 | 0 |

Comparación completa de todas las variantes en robustez.csv, incluidos cohortes pequeños que no permiten conclusiones estables. Las enumeraciones, comparaciones y condiciones en otra oración son proxies, no anotaciones exhaustivas. Las cohortes se solapan.

## Dependencia de longitud y cobertura

| Característica | Correlación con nº palabras | % activación una oración | % activación varias oraciones |
|---|---:|---:|---:|
| matiz_clausula_restrictivo | +0.134 | 2.5% | 8.7% |
| matiz_clausula_condicional | +0.059 | 4.2% | 6.7% |
| matiz_clausula_aproximado | +0.096 | 3.3% | 5.4% |
| matiz_clausula_parcial | -0.003 | 2.6% | 1.3% |
| matiz_parafrasis_preservado | +0.169 | 12.6% | 30.9% |
| matiz_parafrasis_perdido | +0.130 | 10.3% | 18.1% |
| matiz_equivalencia_incierta | -0.008 | 0.1% | 0.0% |
| cobertura_mejor_fuente | -0.474 | 99.8% | 100.0% |
| cobertura_union | -0.289 | 99.8% | 100.0% |
| ganancia_cobertura_union | +0.371 | 46.3% | 73.2% |
| numero_fuentes_complementarias | +0.532 | 46.1% | 73.2% |
| sujeto_generalizado | -0.002 | 0.5% | 0.0% |
| sujeto_generalizado_confianza | -0.002 | 0.5% | 0.0% |

B1 limita recuentos a 2 y B2 a 2; B3 normaliza coberturas por raíces únicas y cuenta como máximo tres fuentes; B4 es binaria/confianza. B1_norm comprueba explícitamente sustituir los recuentos de B1 por proporciones. Ninguna correlación prueba por sí sola un atajo ni una mejora.

## Límites

- El extractor y la arquitectura v3 ya se diseñaron usando todo train. Reentrenar dentro de folds evita fuga de coeficientes/árboles, pero no borra el sesgo de selección previo.
- La reserva histórica ya fue inspeccionada en fases anteriores y v3 se entrenó con todo train. No se presenta como conjunto ciego ni se usa para seleccionar B.
- Las hipótesis parten de errores ya observados. Dos repartos e intervalos por grupos reducen optimismo, pero no sustituyen datos nuevos; seleccionar entre variantes sigue siendo exploratorio.
- B4 se revisó tras observar cobertura nula en un piloto; se conserva el piloto y se repite el protocolo completo. Esta adaptación añade sesgo de selección y no queda eliminada por Bonferroni entre siete variantes.
- No se han leído ni puntuado test/giro para este experimento, ni se han creado sus etiquetas. Los cohortes de respuestas largas son proxies de robustez, no rendimiento sobre giro.
- Las equivalencias y la selección de cláusulas no resuelven semántica abierta, negación y correferencia en general. Cobertura alta puede coexistir con atribución o relación falsa.

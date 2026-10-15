# Falsos positivos del Joseca

Hay **333** respuestas fieles clasificadas como alucinadas; coste **333**, 82.2% del coste total. No son una quinta categoría de alucinación.

Los patrones se calculan solo dentro de fieles y se solapan. No se suman sus costes para obtener el total. Paráfrasis, conversión y síntesis son proxies observables; no nuevas anotaciones manuales.

| Patrón | Fieles en cohorte | FP | Coste FP | % de todos los FP |
|---|---:|---:|---:|---:|
| Síntesis correcta multifuente (proxy) | 170 | 116 | 116 | 34.8% |
| Paráfrasis / vocabulario nuevo (proxy) | 364 | 151 | 151 | 45.3% |
| Aritmética válida según comprobador (proxy) | 93 | 9 | 9 | 2.7% |
| Posible conversión de unidades | 14 | 8 | 8 | 2.4% |
| Contradicción NLI espuria (cualquier fuente) | 193 | 111 | 111 | 33.3% |
| MiniMax pesimista en fiel | 12 | 12 | 12 | 3.6% |
| Bajo respaldo en fuente citada | 321 | 184 | 184 | 55.3% |
| Fragmento bajo, oración alta | 22 | 16 | 16 | 4.8% |
| Sospecha numérica del Joseca | 42 | 33 | 33 | 9.9% |
| Alerta E4 (no componente Joseca) | 0 | 0 | 0 | 0.0% |

## Contribuciones del modelo lineal

Se agrupan las contribuciones al logit centradas por la media del entrenamiento de cada fold. La familia de mayor contribución no prueba causalidad: hay correlaciones y cruces entre características. Puede ser negativa si todas lo son; el intercepto y el corte también determinan la decisión.

| Familia dominante entre FP | Casos |
|---|---:|
| texto | 164 |
| matices | 70 |
| otros | 41 |
| numeros | 35 |
| nli | 13 |
| llm | 10 |

## Ablación diagnóstica de señales a cero, sin reentrenar

Se ponen a cero las características de una familia y sus cruces manteniendo coeficientes y corte. No se modifica el bot ni se propone desplegar esta ablación. Es una intervención descriptiva sobre variables correlacionadas y no una estimación de un modelo reentrenado.

| Familia a cero | FP | FN | ΔFP | ΔFN | ΔCoste |
|---|---:|---:|---:|---:|---:|
| llm | 515 | 10 | +182 | -8 | +150 |
| nli | 583 | 3 | +250 | -15 | +190 |
| numeros | 450 | 10 | +117 | -8 | +85 |
| matices | 296 | 31 | -37 | +13 | +15 |
| texto | 441 | 9 | +108 | -9 | +72 |
| otros | 386 | 16 | +53 | -2 | +45 |

## Ejemplos FP más confiados (solo desarrollo)

- T25::r1 (fold 0, p=0.996): 1,9 millones de pasajeros, de los que unos 570.000 fueron turistas.
- e24::par (fold 0, p=0.989): El programa se imparte a lo largo de tres trimestres e incorpora optativas diversas.
- s38::fiel1 (fold 1, p=0.989): Sus 400 socios facturaron 30 millones exportando aceite a 15 países, respaldados por un sello de calidad.
- H210::r7 (fold 0, p=0.987): El seguro de olivar no cubre en ningún caso la pérdida de cosecha por sequía.
- H109::r1 (fold 3, p=0.946): El 15 % de los alimentos repartidos se compró con fondos públicos.
- H422::r8 (fold 2, p=0.938): Sierra Rasa perdió el 9 % de sus habitantes en esa década, un porcentaje menor que el de Peñagrís.
- H213::r1 (fold 3, p=0.921): El ERTE afecta a 260 trabajadores de la planta de Vilanzo y durará hasta el 31 de julio de 2026.
- H411::r7 (fold 0, p=0.906): Con 600 entrevistas, el sondeo da a Unidos por Gandarela el 36,4 %, al Bloque Ciudadano el 29,1 % y a Alternativa Verde el 11,7 %, lo que permitiría a esta última entrar por primera vez en el pleno con 2 concejales.
- H404::r8 (fold 3, p=0.894): AVT y Unión Serrana, los grupos que votaron la investidura de Clotilde Olivenza, suman 8 de los 15 concejales.
- H212::r9 (fold 1, p=0.892): La tasa de empleo a los seis meses de los titulados en dual superó en 17 puntos a la de los ciclos ordinarios.
- N12::r2 (fold 2, p=0.884): Las obras empezaron en mayo de 2024 y está previsto que terminen en diciembre de 2026, con 7,8 millones adjudicados.
- N28::r3 (fold 2, p=0.882): Según la estimación de cada oficina, el gasto local fue de 21 millones en Altamar y de 12 en Valdeoro.

Las falsas alarmas E4 se analizan aparte; Joseca no ejecuta esos verificadores. Ni una cifra sospechosa ni un matiz presente en otra fuente demuestran que esa característica haya causado el FP: el score surge de la suma de todas las señales.

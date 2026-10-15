# Diagnóstico por tipo — Joseca v1

Objetivo principal: `entregas/joseca_v1/mi_bot.py`, SHA-256 `edd2cfda1bb3e5d6c811ced2c1e0c0b3553aeaefd2080256e921fd8ccf36a510`. Logística de 60 características, peso H=5, L2=10, corte crudo **0,15**. No es el HGB ni el E4 de Candela.

## Métrica oficial verificada

El DOCX (sección 7.1), kit/README.md y evaluar_local.py coinciden: score = 1 − (4 FN + FP)/N_fieles; decisión score ≥0,5. No existen pesos diferentes por tipo. Cada FN cuesta cuatro y cada FP uno. El peso H=5 es un peso de entrenamiento, no una penalización del evaluador. La ponderación 0,7 H + 0,3 no-H es un criterio interno de desarrollo, no una regla del leaderboard. El caso especial del evaluador cuando no hay fieles devuelve recall; no se confunde con la fórmula normal. No se encontró discrepancia con el DOCX.

## Evaluación OOF principal

| Arquitectura / dominio | N | Score | AUROC | FP | FN | Coste | Matriz [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---|
| Joseca OOF / H | 473 | 0.2651 | 0.8598 | 135 | 12 | 183 | [[114,135],[12,212]] |
| Joseca OOF / no-H | 807 | 0.3867 | 0.9661 | 198 | 6 | 222 | [[164,198],[6,439]] |
| Joseca OOF / todos | 1280 | 0.3372 | 0.9342 | 333 | 18 | 405 | [[278,333],[18,651]] |

## Ranking por coste de FN (desarrollo completo al corte actual)

| Orden | Tipo | Casos positivos | TP | FN | Recall | Coste FN | % del coste total, incluyendo FP | IC95% recall por grupos |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Distorsión | 274 | 267 | 7 | 0.974 | 28 | 6.9% | [0.955,0.992] |
| 2 | Sobregeneralización | 97 | 90 | 7 | 0.928 | 28 | 6.9% | [0.867,0.978] |
| 3 | No respaldado | 178 | 176 | 2 | 0.989 | 8 | 2.0% | [0.971,1.000] |
| 4 | Contradicción | 120 | 118 | 2 | 0.983 | 8 | 2.0% | [0.957,1.000] |

Hay empate de coste entre distorsión/sobregeneralización y entre no respaldado/contradicción. El orden dentro de cada empate usa frecuencia y no demuestra una diferencia de coste esperada.

## Estabilidad de los falsos negativos

| Tipo | FN fold 0 | FN fold 1 | FN fold 2 | FN fold 3 | Grupos con FN |
|---|---:|---:|---:|---:|---:|
| Distorsión | 0 | 0 | 3 | 4 | 7 |
| Sobregeneralización | 1 | 3 | 0 | 3 | 6 |
| No respaldado | 0 | 1 | 0 | 1 | 2 |
| Contradicción | 0 | 0 | 0 | 2 | 2 |

La concentración de errores por fold y los IC de recall impiden interpretar pocos FN como una jerarquía estable. El bootstrap remuestrea 2.000 veces grupos completos con semilla 20261015, condicionado a estos modelos OOF: no incluye la variación de volver a seleccionar la arquitectura ni nuevas realizaciones de entrenamiento.

Falsas alarmas sobre fieles: **333**, coste **333**, 82.2% del coste total **405**. No se les asigna un tipo de alucinación.

Todos los 1.707 ejemplos tienen un único campo tipo: 817 fieles con null y 890 positivos con una de las cuatro etiquetas. No hay listas ni categorías múltiples. Los costes por tipo más FP suman exactamente el coste oficial; las cohortes de mecanismo posteriores sí se solapan y no deben sumarse.

## Categorías por arquitectura y dominio

### joseca_oof_corte_actual

| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |
|---|---|---:|---:|---:|---:|---:|
| todos | Distorsión | 274 | 267 | 7 | 0.974 | 28 |
| todos | Contradicción | 120 | 118 | 2 | 0.983 | 8 |
| todos | No respaldado | 178 | 176 | 2 | 0.989 | 8 |
| todos | Sobregeneralización | 97 | 90 | 7 | 0.928 | 28 |
| H | Distorsión | 114 | 107 | 7 | 0.939 | 28 |
| H | Contradicción | 9 | 9 | 0 | 1.000 | 0 |
| H | No respaldado | 38 | 37 | 1 | 0.974 | 4 |
| H | Sobregeneralización | 63 | 59 | 4 | 0.937 | 16 |
| no-H | Distorsión | 160 | 160 | 0 | 1.000 | 0 |
| no-H | Contradicción | 111 | 109 | 2 | 0.982 | 8 |
| no-H | No respaldado | 140 | 139 | 1 | 0.993 | 4 |
| no-H | Sobregeneralización | 34 | 31 | 3 | 0.912 | 12 |

### joseca_oof_corte_anidado_oficial

| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |
|---|---|---:|---:|---:|---:|---:|
| todos | Distorsión | 274 | 259 | 15 | 0.945 | 60 |
| todos | Contradicción | 120 | 116 | 4 | 0.967 | 16 |
| todos | No respaldado | 178 | 168 | 10 | 0.944 | 40 |
| todos | Sobregeneralización | 97 | 80 | 17 | 0.825 | 68 |
| H | Distorsión | 114 | 101 | 13 | 0.886 | 52 |
| H | Contradicción | 9 | 8 | 1 | 0.889 | 4 |
| H | No respaldado | 38 | 34 | 4 | 0.895 | 16 |
| H | Sobregeneralización | 63 | 50 | 13 | 0.794 | 52 |
| no-H | Distorsión | 160 | 158 | 2 | 0.988 | 8 |
| no-H | Contradicción | 111 | 108 | 3 | 0.973 | 12 |
| no-H | No respaldado | 140 | 134 | 6 | 0.957 | 24 |
| no-H | Sobregeneralización | 34 | 30 | 4 | 0.882 | 16 |

### joseca_oof_corte_anidado_ponderado

| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |
|---|---|---:|---:|---:|---:|---:|
| todos | Distorsión | 274 | 265 | 9 | 0.967 | 36 |
| todos | Contradicción | 120 | 116 | 4 | 0.967 | 16 |
| todos | No respaldado | 178 | 174 | 4 | 0.978 | 16 |
| todos | Sobregeneralización | 97 | 84 | 13 | 0.866 | 52 |
| H | Distorsión | 114 | 105 | 9 | 0.921 | 36 |
| H | Contradicción | 9 | 8 | 1 | 0.889 | 4 |
| H | No respaldado | 38 | 36 | 2 | 0.947 | 8 |
| H | Sobregeneralización | 63 | 54 | 9 | 0.857 | 36 |
| no-H | Distorsión | 160 | 160 | 0 | 1.000 | 0 |
| no-H | Contradicción | 111 | 108 | 3 | 0.973 | 12 |
| no-H | No respaldado | 140 | 138 | 2 | 0.986 | 8 |
| no-H | Sobregeneralización | 34 | 30 | 4 | 0.882 | 16 |

### joseca_oof_corte_anidado_H

| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |
|---|---|---:|---:|---:|---:|---:|
| todos | Distorsión | 274 | 265 | 9 | 0.967 | 36 |
| todos | Contradicción | 120 | 117 | 3 | 0.975 | 12 |
| todos | No respaldado | 178 | 174 | 4 | 0.978 | 16 |
| todos | Sobregeneralización | 97 | 86 | 11 | 0.887 | 44 |
| H | Distorsión | 114 | 105 | 9 | 0.921 | 36 |
| H | Contradicción | 9 | 8 | 1 | 0.889 | 4 |
| H | No respaldado | 38 | 36 | 2 | 0.947 | 8 |
| H | Sobregeneralización | 63 | 56 | 7 | 0.889 | 28 |
| no-H | Distorsión | 160 | 160 | 0 | 1.000 | 0 |
| no-H | Contradicción | 111 | 109 | 2 | 0.982 | 8 |
| no-H | No respaldado | 140 | 138 | 2 | 0.986 | 8 |
| no-H | Sobregeneralización | 34 | 30 | 4 | 0.882 | 16 |

### candela_hgb_oof_referencia

| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |
|---|---|---:|---:|---:|---:|---:|
| todos | Distorsión | 274 | 266 | 8 | 0.971 | 32 |
| todos | Contradicción | 120 | 119 | 1 | 0.992 | 4 |
| todos | No respaldado | 178 | 177 | 1 | 0.994 | 4 |
| todos | Sobregeneralización | 97 | 92 | 5 | 0.948 | 20 |
| H | Distorsión | 114 | 106 | 8 | 0.930 | 32 |
| H | Contradicción | 9 | 9 | 0 | 1.000 | 0 |
| H | No respaldado | 38 | 38 | 0 | 1.000 | 0 |
| H | Sobregeneralización | 63 | 59 | 4 | 0.937 | 16 |
| no-H | Distorsión | 160 | 160 | 0 | 1.000 | 0 |
| no-H | Contradicción | 111 | 110 | 1 | 0.991 | 4 |
| no-H | No respaldado | 140 | 139 | 1 | 0.993 | 4 |
| no-H | Sobregeneralización | 34 | 33 | 1 | 0.971 | 4 |

### candela_e4_oof_referencia

| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |
|---|---|---:|---:|---:|---:|---:|
| todos | Distorsión | 274 | 270 | 4 | 0.985 | 16 |
| todos | Contradicción | 120 | 119 | 1 | 0.992 | 4 |
| todos | No respaldado | 178 | 177 | 1 | 0.994 | 4 |
| todos | Sobregeneralización | 97 | 92 | 5 | 0.948 | 20 |
| H | Distorsión | 114 | 110 | 4 | 0.965 | 16 |
| H | Contradicción | 9 | 9 | 0 | 1.000 | 0 |
| H | No respaldado | 38 | 38 | 0 | 1.000 | 0 |
| H | Sobregeneralización | 63 | 59 | 4 | 0.937 | 16 |
| no-H | Distorsión | 160 | 160 | 0 | 1.000 | 0 |
| no-H | Contradicción | 111 | 110 | 1 | 0.991 | 4 |
| no-H | No respaldado | 140 | 139 | 1 | 0.993 | 4 |
| no-H | Sobregeneralización | 34 | 33 | 1 | 0.971 | 4 |

## Mecanismos observables (proxies, no nuevos subtipos anotados)

| Cohorte | N | FP | FN | Coste |
|---|---:|---:|---:|---:|
| Distorsión con año explícito | 76 | 0 | 3 | 12 |
| Distorsión con cuenta sospechosa | 26 | 0 | 2 | 8 |
| Distorsión con posible atribución cruzada | 8 | 0 | 0 | 0 |
| Contradicción sin negación explícita | 111 | 0 | 1 | 4 |
| Contradicción con verbo direccional | 22 | 0 | 2 | 8 |
| No respaldado con causa añadida | 30 | 0 | 0 | 0 |
| No respaldado con MiniMax favorable | 41 | 0 | 2 | 8 |
| Sobregeneralización con absoluto añadido | 24 | 0 | 1 | 4 |
| Sobregeneralización con condición perdida | 28 | 0 | 0 | 0 |
| Respuesta >=60 palabras | 57 | 13 | 1 | 17 |

Los proxies no permiten distinguir exhaustivamente fecha incorrecta frente a fecha añadida, oposición semántica, atribución o pérdida de excepción. Se conservan los tipos oficiales y se evita reclasificar manualmente.

## Ejemplos FN de desarrollo, sin proponer reglas por caso

### Distorsión

- H316::r2 (fold 2, p=0.013): Con una tasa de 1,4 euros por metro cuadrado al mes, una antena que ocupe 10 metros cuadrados de dominio público pagará 140 euros al año.
- H422::r9 (fold 2, p=0.016): Entre 2015 y 2025 la comarca perdió 2.410 habitantes, un 18 % de su población.
- H213::r3 (fold 3, p=0.021): El acuerdo final redujo en 70 el número de trabajadores afectados respecto a la primera propuesta.
- H314::r2 (fold 3, p=0.021): El 3G se apagará primero en las provincias del norte, el 30 de junio de 2027, y nueve meses después en el resto del país.

### Sobregeneralización

- N09::r5 (fold 1, p=0.026): En toda la comunidad, el 30 % de las llamadas al 112 de 2025 fueron no urgentes, y los centros Norte y Sur sumaron 326.000.
- T10::r5 (fold 3, p=0.032): Los 18 nadadores que desplazó el club volvieron de Palma con al menos una medalla.
- H412::r2 (fold 1, p=0.038): Los locales comerciales pagarán un 9 % más de tasa de basuras en 2026, y las viviendas, un 4,5 % más.
- H322::r5 (fold 3, p=0.056): Se instalarán 24 tomas eléctricas en los muelles del puerto, con un coste de 2,8 millones de euros.

### No respaldado

- H216::r8 (fold 3, p=0.103): Los candidatos deben acreditar un nivel B1 de inglés o del idioma del país de destino para poder seguir la formación en la empresa.
- T09::r4 (fold 1, p=0.112): Terminó con un balance de 21-9, lo que le da el factor cancha en la semifinal ante el Morón.

### Contradicción

- R15::r5 (fold 3, p=0.069): Hogar 1G trae router wifi 6 y Hogar 600, wifi 5. Hogar 1G llega a 1 Gbps de bajada, pero su velocidad de subida queda por debajo de la de Hogar 600. En cuanto a garantías, la velocidad mínima por cable Ethernet es del 80 % en Hogar 600 y del 70 % en Hogar 1G, y por wifi no se garantiza ninguna. La instalación es gratuita con permanencia de 12 meses o cuesta 95 euros sin ella.
- T32::r3 (fold 3, p=0.113): Lanzarote recibió 3,1 millones de visitantes, aunque su crecimiento, del 4 %, se quedó por debajo del de Fuerteventura.

## Robustez y variación por escenario

Métricas por cada grupo en `metricas_por_escenario.csv`; cortes por fold en `folds_detalle.json`. Las variables estructurales siguientes son solo diagnósticas. La cohorte técnico/legal es una búsqueda léxica y no identifica exhaustivamente el dominio. No se ha puntuado ni etiquetado el giro.

| Variable/cohorte | N | Score | AUROC | FP | FN |
|---|---:|---:|---:|---:|---:|
| dominio: H | 473 | 0.265 | 0.860 | 135 | 12 |
| dominio: no-H | 807 | 0.387 | 0.966 | 198 | 6 |
| fuentes: 0–1 | 320 | 0.527 | 0.994 | 70 | 0 |
| fuentes: 2–5 | 15 | 0.600 | 1.000 | 2 | 0 |
| fuentes: 6–8 | 921 | 0.271 | 0.906 | 254 | 18 |
| fuentes: >=9 | 24 | 0.364 | 0.972 | 7 | 0 |
| oraciones: 0–1 | 1131 | 0.334 | 0.928 | 307 | 17 |
| oraciones: 2–3 | 83 | 0.409 | 0.996 | 13 | 0 |
| oraciones: >=4 | 66 | 0.346 | 0.956 | 13 | 1 |
| fragmentos: 0–1 | 410 | 0.420 | 0.965 | 100 | 5 |
| fragmentos: 2–4 | 786 | 0.293 | 0.912 | 210 | 12 |
| fragmentos: >=5 | 84 | 0.308 | 0.946 | 23 | 1 |
| palabras: 0–24 | 1081 | 0.367 | 0.940 | 261 | 14 |
| palabras: 25–59 | 142 | 0.155 | 0.890 | 59 | 3 |
| palabras: >=60 | 57 | 0.346 | 0.955 | 13 | 1 |
| entidades: >=2 nombres (proxy) | 200 | 0.296 | 0.914 | 53 | 4 |
| cifras: presentes | 1226 | 0.337 | 0.935 | 316 | 18 |
| fechas: año explícito | 316 | 0.329 | 0.906 | 78 | 6 |
| multifuente: respaldo repartido (proxy) | 275 | 0.200 | 0.852 | 116 | 5 |
| tecnico_legal: palabras documentales (proxy) | 136 | 0.156 | 0.880 | 38 | 4 |

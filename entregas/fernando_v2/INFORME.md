# fernando_v2 — ¿hay algo que mejore de verdad a joseca_v4?

> Documento interno. **No se sube** al repositorio de entregas, ni este informe ni `lab/`.
> Fecha: jueves 15/10, 16:15–17:00.
> Referencia: **joseca_v4**, con nota pública 0,497 (test + giro), 0,481 (solo test) y AUROC 0,913.

## 0. Respuesta corta

**No, con las garantías pedidas. fernando_v2 es la v4 sin cambios** en código, características y modelo; solo he reescrito su documentación.

- **Comprobado:** da 0 de 660 scores distintos a la v4, y su CSV del test es idéntico al que se entregó.
- **No mejorar es un resultado válido**, y aquí es la conclusión honesta.

**El candidato que estuvo a punto de pasar: A1, el verificador de cuentas ampliado de joseca_v5.**

- **En la CV del plan:** +0,067 ± 0,084, gana 11 de 15 particiones, y el hold-out sube de +0,507 a +0,582.
- **La auditoría independiente encontró que esa evidencia está contaminada:**
  - Las reglas de A1 se escribieron mirando casos del train, y sus comentarios citan casi literalmente 7 familias H.
  - Cinco de esas familias están en la CV y dos (H204, H320) en el hold-out.
  - Justo esos dos casos explican la mitad de la mejora del hold-out.
- **Con evaluación limpia,** excluyendo las 7 familias de diseño (§3.1):
  - CV: **+0,029 ± 0,082**, gana solo **8 de 15** (el criterio pedía 10).
  - Hold-out sin H204/H320: v4 +0,559 → A1 +0,610, que no empeora pero depende de 2–3 decisiones.

**Lo que queda:** A1 es una **mejora probable pero pequeña (≈ +0,03)** que no supera el umbral de garantías. Está montado y probado en `lab/candidato_A1/` (`mi_bot.py` + `predicciones.csv` de test + giro) por si decidís arriesgarlo. Cambia el 5 % de las decisiones del test y el 8,6 % de las del giro. El resto de cambios no mejora o empeora (§1).

## 1. Ablación: cada cambio por separado frente a la v4

**Montaje:**

- **Banco** (`lab/banco.py`): CV GroupKFold por documento, 5 particiones × semillas 0, 1 y 2 con *shuffle* real (comprobado que las semillas dan repartos distintos).
- **Datos:** solo se evalúan las 70 familias H fuera del hold-out (503 casos). Los no-H siempre entrenan.
- **Modelo:** el de la v4, reentrenado en cada partición: GradientBoosting de 200 árboles, profundidad 4, lr 0,05, subsample 0,8, hoja ≥ 20, peso H = 5 y UMBRAL = 0,18.
- **Reproducción de la v4:** al reentrenarla con su `entrenar.py` da el 100 % de decisiones idénticas en test y giro. `lab/construir_final.py` también la reproduce: 0 de 660 decisiones distintas.

**Columnas de la tabla:**

- **Dif.:** diferencia pareada media con la v4 en las 15 particiones ± desviación. **G/E:** particiones ganadas / empatadas.
- **k:** falsas alarmas que quita; **e:** escapes que añade (sumas de las 15 particiones). El cambio compensa si 4e < k.
- **Giro:** % marcado con un modelo entrenado en el 80 % (total, y con 3 / 4 oraciones).
- **Δ test / Δ giro:** % de decisiones que cambian frente a la v4.

| Cambio | Origen | CV | Dif. | G/E | k | e | 4e<k | Giro (3 / 4 or.) | Δ test | Δ giro |
|---|---|---|---|---|---|---|---|---|---|---|
| — (v4) | | +0,512 ± 0,153 | — | — | — | — | — | 57,7 % (45,2 / 87,7) | — | — |
| **A1** verificador ampliado | v5 | **+0,578** | **+0,067 ± 0,084** | **11 / 1** | 21 | −8 | sí | 55,9 % (43,9 / 84,6) | 4,3 % | 4,5 % |
| A2 regla «cuenta mal ⇒ p ≥ 0,5» | v5 | +0,537 | +0,025 ± 0,037 | 5 / 10 | 0 | −5 | sí | 58,2 % (45,8 / 87,7) | 0,0 % | 0,5 % |
| A3 features de Candela (cb_*) | v5 | +0,521 | +0,010 ± 0,053 | 5 / 6 | −4 | −3 | sí | 56,8 % (44,5 / 86,2) | 3,4 % | 2,7 % |
| A4 «plan → seguro» | v5 | +0,515 | +0,003 ± 0,044 | 5 / 6 | −1 | −1 | sí | 55,0 % (43,9 / 81,5) | 2,0 % | 3,6 % |
| A5 cambios de unidad | v5 | +0,507 | −0,004 ± 0,054 | 4 / 4 | −11 | −2 | no | 58,6 % (46,5 / 87,7) | 2,3 % | 1,8 % |
| A6 comparación con cifra no citada | v5 | +0,522 | +0,010 ± 0,060 | 9 / 2 | −3 | −3 | sí | 55,0 % (43,9 / 81,5) | 2,7 % | 4,5 % |
| A7 peor oración, UMBRAL_LARGAS (control) | v5 | +0,509 | −0,003 ± 0,012 | 2 / 9 | −2 | 0 | no | 62,7 % (54,2 / 83,1) | 0,2 % | **17,7 %** |
| B1 capa 1 explícita | fernando_v1 | +0,512 | 0,000 | 0 / 15 | 0 | 0 | no | 57,7 % (45,2 / 87,7) | 0,0 % | 0,0 % |
| B2 poda de features | fernando_v1 | +0,434 | −0,077 ± 0,100 | 3 / 0 | −29 | 8 | no | 60,9 % (51,0 / 84,6) | 7,3 % | 9,5 % |
| B3 sin n_oraciones, huecos / n | fernando_v1 | +0,511 | −0,001 ± 0,051 | 6 / 5 | −8 | −2 | no | 55,9 % (43,9 / 84,6) | 2,5 % | 6,4 % |
| B3b (B3 + topes de recuentos) | fernando_v1 | idéntica a B3 | | | | | | | | |
| B4 boosting monótono (HistGB) | fernando_v1 | +0,459 | −0,053 ± 0,086 | 4 / 0 | −47 | −1 | no | 66,8 % (56,1 / 92,3) | 8,2 % | 15,5 % |
| B4_ref (HistGB sin restricciones, control) | — | +0,516 | +0,004 ± 0,114 | 6 / 2 | −13 | −4 | sí | 59,1 % (47,1 / 87,7) | 4,1 % | 6,8 % |
| B5 pesos 4:1 sin reescalado | fernando_v1 | +0,402 | −0,110 ± 0,120 | 2 / 1 | 57 | 36 | no | 44,5 % (31,0 / 76,9) | 7,0 % | 13,2 % |
| C1 corrección de estadísticos de orden | nuestro | +0,485 | −0,026 ± 0,135 | 3 / 2 | −18 | 1 | no | 48,6 % (37,4 / 75,4) | 5,5 % | 16,4 % |
| C2 prior por oración + noisy-OR | nuestro | +0,513 | +0,001 ± 0,028 | 3 / 10 | 5 | 1 | sí | 48,2 % (38,7 / 70,8) | 0,2 % | **21,4 %** |

**Lectura:**

- **A1 es la única con una mejora clara y estable.**
- **A2 y A6 no pierden casi nunca, pero ganan pocas particiones:** A2 se activa en 23 casos H (21 alucinan) y A6 en muy pocos.
- **B1 no cambia ninguna decisión:** la v4 ya marca todos los casos con veredicto LLM negativo.
- **B5:** con pesos 4:1 el corte óptimo en validación sale ≈ 0,20 y no 0,5. Lo explico en §5.3.
- **Lo que empeora:**
  - B2: la poda quita 54 de 74 features, incluidas todas las de cifras y entidades.
  - B4: las restricciones monótonas (el control B4_ref queda a la par).
  - C1: la corrección de estadísticos de orden.
- **Los cambios pensados para el giro (A7, C1, C2)** mueven muchas decisiones del giro (16–21 %) sin ninguna mejora en H. No hay etiquetas del giro con las que justificarlos. Además, joseca_v5, que incluye A7 entre otras cosas, bajó la nota pública de 0,497 a 0,377.

## 2. Selección hacia delante (greedy) y combinaciones

**Criterios para aceptar un paso:**

1. Diferencia pareada media > 0 y gana en ≥ 10 de 15 particiones.
2. 4e < k frente al paso anterior.
3. Giro marcado entre el 40 y el 65 %, sin agrandar el salto entre 3 y 4 oraciones.
4. ≤ 10 % de decisiones cambiadas en el test.

**Paso 1:** de las 16 variantes, solo **A1** cumple los 4. Se aceptó en ese momento, antes de la auditoría (ver §3.1).

**Paso 2:** combinaciones A1 + X frente a A1 (`lab/resultados/greedy.md`):

| Combinación | Dif. frente a A1 | Ganadas / empatadas frente a A1 | Frente a v4 | ¿Pasa? |
|---|---|---|---|---|
| A1 + A6 | +0,026 ± 0,047 | 7 / 5 | +0,092 (13/15) | no (gana < 10) |
| A1 + A2 | +0,020 ± 0,035 | 4 / 11 (no pierde ninguna) | +0,087 (11/15) | no (gana < 10) |
| A1 + B3 | +0,006 ± 0,085 | 6 / 4 | +0,073 | no |
| A1 + A4 | +0,004 ± 0,031 | 6 / 7 | +0,070 | no |
| A1 + A3 | −0,011 ± 0,066 | 4 / 1 | +0,056 | no |
| *A1 + A2 + A6 (combinación con sentido)* | +0,042 ± 0,071 | 9 / 2 | **+0,108 (14/15)** | no (9 < 10 frente a A1) |

**La selección se para en A1.**

**Nota para quien decida:** A1 + A2 + A6 es la combinación con mejor CV (+0,620). No se elige porque frente a A1 gana 9 particiones y no 10, y no se ha evaluado en el hold-out: el plan solo permite dos evaluaciones ahí, v4 y la mejor configuración. Es la candidata natural para una v3, validada con otros datos.

## 3. Hold-out (20 % de las familias H, evaluado una sola vez)

18 familias y 128 casos, fijados al azar con la semilla 20261015 en `lab/holdout_familias.txt` **antes** de cualquier experimento. Solo se evaluaron v4 y A1:

| Bot | Puntuación | AUROC | Falsas alarmas | Escapes |
|---|---|---|---|---|
| v4 | +0,507 | 0,929 | 17 | 4 |
| **A1 (fernando_v2)** | **+0,582** | 0,937 | 16 | 3 |

- **Decisiones que cambian:** 4 de 128. Tres a mejor (H204::r9 deja de ser falsa alarma, H318::r2 deja de escaparse, H320::r8 deja de ser falsa alarma) y una a peor (H415::r1 pasa a falsa alarma).
- Con ~67 fieles en el hold-out, cada escape vale ≈ 0,06: la mejora de +0,075 equivale a un escape menos y una falsa alarma menos. Es una muestra muy pequeña.

### 3.1 Corrección tras la auditoría: familias usadas para diseñar las reglas

Los comentarios de las reglas que A1 copia de joseca_v5 citan frases de **H204** y **H320** (hold-out) y de **H210, H212, H213, H303 y H420** (CV). En esas familias, la mejora de A1 se debe en parte a que las reglas se escribieron viéndolas. Las excluyo de la evaluación, usando las mismas predicciones y sin ninguna evaluación nueva en el hold-out (`lab/resultados/sin_familias_de_diseno.txt`):

| Medida | Todas las familias | Sin familias de diseño |
|---|---|---|
| CV, diferencia pareada A1 − v4 | +0,067 ± 0,084 (11 G / 1 E / 3 P) | **+0,029 ± 0,082 (8 G / 3 E / 4 P)**; por semilla +0,025 / +0,001 / +0,062 |
| Hold-out v4 → A1 | +0,507 → +0,582 | +0,559 → +0,610 (112 casos; FP 14 → 15, FN 3 → 2) |

**Conclusión:** con evaluación limpia, A1 no cumple el criterio 1 (ganar en ≥ 10 de 15 particiones). Por la regla del encargo, **fernando_v2 = v4**.

## 4. Configuración final

**fernando_v2/mi_bot.py = joseca_v4/mi_bot.py**: las mismas 74 features, los mismos 200 árboles, UMBRAL 0,18 y los mismos scores.

**Único cambio: el docstring.**
- El de la v4 hablaba de "otros equipos" y de "regresión logística", cuando el modelo es gradient boosting.
- El nuevo describe lo que hace el código y por qué no se cambió nada.

**Candidato A1 (no entregado), en `lab/candidato_A1/`.** Es la v4 con estas reglas del verificador de joseca_v5, y el modelo reentrenado con todo el train por `lab/construir_final.py` (Python puro frente a sklearn: 1,6e-7):

| Regla | Dónde |
|---|---|
| «1.900 personas» no es un año | `extraer_numeros` |
| Fecha + plazo = fecha | `comprobar_plazos` |
| «X más al día» no es comparación | bloque D |
| Diferencia con verbo («superó en X a») | bloque F |
| Proporciones con palabra («triplica», «más de un tercio») | bloque G |

**Antes de entregarlo,** conviene parafrasear sus comentarios, que citan frases literales del train (hallazgo M4 de la auditoría).

## 5. Matemáticas comprobables a mano

### 5.1 La métrica

  **puntuación = 1 − (4·FN + FP)/N₀**

N₀ es el nº de respuestas fieles. Marcarlo todo da 0.

**Ejemplo con el hold-out:** 128 casos, de los que 67 son fieles. Para A1: 1 − (4·3 + 16)/67 = 1 − 28/67 = **0,582**. Para la v4: 1 − (4·4 + 17)/67 = 1 − 33/67 = **0,507**.

### 5.2 El corte 0,2

Marcar cuesta (1 − p) en falsas alarmas y no marcar cuesta 4p en escapes. Conviene marcar si 4p > 1 − p, es decir, si **p > 0,2**. La v4 corta en 0,18, que es el óptimo de su validación, y lo lleva a 0,5 con un reescalado lineal a trozos que conserva el orden.

### 5.3 Por qué B5 no lleva el corte a 0,5

En teoría, ponderar los positivos ×4 multiplica las *odds* por 4: q/(1 − q) = 4p/(1 − p), así que cortar en q = 0,5 equivale a p = 0,2. En la práctica, el boosting de 200 árboles de profundidad 4 se ajusta casi perfectamente al train: la AUROC dentro de muestra es de 0,999. Sus probabilidades en el entrenamiento quedan cerca de 0 y 1 tanto con pesos como sin ellos, y la reponderación apenas mueve la calibración fuera de muestra. Por eso el corte óptimo vuelve a ≈ 0,20 y B5 con 0,5 pierde 0,11. **El reescalado de la v4 es necesario con este modelo.**

### 5.4 La diferencia pareada y su error estándar

- dᵢ = puntuación_A1,i − puntuación_v4,i en cada partición i = 1..15, con las mismas particiones para los dos bots.
- **Media:** d̄ = 0,067. **Desviación:** s = 0,084.
- **Error estándar ingenuo:** s/√15 = 0,022, lo que daría t ≈ 3,1.
- **Las 15 particiones no son independientes.** Son 3 repeticiones de una CV de 5 sobre los mismos 503 casos, y los entrenamientos se solapan en el 75 %. Una cota prudente usa n_ef ≈ 5: s/√5 = 0,038, es decir, t ≈ 1,8. La mejora es **probable pero no concluyente**, y por eso hacía falta el hold-out.
- **Optimismo por selección:** se eligió la mejor de 16 variantes, así que la media está algo inflada. Además, las reglas de v5 se escribieron mirando errores de H, una fuga de diseño que la CV no puede quitar y que, como encontró la auditoría, también alcanzó al hold-out (§3.1).

### 5.5 C1 y C2 (no entran en el bot final)

- **C1, estadísticos de orden.** Para un mínimo sobre n elementos, F_min,n(m) = 1 − (1 − F(m))ⁿ. Para comparar con respuestas de n₀ elementos se busca el m′ que cumple F_min,n₀(m′) = F_min,n(m), es decir, F(m′) = 1 − (1 − F(m))^(n/n₀). Para un máximo, F(m′) = F(m)^(n/n₀). Con n ≤ n₀ no cambia nada, y está comprobado con un assert (`variantes/C1/test_c1.py`).
- **C2, prior por oración y noisy-OR.**
  - Δ = logit(π_train/n) − logit(π_train), con π_train = 0,494.
  - pᵢ′ = σ(logit pᵢ + Δ).
  - P = 1 − ∏(1 − pᵢ′), y se marca si P > 0,2.

## 6. Riesgos y recomendación

**Recomendación: entregar fernando_v2, que es la v4.**

- Es la opción con garantías: su nota pública conocida es 0,497 en test + giro.
- No hay ningún cambio que la mejore con evidencia limpia.

**Si queréis arriesgar, el candidato es `lab/candidato_A1/`:**
- **Ganancia esperada con evaluación limpia:** ≈ +0,03 en casos tipo H, entre +0,00 y +0,05 según la semilla. En el leaderboard, donde solo cambia el 5 % de las decisiones del test, el efecto sería de +0,01 a +0,03, muy por debajo del ruido de ±0,07. **No se distinguiría de la v4 en la nota pública.**
- **Riesgo bajo pero no nulo:** cambia 19 de 220 decisiones del giro, que no tiene etiquetas.

**Lo que NO recomiendo:**
- **joseca_v5, o añadir A7:** el control lo confirma y la entrega 6 bajó a 0,377.
- **C1 o C2 sin etiquetas del giro:** cambian el 16–21 % de las decisiones del giro sin ninguna evidencia a favor.
- **Las restricciones monótonas (B4), la poda (B2) o los pesos 4:1 sin reescalado (B5):** empeoran claramente.

**Riesgos de la propia v4:**

1. **Respuestas largas:** marca el 45 % de las de 3 oraciones y el 83–88 % de las de 4. Ninguna corrección probada lo arregla sin evidencia.
2. **UMBRAL = 0,18** se eligió con todo el train.
3. **Dependencia del juez LLM** en el giro, donde casi no rechaza oraciones.

## 7. Auditoría independiente y tests

**Tests** (`python -m pytest entregas/fernando_v2/lab/tests -q -p no:cacheprovider`): **9 passed**. Cubren:

- **CSV:** 660 ids de test y giro, formato, sin repetidos, punto decimal, y coincide con el bot.
- **Determinismo:** dos ejecuciones dan el mismo CSV.
- **0 excepciones** en `caracteristicas()` y `score()` en train, test y giro.
- **Imports:** solo stdlib, numpy y sklearn.
- **Sin ficheros ni red.**
- **Sin ids ni 6-gramas del test o del giro** que no estén también en el train.
- **Documentación:** sin menciones a otros equipos.
- **C1:** no se usa en el bot final.
- **Signos:** no aplica, el modelo no es logístico.

Los tests pasaron con el candidato A1 en su sitio y vuelven a pasar con la v4 final.

**Auditoría:** subagente opus/high nuevo, sin acceso a mi razonamiento. Re-ejecutó la CV, los tests, la reconstrucción del modelo y la comparación con el CSV.

### 7.1 Hallazgos de la auditoría y qué se hizo

| Gravedad | Hallazgo | Qué se hizo |
|---|---|---|
| **CRÍTICO** | El hold-out está contaminado: las reglas v6 de A1 citan frases de H204::r9 y H320::r8, que son justo 2 de las 4 decisiones que cambian | Recalculado sin esas familias (§3.1). **Cambia la decisión:** fernando_v2 = v4 |
| IMPORTANTE | La CV de A1 es optimista: 30 de las 53 unidades de ganancia vienen de 4 casos citados en los comentarios (H210, H212, H213, H420) | Recalculado sin las familias de diseño: +0,029 y 8 de 15 ganadas (§3.1) |
| IMPORTANTE | Optimismo por seleccionar entre 16+ variantes; las 15 particiones equivalen a unas 5 efectivas | Declarado (§5.4); A1 presentado como "probable, ≈ +0,03" |
| MENOR | UMBRAL 0,18 elegido por la v4 con todo el train, hold-out incluido | Declarado (§6). Afecta igual a las dos versiones |
| MENOR | Los criterios 3 y 4 usan distribuciones de test y giro sin etiquetas | Declarado: no usan etiquetas ni casos concretos, solo % marcado y % de decisiones cambiadas |
| MENOR | C1 (`calc_cdf.py`) y C2 (`calc_pi.py`) calculan constantes con todo el 80 % de H, una fuga leve dentro de la CV | Sin efecto: ni C1 ni C2 se eligieron, y las dos quedan ≤ v4 |
| MENOR | Los comentarios del candidato A1 citan frases literales del train | Pendiente si se entrega A1. fernando_v2 (= v4) no lleva esas reglas |
| MENOR | `banco.py` escribe `resultado.json` dentro de `lab/variantes/` | Es intencionado (resultados de la ablación). Re-ejecutar los regenera, idénticos porque el banco es determinista |

**Sin hallazgos:**
- No hay ids ni n-gramas de test o giro en el diff.
- Imports: solo stdlib.
- El modelo embebido reproduce sklearn (1,6e-7).
- 0 excepciones silenciosas en 2.367 casos.
- El CSV coincide con el bot.
- La CV está bien agrupada.
- `holdout.json` existe solo para v4 y A1.
- Todas las cifras declaradas se reproducen exactamente.

## 8. Consumo: subagentes lanzados

| # | Fase | Modelo / esfuerzo | Para qué | Tokens |
|---|---|---|---|---|
| 1 | 0 | sonnet / medium | Mapa de código v4, v5 y fernando_v1, y dónde está cada cambio del catálogo | 81 k |
| 2 | 0 | haiku / low | Reentrenar la v4 y comprobar que reproduce sus predicciones (100 %) | 52 k |
| 3 | 2 | sonnet / medium | A1 y A6 | 86 k |
| 4 | 2 | sonnet / medium | A2 y A4 | 66 k |
| 5 | 2 | sonnet / medium | A3 y A5 | 83 k |
| 6 | 2 | sonnet / medium | B1, B3, B3b y B5 | 68 k |
| 7 | 2 | sonnet / medium | C1, C2 y A7 | 84 k |
| 8 | 2 | sonnet / medium | B2, B4 y B4_ref | 73 k |
| 9 | 3 | sonnet / medium | Combinaciones A1 + {A2, A6, A3, A4, B3} | 66 k |
| 10 | 5 | opus / high | Auditoría independiente: fugas, reglas y re-ejecución | 91 k |

**Hecho en la sesión principal:**

- Banco, comparador, hold-out y `construir_final.py`.
- Combinación A1 + A2 + A6 (fusión con `git merge-file`).
- Decisiones, hold-out, construcción del bot final y este informe.

## 9. Cómo regenerar y entregar

```
# CV de una variante y comparación pareada (desde la raíz)
.venv\Scripts\python entregas\fernando_v2\lab\banco.py entregas\fernando_v2\lab\variantes\A1
.venv\Scripts\python entregas\fernando_v2\lab\comparar.py A1

# candidato A1 (entrena con todo el train y embebe los árboles)
cd entregas\fernando_v2\lab
..\..\..\.venv\Scripts\python construir_final.py variantes\A1 candidato_A1\mi_bot.py   # candidato A1, no es la entrega

# CSV test + giro (desde kit/) y tests
python generar_predicciones.py --bot ../entregas/fernando_v2/mi_bot.py --test test_publico.jsonl giro_publico.jsonl --salida ../entregas/fernando_v2/predicciones.csv
.venv\Scripts\python -m pytest entregas\fernando_v2\lab\tests -q -p no:cacheprovider
```

**Entrega oficial:** `entregas/gartic2.0/entrega_N/` con **solo** `mi_bot.py` y `predicciones.csv` (660 líneas + cabecera).

Como fernando_v2 = v4, entregarla da los mismos scores que la entrega de la v4. Solo tiene sentido como nueva entrega si queréis la documentación corregida.

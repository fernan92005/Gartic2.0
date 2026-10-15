# fernando_v1 — informe técnico

> Uso interno del equipo. **No se sube** al repositorio oficial (ni este informe ni `lab/`).
> Todas las cifras se han medido el jueves 15/10 entre las 13:25 y las 14:10 con
> `.venv` (Python 3.14, numpy 2.3.5, scikit-learn 1.7.2) y se pueden regenerar con los
> scripts de `lab/` (salidas guardadas en `lab/resultados/`).

## 1. Resumen

**fernando_v1** es una cascada:

- **Capa 1.** Si el juez LLM dice que alguna oración no está respaldada o está contradicha, el score es alto.
- **Capa 2.** Si no, decide un especialista de la "zona gris": una regresión logística entrenada con pesos 4:1 y siete señales booleanas o proporciones, agregadas por el peor caso.

`mi_bot.py` solo usa `math`, `re` y `unicodedata`: funciona sin numpy ni sklearn y no lee ningún fichero. Tarda 1,2 ms por caso y no ha dado ninguna excepción en 2.147 casos.

| Bot | **CV agrupada en H** (5 particiones × 3 semillas) | AUROC CV | FP / FN por partición | `--prefijo H` (dentro de muestra) | Train completo (dentro de muestra) |
|---|---|---|---|---|---|
| **fernando_v1** | **+0,362 ± 0,095** | 0,872 ± 0,027 | 30,4 / 3,1 | 0,398 · AUROC 0,882 | 0,432 · AUROC 0,933 |
| joseca_v3 | **+0,381 ± 0,141** | 0,905 ± 0,027 | 28,9 / 3,1 | 0,596 · 0,972 | 0,630 · 0,983 |
| joseca_v2 | +0,300 ± 0,094 | 0,878 ± 0,026 | 32,7 / 3,5 | 0,401 · 0,915 | 0,368 · 0,951 |
| joseca_v1 | +0,281 ± 0,124 | 0,879 ± 0,030 | 33,2 / 3,7 | 0,326 · 0,910 | 0,329 · 0,951 |
| candela_v1 | +0,278 ± 0,173 | 0,874 ± 0,036 | 30,5 / 4,4 | 0,503 · 0,955 | 0,573 · 0,980 |
| ejemplo_bot | −0,445 ± 0,214 | 0,675 ± 0,041 | 34,0 / 15,6 | −0,443 · 0,676 | 0,024 · 0,802 |

**Cómo leer la tabla:**

- **La cifra que cuenta es la CV agrupada.** La de `evaluar_local` está medida sobre los mismos casos con los que se entrenó el modelo embebido, así que es optimista.
- **Brecha dentro de muestra − CV:**

  | Bot | Brecha |
  |---|---|
  | fernando_v1 | 0,04 |
  | joseca_v2 | 0,10 |
  | candela_v1 | 0,22 |
  | joseca_v3 | 0,21 |

  Cuanto mayor es la brecha, más memoriza el bot el train. fernando_v1 es el que menos.
- **Diferencias dentro del ruido:** el error estándar de una media de 15 particiones es ≈ σ/√15 ≈ 0,025–0,036. joseca_v3 y fernando_v1 están a 0,019, es decir, son estadísticamente indistinguibles. fernando_v1 tiene menos varianza entre particiones (0,095 frente a 0,141).
- **Optimismo por elegir variante:** se probaron 30 variantes de modelo. Según la auditoría, elegir la mejor sobre las mismas particiones infla la media unos 0,03. La estimación honesta de fernando_v1 es, por tanto, **≈ +0,33**.
- **Optimismo de los compañeros:** los bots de los compañeros se reentrenaron por partición, pero conservan su umbral y sus reglas elegidos con todo el train, así que su CV también es algo optimista. La de joseca_v3, re-auditada con CV anidada, da **+0,374**.

## 2. Qué hace cada propuesta de los compañeros

Las cuatro se evaluaron con un subagente cada una. Los informes completos están en `scratchpad/` de la sesión. Aquí van los números que se han verificado.

### joseca_v1: regresión logística (Newton, L2) con 60 features, peso H = 5, l2 = 10, UMBRAL 0,15

- **Fuerte:**
  - Solo stdlib.
  - La validación por familias es correcta.
  - La idea de comprobar cuentas (`num_cuenta_correcta`, AUROC 0,402) funciona.
- **Riesgos:**
  - 16 de 36 features base tienen el peso con el signo cambiado (las tres `nli_contr_*`, `anadido_universal`, `num_inexplicados`, etc.).
  - 8 features tienen VIF infinito y 25 pares tienen |r| > 0,8.
  - 44 de 60 features tienen AUROC entre 0,45 y 0,55.
  - `n_oraciones` pesa −0,21: más oraciones ⇒ menos sospecha.
  - Marca el 70 % de los casos.
  - Su UMBRAL 0,15 no es el óptimo con sus propios scores (0,52 en H).

### joseca_v2: como v1 con atribución cruzada rehecha, plazos y horas, y peso H = 5, l2 = 3

Lo de peso H = 5 y l2 = 3 lo he verificado reproduciendo sus pesos exactos.

- **CV:** +0,300 ± 0,094, frente a +0,281 de v1. La mejora es menor que la desviación.
- **Lo que no aporta:**
  - `entidad_ajena`, el cambio principal, *baja* el score cuando el LLM dice "respaldada" (−0,61 en z).
  - `hora_inventada` vale 0 en todo el train y el test.
- **"Recuentos limitados a 2":** solo es verdad para `matiz_*`, `entidad_*` y `fecha_*`. `n_oraciones` (máximo 11) y los `num_*` no tienen tope.
- **Estrés:** una alucinación mezclada con 3 respuestas fieles solo se detecta el **32,7 %** de las veces (cada escape cuesta 4).

### joseca_v3: mismas 63 features que v2, pero gradient boosting (300 árboles, profundidad 3, lr 0,03) embebido como datos, UMBRAL 0,16

- **Es la mejor propuesta de los compañeros:** +0,381 ± 0,141 y AUROC 0,905.
- **Su "+0,42" es optimista:** elige configuración y corte sobre las mismas predicciones fuera de muestra. Con CV anidada da **+0,374 ± 0,014**, un sesgo de ≈ +0,045. Además, con su propio criterio el corte sería 0,19, no el 0,16 que lleva el bot.
- **Árboles embebidos:** reproducen sklearn con un error máximo de 6e-3 en train. El redondeo a 6 decimales cambia 1 decisión en el test.
- **De qué depende:**
  - Casi todo es la confianza del LLM: importancia 0,36 y ΔAUROC 0,087 al permutarla.
  - 28 de 63 features tienen importancia < 0,001.
  - `palabras_nuevas_frac` es la 3.ª más importante y es el atajo "paráfrasis ⇒ fiel".
- **Respuestas largas:** el AUROC baja a 0,795 con ≥2 oraciones (n = 23). En el estrés, la detección cae del 99 % al 72 % cuando la alucinación va diluida.

### candela_v1: boosting propio con 105 features, árboles embebidos, una regla `sci_relation_risk` y reescalado con `_SCI_THRESHOLD` = 0,146

- **Sobreajuste:** +0,503 en H dentro de muestra frente a **+0,278 ± 0,173** en CV.
- **No es reproducible:** no hay script de entrenamiento, y el bias embebido no corresponde a ningún subconjunto del train.
- **La regla:** acierta siempre (11 disparos en H, 0 fieles) y aporta +0,049 en CV. Es una regla escrita a mano y no podemos saber si se diseñó mirando esos casos.
- **Redundancia:** 124 pares con |r| > 0,8 y 66 VIF infinitos.
- **Fragilidad:** con respuestas fieles concatenadas, las falsas alarmas pasan del 40 % al 65 %.

**Cumplimiento:** las cuatro cumplen las reglas. No llevan ids del test y los 6-gramas que coinciden con el test están en comentarios cuyo texto también aparece en el train. Ninguna lanza excepciones y todas tardan entre 0,6 y 2 ms por caso.

## 3. Ideas de los compañeros reimplementadas y descartadas

| Idea | De quién | Qué he hecho | Número |
|---|---|---|---|
| Comprobar si la cifra sale de una cuenta exacta (suma, diferencia, %, unidad) y usarlo como **respaldo** | joseca (`num_cuenta_correcta`, `num_derivado_local`) | Reimplementado a mi manera: `num_derivado_local`, solo con las cifras de las fuentes alineadas o citadas, y exigiendo que **todas** las cifras nuevas de la oración salgan de cuentas | Es la feature con más peso (−1,14). Exigir "todas" en vez de "alguna" subió la CV de +0,322 a **+0,362** |
| Matiz condicional de la fuente que desaparece | joseca (`matiz_perdido_fuente_condicional`, AUROC 0,606 en zona gris) | Reimplementado: `modal_perdido_citada`, sobre la fuente que cita el LLM y solo en su oración más parecida | AUROC 0,582 en zona gris, peso +0,49 |
| Peor oración, nunca la media | las dos | Aplicado a todo | — |
| `palabras_nuevas` (proporción de palabras que no están en las fuentes) | joseca v1–v3, candela (`texto_novedad`) | **Descartado.** Mi equivalente, `prop_raices_nuevas`, tiene AUROC 0,369: más palabras nuevas ⇒ fiel (la paráfrasis). Es el atajo del que avisa EJEMPLOS.md, y la restricción de signo lo elimina | — |
| Solape con la fuente citada | candela (`texto_cobertura`) | **Descartado.** `cob_citada` tiene AUROC 0,598 en zona gris en el sentido "copia ⇒ alucina": otro atajo léxico | — |
| Atribución cruzada (cifra de otra entidad) | joseca v2 (`entidad_ajena`) | Probado con dos implementaciones propias: `num_cruzada` (fuente alineada por NLI y similitud) y una ligadura entidad–cifra | AUROC 0,478 y 0,514: no discriminan, igual que en joseca (precisión 0,35). Descartado |
| Regla de relaciones numéricas | candela | No reimplementada: es una regla escrita a mano y no se puede auditar si se hizo mirando casos | — |
| Gradient boosting | joseca v3, candela | Comparado (HGB monótono) | HGB profundidad 2: +0,341 ± 0,135, frente a logística +0,362 ± 0,095. Pierde en media y en varianza |

## 4. Qué es nuevo en fernando_v1

1. **Cascada explícita con un especialista entrenado solo en la zona gris.** La capa 1 resuelve el 20 % de H con una regla medida: 92 % de alucinaciones.
2. **Ninguna feature crece con la longitud.** Todo son booleanos o proporciones agregados por el peor caso. No hay `n_oraciones`, ni número de cifras, ni número de fuentes.
3. **Signos obligatorios.** Cada feature tiene un signo esperado declarado en `entrenar.SIGNO`. Las que salen con el signo cambiado se eliminan y el modelo se reajusta. Los 7 pesos finales son coherentes, y hay un test que lo vigila.
4. **Verificador aritmético local y estricto:**
   - Solo hace cuentas con las fuentes alineadas.
   - Incluye "X % de Y", "el resto" y diferencias de meses entre fechas.
   - Tolerancia del 0,2 %, y redondeo a entero solo para porcentajes ≤ 100.
   - Se quitaron ×2, ÷2 y ×100, porque convertían en "derivable" casi cualquier entero.
5. **Selección de modelo con la regla de un error estándar.** Además, el reescalado del corte se decide con **CV anidada**, no sobre las mismas predicciones.
6. **Léxico auditable.** Cada patrón de las listas aparece al menos una vez en el train, y un test lo comprueba. Se quitaron 28 patrones genéricos sin soporte.
7. **Coeficientes embebidos sin pickle.** No depende de la versión de sklearn del jurado.

## 5. Estudio de features (H, zona gris = 503 casos, 179 alucinan)

Se genera con `lab/estudio_features.py` y la salida está en `lab/resultados/features.txt`.

- **Cómo leer el AUROC:** más de 0,5 significa "más alto ⇒ alucina"; menos de 0,5 significa "más alto ⇒ fiel".
- **Columnas de pesos:** "peso" va sobre la feature estandarizada; "peso/escala" es el efecto por unidad cruda.

| feature | signo esperado | AUROC H | AUROC zona gris | VIF (todas) | peso | peso/escala |
|---|---|---|---|---|---|---|
| **llm_conf_min** | − | 0,286 | 0,287 | 1,2 | **−0,836** | −19,97 |
| **num_derivado_local** | − | 0,366 | 0,358 | 2,6 | **−1,137** | −2,79 |
| **nli_mejor_min_frag** | − | 0,328 | 0,379 | 4,5 | **−0,367** | −0,96 |
| **frag_sin_respaldo** | + | 0,627 | 0,583 | 4,0 | **+0,128** | +0,38 |
| **modal_perdido_citada** | + | 0,573 | 0,582 | 2,6 | **+0,487** | +1,71 |
| **matiz_alcance_perdido** | + | 0,549 | 0,550 | 4,8 | **+0,341** | +1,15 |
| **num_ausente** | + | 0,526 | 0,542 | 1,5 | **+0,295** | +1,21 |
| nli_citada_min | − | 0,351 | 0,409 | 1,9 | fuera | |
| nli_comb_min_frag | − | 0,339 | 0,394 | 8,8 | fuera (r = 0,84 con nli_mejor) | |
| nli_todas_min_frag / _orac | − | 0,383 / 0,395 | 0,443 / 0,481 | 2,8 / 2,5 | fuera | |
| num_prop_fuera | + | 0,382 | 0,400 | 3,2 | fuera (signo invertido) | |
| prop_raices_nuevas | + | 0,373 | 0,369 | 1,1 | fuera (signo invertido: atajo paráfrasis) | |
| sim_min_frag | − | 0,563 | 0,561 | 1,3 | fuera (signo invertido) | |
| nli_contra_max | + | 0,473 | 0,419 | 1,4 | fuera (signo invertido) | |
| alcance_perdido_citada | + | 0,537 | 0,541 | 4,8 | fuera (r = 0,87 con matiz_alcance) | |
| matiz_modal_perdido | + | 0,570 | 0,576 | 2,4 | fuera (la versión "citada" es mejor) | |
| num_no_derivable_local | + | 0,522 | 0,550 | 3,0 | fuera (L2 la deja en ≈0 y el filtro la quita) | |
| num_cruzada, num_cruzada_llm, matiz_num_perdido, universal_anadido, causal_anadido, nombre_nuevo, llm_null, llm_sin_fuente | + | 0,48–0,52 | 0,47–0,51 | ≈1 | fuera (|AUROC − 0,5| < 0,03) | |
| comparativa_anadida | + | 0,461 | 0,473 | 1,2 | fuera (signo invertido: las cuentas fieles comparan) | |

- **Pares con |r| > 0,8 en la zona gris:** nli_mejor ~ nli_comb (0,838), nli_comb ~ frag_sin_respaldo (−0,832) y matiz_alcance ~ alcance_citada (0,870). En cada par se queda la de mayor |AUROC − 0,5|.
- **En el modelo final:** |r| máximo de 0,697 y **VIF entre 1,04 y 2,02**, sin colinealidad.
- **Las features léxicas puras discriminan poco:** universales, causales y comparativas añadidas tienen AUROC 0,46–0,52. En H, las alucinaciones de la zona gris son sobre todo distorsiones de cifras (95) y sobregeneralizaciones (64), y casi nunca añaden un "porque" o un "todos". Las listas se quedan en el bot porque no cuestan nada, pero el modelo no las usa.

## 6. Las matemáticas, paso a paso

### 6.1 La métrica

Sea N₀ el nº de casos fieles, FN las alucinaciones con score < 0,5 y FP los fieles con score ≥ 0,5:

  **puntuación = 1 − (4·FN + FP) / N₀**

- Marcarlo todo da FN = 0 y FP = N₀, así que puntuación = 0.
- **Comprobación con nuestra CV:** cada partición tiene ≈ 334/5 = 66,8 fieles, con FP = 30,4 y FN = 3,1 de media. Entonces 1 − (4·3,1 + 30,4)/66,8 = 1 − 42,8/66,8 = **0,359**. La media de las puntuaciones por partición es 0,362; no coincide exactamente porque la media de cocientes no es el cociente de medias.

### 6.2 Por qué el corte óptimo es p = 0,2

Para un caso con probabilidad real p de alucinar:

- Si **no** lo marcamos, el coste esperado es 4·p (escape).
- Si lo marcamos, el coste esperado es 1·(1 − p) (falsa alarma).

Conviene marcar cuando 4p > 1 − p, es decir, cuando 5p > 1, es decir, cuando **p > 1/5 = 0,2**.

### 6.3 Por qué entrenar con pesos 4:1 lleva ese corte a 0,5

La regresión logística con pesos w₁ = 4 (positivos) y w₀ = 1 (negativos) maximiza la verosimilitud ponderada. Su óptimo en población cumple q(x) = 4·p(x) / (4·p(x) + 1 − p(x)), donde q es la probabilidad que da el modelo y p la real. Entonces:

  q ≥ 0,5 ⇔ 4p ≥ ½·(4p + 1 − p) ⇔ 8p ≥ 4p + 1 − p ⇔ **4p ≥ 1 − p** ⇔ **p ≥ 0,2**.

Dicho de otra forma, multiplicar por 4 el peso de los positivos multiplica las *odds* por 4: q/(1 − q) = 4·p/(1 − p). Cortar en q = 0,5 (odds 1) equivale a cortar en p/(1 − p) = 1/4, es decir, en p = 0,2.

**Comprobación empírica:**

- El mejor corte de `evaluar_local --prefijo H` es **0,471**, cerca de 0,5 como predice la teoría.
- En validación fuera de muestra el óptimo sale 0,58, pero no es estable: la CV anidada lo descarta (§6.4).

### 6.4 Reescalado: probado y descartado

Reescalado lineal a trozos con umbral t, que lleva t a 0,5 y conserva el orden (el AUROC no cambia):

  s' = 0,5·s/t si s < t;  s' = 0,5 + 0,5·(s − t)/(1 − t) si s ≥ t

**Cómo se decide si aplicarlo:** en cada una de las 15 particiones externas se elige t con una CV interna (GroupKFold 5) **solo sobre su entrenamiento**, y se mide en la partición externa (`lab/umbral_anidado.py`).

| | Resultado |
|---|---|
| Corte fijo 0,5 | **+0,362 ± 0,095** |
| Umbral elegido anidado | +0,359 ± 0,092 |
| Umbrales elegidos | 0,50–0,59 |
| Particiones | mejora en 2 de 15, empeora en 3 |

**Conclusión:** el reescalado no aporta y el bot corta en 0,5 (`MODELO["umbral"] = 0.5`). `entrenar.py` solo lo activaría si la CV anidada ganara más de 0,01.

### 6.5 La capa 1

En H, 128 casos tienen alguna oración "no_respaldada" o "contradicha", y el **92,2 %** alucina. Por tramo de confianza del LLM:

| Confianza del LLM | Casos | % alucina |
|---|---|---|
| < 0,7 | 3 | 67 % |
| 0,7–0,85 | 5 | 100 % |
| 0,85–0,95 | 35 | 91 % |
| ≥ 0,95 | 85 | 93 % |

En todos los tramos la proporción supera 0,2, así que lo óptimo es marcar **siempre**. Por eso la confianza no cambia la decisión y solo ordena: score = 0,85 + 0,10·confianza ∈ [0,85; 0,95].

**Coste:** 10 falsas alarmas en H, las mismas que tienen todos los compañeros, y 0 escapes.

### 6.6 El especialista

  z = b + Σₖ cₖ·(xₖ − μₖ)/σₖ,   p = 1/(1 + e^(−z))

Con b = 0,4596 y los cₖ, μₖ y σₖ embebidos en `MODELO` (ver la tabla de §5). Se entrena solo con los casos H de la zona gris de las particiones de entrenamiento, con peso 4 en las alucinaciones y C = 0,3 (L2).

**Ejemplo para comprobar a mano:** un caso con todas las features en su media da z = b = 0,46 y p = 0,61, así que se marca. Que el caso medio de la zona gris (35,6 % de alucinaciones, por encima de 0,2) se marque es lo que pide el coste 4:1.

### 6.7 AUROC, Pearson, VIF y error estándar

- **AUROC** = P(score de una alucinación al azar > score de un fiel al azar), contando los empates como ½. Es la U de Mann–Whitney dividida entre n₁·n₀.
- **Pearson:** r = cov(x, y)/(σₓ·σᵧ).
- **VIF de la feature j:** se regresa xⱼ (estandarizada) sobre las demás con intercepto, se obtiene R²ⱼ y VIF = 1/(1 − R²ⱼ). VIF = 1 significa sin colinealidad; VIF > 10 es colinealidad grave.
- **Regla de un error estándar:** EE = σ_particiones/√15. Entre las variantes con media ≥ mejor − EE se elige la más simple (logística) y, dentro de ella, la de mayor media.

| Variante | CV |
|---|---|
| Mejor media: logreg C = 0,3 L2 solo H | +0,362 ± 0,095 (EE 0,025), elegida |
| HGB monótono, profundidad 2 | +0,341 ± 0,135 |
| HGB monótono, profundidad 3 | +0,339 ± 0,133 |
| Logística L1, C = 0,1 | +0,355 ± 0,109 |

**Añadir los casos no-H al entrenamiento siempre empeora:**

| Peso de los no-H | CV |
|---|---|
| 0,3 | +0,297 ± 0,146 |
| 1 | +0,183 ± 0,190 |

El train fácil tiene otra prevalencia (11 % frente a 36 % en la zona gris) y descoloca la calibración.

## 7. Estrés para el giro

Se genera con `lab/estres_giro.py` y las salidas están en `lab/resultados/estres_*.txt`. El OOF son las predicciones fuera de muestra de la CV agrupada. "Concatenadas" significa juntar 2–4 respuestas H del mismo documento: con 0 alucinaciones el resultado sigue siendo fiel, y con 1 alucina.

| Bot | OOF 1 oración (n = 608) | OOF ≥2 oraciones (n = 23) | Fieles ×2 / ×3 / ×4 marcadas | 1 alucinación + fieles ×2 / ×3 / ×4 detectada | % marcado train-H → test |
|---|---|---|---|---|---|
| **fernando_v1** | +0,368 · AUROC 0,878 | +0,062 · 0,571 | 83 / 96 / 100 % | **97 / 99 / 100 %** | 67,8 → 60,9 |
| joseca_v3 | +0,374 · 0,909 | +0,312 · 0,804 | 50 / 34 / 28 % | 90 / 81 / 82 % | 66,1 → 58,6 |
| joseca_v2 | +0,261 · 0,878 | +0,562 · 0,812 | 40 / 22 / 8 % | 74 / 59 / **42 %** | 70,8 → 65,2 |
| joseca_v1 | +0,267 · 0,877 | +0,438 · 0,866 | 46 / 32 / 16 % | 76 / 61 / 49 % | 72,4 → 67,0 |
| candela_v1 | +0,289 · 0,879 | +0,125 · 0,732 | 65 / 63 / 67 % | 98 / 94 / 93 % | 67,0 → 57,7 |

**Recall fuera de muestra por tipo, fernando_v1:**

| Tipo | Recall |
|---|---|
| Distorsión | 0,97 |
| Sobregeneralización | 0,88 |
| No respaldado | 0,98 |
| Contradicción | 1,00 |

**Lectura honesta:**

- **Respuestas largas fieles:** fernando_v1 las marca casi todas. Con más oraciones, el mínimo de la confianza del LLM y del NLI baja, y el respaldo por cuentas exige que *todas* las oraciones lo tengan. Es el precio de agregar por el peor caso, que es lo que pide el diseño.
- **Coste esperado en el estrés de 3 respuestas,** con coste = %FP + 4·%escapes:

  | Bot | Coste |
  |---|---|
  | **fernando_v1** | 0,96 + 4·0,01 = **1,00** |
  | joseca_v3 | 0,34 + 4·0,19 = **1,11** |
  | joseca_v2 | 0,22 + 4·0,41 = **1,85** |

  Como un escape cuesta 4, dejar pasar alucinaciones diluidas es peor que dar falsas alarmas.
- **En la práctica,** si el giro trae muchas respuestas de 3 o más oraciones, fernando_v1 se acercará a "marcarlo todo" en esa parte (puntuación ≈ 0) y los de joseca, que dejan pasar alucinaciones, a valores negativos.
- **Deriva:** el test marca 7 puntos menos que train-H (60,9 % frente a 67,8 %) y los deciles están desplazados hacia abajo. Pasa igual en todos los bots, así que es una característica del test y no una alarma específica.
- **Muestra pequeña:** en H solo hay 23 casos de ≥2 oraciones. Las cifras de esa columna tienen mucho ruido.

## 8. Limitaciones y casos en los que falla (ejemplos reales del train, OOF)

1. **Cuentas mal hechas que el verificador da por buenas.** El verificador comprueba que *alguna* pareja de cifras de las fuentes produzca la cifra, pero no que sean los operandos de los que habla la frase:
   - H314::r2 (score 0,09, escape): «…el 30 de junio de 2027, y **nueve meses** después en el resto del país». La fuente dice 31 de diciembre, que son 6 meses, pero 9 sale de otra resta.
   - H305::r2 (0,12): «812 eran mujeres, **más del 70 %** del total». 812/1.326 = 61 %, pero 70 coincide con otra cuenta.
   - H422::r9 (0,13): «perdió 2.410 habitantes, **un 18 %**». 2.410/15.820 = 15,2 %.

   **Siguiente paso:** ligar la cuenta a las cifras que aparecen en la propia frase.
2. **Cuentas bien hechas con el juez LLM dudando:**
   - H215::r1 (1,00, falsa alarma): «319 parados menos que en septiembre de 2025». 4.105 − 3.786 = 319, pero 3.786 está en una fuente que ni el LLM ni el NLI alinean. Con confianza 0,65, la capa 2 marca.
   - H309::r1: «ocho menos que en 2024» (26 − 18). Confianza 0,70.
3. **Dependencia del juez LLM:** `llm_conf_min` es la feature más fuerte, con −20 por unidad cruda. Una bajada de 0,05 en la confianza sube unos +0,85 logits. Si en los textos técnicos o legales del giro el LLM es menos seguro, el bot marcará más.
4. **Respuestas largas:** ver §7.
5. **Sobregeneralizaciones sin marcador léxico:** el recall es 0,88, el más bajo por tipo. Cuando la respuesta omite un restrictor que no está en las listas («de grado superior», «con mayor riesgo»), no hay señal.

## 9. Tests y auditoría

### Tests

`python -m pytest entregas/fernando_v1/lab/tests -q` da **15 passed** en 10 s. Cubren:

- **Formato del CSV:** cabecera, punto decimal, sin ";", sin ids repetidos y todos los ids del test (y del giro si existe).
- **Coherencia:** el CSV coincide con el bot.
- **Scores:** en [0, 1] para todo train y test, con 0 excepciones.
- **Determinismo:** dos ejecuciones dan el mismo CSV.
- **Portabilidad:** funciona desde la raíz.
- **Imports:** análisis con `ast` (solo stdlib, numpy y sklearn).
- **Sin ficheros ni red.**
- **Sin caracteres de control.** Este test nació de un fallo real: un `\b` se coló como carácter de retroceso y desactivó la regex de fechas.
- **Contra el test:** sin ids `t0…` y sin 6-gramas del test.
- **Signos del modelo** coherentes.
- **Léxico con soporte en el train.**
- **9 casos sintéticos de robustez:** `llm=None`, veredicto null, sin fragmentos, sin cifras, una sola fuente, 10 oraciones, texto legal con muchos "no"/"si", caso vacío y señales rotas. Ninguno lanza excepción y el contador queda en 0.
- **Null:** un veredicto null da score ≥ 0,5.
- **Rendimiento:** 1,4 ms por caso, con un límite de 50.
- **Regresión:** CV ≥ 0,30.

### Auditoría independiente (subagente nuevo, sin acceso a mi razonamiento)

- **Cifras:** re-ejecutó `validar_cv.py`, los tests y `evaluar_local`. Todas coinciden con las declaradas al momento de la auditoría.
- **Hallazgos y qué se hizo con cada uno:**

| Hallazgo | Gravedad | Qué se hizo |
|---|---|---|
| Respuestas largas ⇒ casi todo marcado | CRÍTICO | **Documentado**, no corregido: choca con el mandato de agregar por el peor caso, y el análisis de coste de §7 muestra que es la opción menos mala de las medidas. Siguiente paso: features por proporción de oraciones y entrenar con concatenaciones sintéticas |
| `modal_perdido_citada` salta con cualquier "salvo/puede" de un texto legal largo | CRÍTICO | **Corregido:** ahora solo mira la oración de la fuente más parecida a la respuesta |
| Optimismo por elegir entre 30 variantes (≈ 0,03) | IMPORTANTE | **Declarado** en §1 (estimación honesta ≈ +0,33). Selección cambiada a la regla de 1 EE |
| Patrones sin ninguna aparición en el train (p. ej. "previsiblemente", "provisional") | IMPORTANTE | **Corregido:** se quitaron los 28 patrones sin soporte y se añadió un test. Eran español genérico escrito a priori, no sacado del test, pero así es auditable |
| El diseño se hizo mirando todo H; `evaluar_local` es dentro de muestra | IMPORTANTE | Aceptado y declarado: la cifra buena es la CV |
| La CV de los compañeros conserva sus umbrales de train | IMPORTANTE | Declarado en §1 |
| El umbral se elegía sobre el OOF | MENOR | **Corregido:** ahora se decide con CV anidada |
| NEUTRO = medias del train; formatos raros ⇒ 0,9 | MENOR | Sin impacto: 0 excepciones. En el giro hay que comprobar que el contador quede en 0 |

### Fallos de mis propias herramientas encontrados por los subagentes (corregidos)

- `GroupKFold` sin `shuffle` daba la misma partición con las 3 semillas: el "±" medía una sola CV.
- La clave de selección de variante estaba invertida.
- liblinear sin `random_state` no era determinista.

## 10. Cómo regenerar y cómo entregar

```
# (opcional) reentrenar y reescribir MODELO en mi_bot.py
.venv\Scripts\python entregas/fernando_v1/lab/entrenar.py

# validar
.venv\Scripts\python entregas/fernando_v1/lab/validar_cv.py --guardar-oof
.venv\Scripts\python entregas/fernando_v1/lab/estres_giro.py --bot entregas/fernando_v1/mi_bot.py
.venv\Scripts\python -m pytest entregas/fernando_v1/lab/tests -q -p no:cacheprovider

# CSV antes del giro (desde kit/)
python generar_predicciones.py --bot ../entregas/fernando_v1/mi_bot.py --test test_publico.jsonl --salida ../entregas/fernando_v1/predicciones.csv

# CSV desde las 15:30 (test + giro en UN solo CSV)
python generar_predicciones.py --bot ../entregas/fernando_v1/mi_bot.py --test test_publico.jsonl giro_publico.jsonl --salida ../entregas/fernando_v1/predicciones.csv
```

Tras generar el CSV del giro:

1. Vuelve a pasar los tests. `test_csv_formato` exige que estén también los ids del giro, y `test_sin_ids_ni_frases_del_test` revisa también el giro.
2. Mira el contador de excepciones con el giro (`estres_giro.py` lo imprime). Debe ser 0.
3. **No reentrenes mirando el giro:** no tiene etiquetas.

**Entrega oficial:**

- Carpeta: `entregas/gartic2.0/entrega_N/`, con **solo** `mi_bot.py` y `predicciones.csv`.
- **Nunca** subas `lab/`, `INFORME.md` ni `resultados/`.
- `mi_bot.py` es autocontenido: solo usa stdlib y no lee nada del disco. No hace falta copiar nada más.
- Genera el CSV siempre con el script y no lo abras con Excel.

**Entorno:** el Control de aplicaciones de Windows bloqueaba las DLL de numpy 2.5.4 y de scipy 1.18.1. Dejé en `.venv` numpy 2.3.5, scipy 1.16.3 y scikit-learn 1.7.2. Solo afecta a `lab/`; `mi_bot.py` no necesita numpy.

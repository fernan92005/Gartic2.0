# Fase 4 — diagnóstico de entregas/joseca_v1

**No se modifica ningún bot, parámetro ni umbral. No se genera una entrega.**

## Versión y procedencia

- Objetivo: `entregas/joseca_v1/mi_bot.py`, SHA-256 `edd2cfda1bb3e5d6c811ced2c1e0c0b3553aeaefd2080256e921fd8ccf36a510`.
- El archivo coincide byte por byte con `b6e5a97:kit/mi_bot.py`. El mensaje de ese commit registra entrega 1 con score público 0,353 y AUROC 0,905. Esto identifica la versión dentro del repositorio; no sustituye un recibo del servidor ni permite recalcular métricas públicas sin etiquetas.
- Es una regresión logística de 60 características, no E4. Se recuperó de manera única peso H=5 y L2=10: entrenar con todos los casos y redondear sus parámetros a seis decimales reproduce exactamente MEDIAS/ESCALAS/PESOS/SESGO embebidos. La comparación de nueve configuraciones fue una auditoría de identidad, no selección de un modelo nuevo.
- E4 de referencia: `entregas/candela_v1/mi_bot.py`, SHA-256 `cc27f79f220da13664361a30d8426cd0dcbbbcfe70a825972b70a8e5437c53e3`; coincide con el candidato corregido de fase 3. Se analiza por separado.
- En el checkout actual no existen `kit/mi_bot.py`, los informes anteriores ni los manifiestos de particiones originales. No se crean ni se recuperan en rutas de entrega.

## Reconstrucción y límites de OOF

Se reconstruyó el algoritmo histórico: componentes conectados por ID base, pregunta normalizada, fuentes idénticas o solape >=80% del conjunto menor; semilla 20261015, StratifiedGroupKFold de cuatro particiones. Resultan 337 grupos, desarrollo de 1.280 casos (473 H) y reserva histórica de 427 (158 H). Desarrollo usa otros cuatro folds con semilla 20261016. No hay grupos compartidos. La coincidencia de tamaños y del score H OOF del HGB histórico sirve de comprobación, pero no sustituye una comparación bit a bit con el manifiesto original ausente.

Para Joseca se reentrenan estándarización, coeficientes e intercepto en cada entrenamiento. No se usan sus pesos congelados de todo train para producir OOF. Se fijan las mismas 60 características y regularización ya identificadas. La calibración anidada genera tres particiones internas por fold externo, semilla 20261017 + fold. No se seleccionan hiperparámetros nuevos.

Las características y la configuración actuales se eligieron originalmente con train: aunque los coeficientes OOF no ven su grupo, sigue existiendo sesgo de selección de arquitectura/variables. La evaluación no es una nueva prueba ciega.

## Cuatro contextos que no se deben mezclar

| Arquitectura / dominio | N | Score | AUROC | FP | FN | Coste | Matriz [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---:|---|
| Joseca OOF desarrollo | 1280 | 0.3372 | 0.9342 | 333 | 18 | 405 | [[278,333],[18,651]] |
| Joseca pesos congelados / train (in-sample) | 1707 | 0.3293 | 0.9511 | 448 | 25 | 548 | [[369,448],[25,865]] |
| Joseca pesos congelados / reserva histórica (también entrenada) | 427 | 0.3252 | 0.9569 | 119 | 5 | 139 | [[87,119],[5,216]] |

La reserva histórica ya forma parte del entrenamiento del Joseca publicado (ver reproducción exacta de sus parámetros con todo train), por lo que sus métricas congeladas NO son independientes. Para Candela, el HGB sí procedía de desarrollo, pero esa reserva fue inspeccionada en fases anteriores: aquí sus cifras congeladas son solo referencia histórica, no selección. El leaderboard 0,353/0,905 corresponde a otro conjunto sin etiquetas locales; no se compara como si fueran los mismos ejemplos.

## Reproducción adicional del protocolo original del autor

Se fija peso H=5/L2=10 y se repiten las semillas originales 0,1,2 sobre familias H, sin buscar configuración ni corte. Todo no-H entrena siempre, así que este protocolo produce OOF solamente para H y no se le atribuye AUROC OOF no-H.

| Semilla | N H | Score al corte 0,15 | AUROC | FP | FN |
|---|---:|---:|---:|---:|---:|
| 0 | 631 | 0.2635 | 0.8770 | 170 | 19 |
| 1 | 631 | 0.2874 | 0.8760 | 170 | 17 |
| 2 | 631 | 0.2754 | 0.8756 | 166 | 19 |

## Archivos

- `diagnostico_tipos.md` y `metricas_por_tipo.csv`: ranking, TP/FN/recall/coste por tipo y dominio; FP separados.
- `diagnostico_nli_minimax.md` y `distribuciones_senales.csv`: distribuciones, desacuerdos y errores de jueces.
- `diagnostico_e4.md` y `contribucion_e4.csv`: contribución del E4 de Candela, claramente separado de Joseca.
- `falsos_positivos.md`: cohortes y contribuciones del Joseca.
- `calibracion_y_umbral.md`, `curva_coste_oof.csv`, `curva_coste.svg`: umbral actual y calibración interna, sin modificaciones.
- `prioridades_mejora.md`: propuestas futuras; ninguna implementada.
- `predicciones_oof.json/.csv`, `folds_detalle.json`, `particiones_reconstruidas.json`: datos para auditar cada tabla.
- `procedencia.json`, `hashes_antes.json`, `integridad.json`: identificación e integridad.

## Ejecutar

Desde la raíz del repositorio, con NumPy y scikit-learn instalados (versiones exactas en resumen_metricas.json → environment). En esta máquina se usó /private/tmp/guardian-venv/bin/python; puede sustituirse python por la ruta de ese entorno:

```bash
python kit/experimentos/fase4_diagnostico/ejecutar_diagnostico.py
```

verificar_diagnostico.py comprueba las métricas con el evaluador oficial, reconstruye cada score con sus coeficientes de fold y verifica separación de grupos externa/interna, cobertura una vez por caso y hashes. Los resultados se guardan en verificacion.json.

No se llama al main de entrenar.py, que podría sobrescribir pesos. Se importan sus funciones puras de entrenamiento. Los cachés numéricos del diagnóstico no son archivos necesarios para la inferencia del bot. Los datos oficiales y ambos bots se verifican por hash al finalizar. No se consulta ninguna API ni se usan modelos de IA adicionales.

# Experimento B — matices, cobertura y alcance

**Base: entregas/joseca_v3/mi_bot.py. La versión original y su entrenador se conservan intactos; copia byte a byte en baseline_v3/.**

SHA-256 v3: `ff65cd4dcbd63cacce311ada7e28980a5879e9ceb00b9d2962cc963493c95de9`. El reentrenamiento B0 usa la configuración original y reproduce 64 de 300 árboles exactamente al redondeo original. No hay igualdad completa de árboles; diferencia máxima de probabilidades congeladas/scikit-learn: 0.0062. No cambia ninguna decisión al corte original en train. La causa de las diferencias numéricas/empates no se ha identificado con certeza; el entorno original no está fijado. La copia congelada conserva la versión exacta. B0 OOF reentrena la misma arquitectura, no reutiliza los árboles congelados. Esta comprobación es sobre entrenamiento, no OOF.

## Configuraciones

| ID | Nuevas características |
|---|---|
| B0 | ninguna |
| B1 | matices en cláusulas relevantes |
| B2 | equivalencias fuertes, pérdidas e incertidumbre |
| B3 | cobertura individual/unión y complementariedad |
| B4 | ampliación de sujeto singular a colectivo |
| B5 | B1+B2 |
| B6 | B1+B2+B3 |
| B7 | B1+B2+B3+B4 |

Las originales se conservan. Cuatro ablaciones auxiliares verifican normalización B1, aporte B3 respecto a NLI y sustitución explícita de matices. No se busca una nueva configuración GB ni se reutilizan árboles v3 con variables adicionales. No se aplica ninguna regla que reduzca automáticamente el riesgo.

## Resultado y recomendación

**No se recomienda sustituir v3. No se genera mi_bot_candidato_b.py porque ninguna variante supera los controles de estabilidad.**

B0 OOF agrupado/anidado H: score 0.3494, AUROC 0.8989, FP 130, FN 8. Es una evaluación distinta del protocolo original con corte seleccionado sobre el propio OOF y del leaderboard.

## Documentación

- resultados_ablaciones.md: métricas, transiciones, repetición, IC y criterios de selección.
- diagnostico_falsas_alarmas.md: auditoría 39/70/2, cohortes actuales y seguimiento por ID.
- diagnostico_sobregeneralizacion.md: escapes originales reconstruidos y FN de B0 actual.
- validacion_generalizacion.md: robustez estructural, longitud y limitaciones.
- comparativa_por_tipo.csv, metricas.csv, transiciones.csv y cambios_por_caso.csv: tablas completas.
- predicciones_oof.csv y predicciones_repetidas.json: scores fuera de fold; folds.json incluye particiones internas/externas.
- protocolo.json: configuración, hashes y versiones; seleccion.json: controles; integridad.json y verificacion.json: comprobaciones.
- baseline_v3/: copia original inalterada; caracteristicas_b.py: funciones autónomas; test_caracteristicas_b.py: controles sintéticos.

## Reproducir

Se utilizó /private/tmp/guardian-venv/bin/python, con NumPy 2.5.4 y scikit-learn 1.9.1. Sustituir python por ese intérprete si corresponde:

```bash
python -m unittest discover -s kit/experimentos/propuesta_b_matices -p test_caracteristicas_b.py
python kit/experimentos/propuesta_b_matices/experimentar.py
```

Para regenerar solo informes a partir de resultados ya guardados: `python kit/experimentos/propuesta_b_matices/experimentar.py --solo-informes`. No ejecuta el main del entrenador original. No realiza commits, PR, envíos oficiales, APIs ni descarga de recursos.

Los tres conteos previos no tienen manifiesto local verificable. Las métricas de entrenamiento, OOF principal, protocolo original retrospectivo y leaderboard se distinguen expresamente. La inferencia de un eventual candidato utiliza solo Python estándar.

piloto_sujeto_singular/ conserva la primera ejecución, en la que B4 era constante. La ejecución principal añade el mecanismo de subconjunto numerado que faltaba en esa primera implementación. Se repite toda la comparación y se declara la adaptación posterior al piloto; no es un estudio confirmatorio.

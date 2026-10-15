# Candela v2 — resultado del experimento B

El experimento B no produjo una mejora suficientemente estable para sustituir v3.
Por ese motivo, `mi_bot.py` y `entrenar.py` son copias exactas de `entregas/joseca_v3/`.
El nombre de esta carpeta identifica el resultado archivado; no implica una arquitectura nueva ni una mejora validada.

## Contenido

- `mi_bot.py`: bot autónomo de referencia, con Python estándar.
- `entrenar.py`: entrenador original de v3. Su ejecución normal sobrescribe los árboles del bot de esta carpeta; usar `--no-escribir` para evaluar sin modificarlos.
- `experimento_b/`: resultados completos, predicciones OOF, particiones, scripts, pruebas, auditorías y piloto inicial conservado.
- `procedencia.json`: hashes y decisión del experimento.

## Resultados

B0 obtuvo score H OOF 0,3494 y AUROC H 0,8989. B7 obtuvo score H 0,3896 y AUROC H 0,8955, pero no superó los controles de estabilidad y regresión. No se generó `mi_bot_candidato_b.py`.

Consultar [resultados de ablaciones](experimento_b/resultados_ablaciones.md),
[diagnóstico de falsas alarmas](experimento_b/diagnostico_falsas_alarmas.md) y
[diagnóstico de sobregeneralización](experimento_b/diagnostico_sobregeneralizacion.md).

## Reproducción desde la raíz del repositorio

Con NumPy y scikit-learn en las versiones documentadas en `experimento_b/protocolo.json`:

```bash
python -m unittest discover -s entregas/candela_v2/experimento_b -p test_caracteristicas_b.py
python entregas/candela_v2/experimento_b/experimentar.py
```

Los scripts reutilizan los manifiestos de `kit/experimentos/fase4_diagnostico/`.
Las rutas de ejecución que aparecen en los informes archivados conservan la ubicación original del experimento; los comandos anteriores ejecutan esta copia.

Los originales de Joseca y los resultados de `kit/experimentos/propuesta_b_matices/` se conservan. No se ha generado ni enviado una entrega oficial.

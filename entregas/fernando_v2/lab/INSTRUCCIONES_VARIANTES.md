# Reglas comunes para implementar variantes (fase 2)

- Python: C:\INGENIERO\malackaton\Gartic2.0\.venv\Scripts\python.exe (no instalar nada). `export PYTHONIOENCODING=utf-8`.
- Cada variante = carpeta `entregas/fernando_v2/lab/variantes/<ID>/` con `mi_bot.py` = COPIA de
  `entregas/joseca_v4/mi_bot.py` con UN solo cambio, y opcionalmente `config.json` (ver docstring de `lab/banco.py`).
- Escribe SOLO dentro de tus carpetas `variantes/<ID>/`. No toques banco.py, comparar.py, holdout_familias.txt,
  variantes/v4/, entregas/joseca_*/, kit/.
- Evalúa SIEMPRE con: `python entregas/fernando_v2/lab/banco.py entregas/fernando_v2/lab/variantes/<ID>` (desde la raíz
  del repo; ~20-60 s). NUNCA uses `--holdout`. Compara con: `python entregas/fernando_v2/lab/comparar.py <ID>`.
- El banco sustituye `probabilidad` por el modelo de cada partición y llama a `Guardian().score(caso)`: toda tu lógica
  debe pasar por `caracteristicas()`, `NOMBRES`, `UMBRAL`, `reescalar()` y `Guardian.score()`. Si añades features,
  añádelas a `caracteristicas()` y a la lista `NOMBRES` (el modelo de la v4 embebido en ARBOLES deja de valer, da igual:
  el banco reentrena). Si quitas features, usa `"excluir"` en config.json.
- Reglas del reto: mi_bot.py solo stdlib/numpy/sklearn; nada de reglas o palabras para casos concretos del test o del
  giro (puedes leer test/giro solo para ver formato); listas de palabras de español general.
- Prohibido elegir umbrales o parámetros mirando la CV y volver a evaluar sobre ella sin decirlo: si un cambio tiene un
  parámetro (p. ej. PROB_REGLA, UMBRAL_LARGAS), usa el valor del catálogo/v5 o uno fijado a priori y dilo.
- Si el `mi_bot.py` lanza excepciones, la v4 devuelve 1.0 en silencio: comprueba con un try/except propio que
  `caracteristicas()` no falla en train, test y giro (cuenta las excepciones y repórtalas).
- Devuelve SOLO: la fila de `comparar.py` para cada variante, una línea de qué cambiaste (funciones/líneas) y de dónde
  viene, el nº de excepciones, y problemas encontrados. Nada de código ni salidas largas.

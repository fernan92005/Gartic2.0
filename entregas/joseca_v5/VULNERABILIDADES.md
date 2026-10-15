# joseca_v5: vulnerabilidades conocidas

> Documento interno del equipo (no se sube al repositorio de entregas).
> Datos medidos el 15/10 por la tarde, después de la entrega 6.

## 1. Resumen

| Entrega | Bot | Qué se puntúa | Nota pública | AUROC |
|---|---|---|---|---|
| 1 | joseca_v1 | test | 0,353 | 0,905 |
| 3 | joseca_v3 | test | 0,459 | 0,910 |
| 4 | joseca_v4 | test | 0,481 | 0,919 |
| **6** | **joseca_v5** | **test + giro** | **0,377** | **0,904** |

La entrega 6 es la primera que incluye el giro, y la nota baja. Nuestras estimaciones previas eran de unos +0,6 en el test y +0,45 en el giro.

No sabemos cuánto saca cada parte por separado, pero se puede estimar. El test tiene 265 respuestas fieles; si su parte rinde como en las entregas anteriores (0,48–0,55), **el giro estaría en torno a 0, o incluso por debajo**. Eso es lo mismo que marcarlo todo como alucinación.

**Conclusión: el bot se comporta bien con casos como los del train y el test, pero no generaliza al giro.** El pseudo-giro, nuestra referencia para el giro, no lo ha anticipado.

## 2. Qué cambia en el giro: las señales en las que se apoya el bot desaparecen

Porcentaje de **oraciones** en las que se activa cada señal. Las respuestas del giro se han partido en oraciones, que es como las puntúa el bot.

| Señal | H fieles | H alucinaciones | Test | **Giro** |
|---|---|---|---|---|
| El juez LLM dice «contradicha» | 1 % | 26 % | 15 % | **4 %** |
| El juez LLM dice «no respaldada» | 1 % | 13 % | 2 % | **1 %** |
| Confianza media del LLM al respaldar | 0,94 | 0,58 | 0,80 | **0,93** |
| Cuenta comprobada y falsa | 1 % | 12 % | 4 % | **1 %** |
| Cifra sospechosa | 3 % | 18 % | 10 % | **3 %** |
| «Plan → seguro» (tiempos verbales) | 0 % | 11 % | 4 % | **1 %** |
| **Matiz perdido por oración** | 9 % | 31 % | 18 % | **19 %** |
| Mejor respaldo NLI de una sola fuente (media) | 0,53 | 0,31 | 0,47 | **0,73** |
| **Oraciones por encima del corte** | 13 % | 97 % | 48 % | **35 %** |

Cómo leer la tabla:

- **Las señales fuertes casi no se activan en el giro.** El juez LLM, el verificador de cuentas, las cifras sospechosas y «plan → seguro» salen en el giro con la misma frecuencia que en los casos H *fieles*. O el giro tiene pocas alucinaciones, o las tiene de un tipo que ni el LLM ni nuestras comprobaciones ven. El profesor avisó de que el giro es difícil, así que lo segundo es lo más probable.
- **Lo que sí salta en el giro son señales flojas.** «Matiz perdido por oración» salta en el 19 % de las oraciones. En H salta en el 9 % de las fieles, así que también se activa con frecuencia en respuestas fieles. En el giro decide casi a ciegas.
- **El efecto multiplicador.** El 35 % de las oraciones del giro supera el corte. Con 3,3 oraciones por respuesta, la probabilidad de que *alguna* lo supere es de 1 − 0,65^3,3 ≈ 76 %. Por eso el bot marca el **71 % del giro**: la mayoría de esas alarmas no se basan en ninguna prueba concreta.

## 3. Vulnerabilidades

### V1. Cambio de dominio: el bot está ajustado a noticias municipales

- **Qué pasa:** el train (casos H) son noticias de ayuntamientos, asociaciones y empresas pequeñas. El giro son documentos técnicos y legales: auditorías, inspecciones, pólizas, plazos.
  - Las listas de palabras (matices, causas, alcance) y los patrones de cuentas están hechos mirando el train.
  - En los textos técnicos los matices son constantes («salvo», «en el plazo de», «como máximo») y las cifras llevan unidades.
- **Evidencia:** «matiz perdido por oración» se activa el doble que en los casos fieles de H, y las señales de cifras casi desaparecen (tabla anterior).
- **Mitigación:** reducir el peso de las señales léxicas cuando no hay pruebas fuertes (ver V3) y revisar a mano una muestra del giro para ver qué tipo de alucinaciones trae.

### V2. El pseudo-giro no representa al giro

- **Qué pasa:** el pseudo-giro junta 3–4 respuestas H de una oración. Cada oración sigue siendo una oración H, con sus mismas señales. El giro real tiene otras fuentes (más largas, 127 caracteres frente a 103), otro vocabulario y otras alucinaciones.
- **Evidencia:**
  - El pseudo-giro estimaba +0,45 y el giro real parece rondar 0.
  - El AUROC de la entrega bajó de 0,919 a 0,904.
- **Mitigación:** no fiarse del pseudo-giro para decidir. Si hay que usarlo, darle menos peso que a la validación de H y comprobar siempre qué proporción marca el bot en el giro real.

### V3. Agregación por la peor oración: multiplica las falsas alarmas

- **Qué pasa:** en respuestas de varias oraciones, la puntuación es la de la peor oración, con un corte propio (`UMBRAL_LARGAS = 0,25`). Esto protege contra alucinaciones diluidas (en el pseudo-giro detectaba el 98 %), pero con señales ruidosas convierte cualquier oración dudosa en una alarma.
- **Evidencia:** marca el 71 % del giro, frente al 48 % del test.
- **Mitigación:**
  - Subir el corte de respuestas largas o volver a mezclar «respuesta entera» y «peor oración».
  - Exigir que la peor oración tenga **alguna prueba concreta** (LLM negativo, cuenta falsa, «plan → seguro», cifra sospechosa) para pasar de cierto score. Si no la tiene, limitar su probabilidad.
  - Comprobar el porcentaje marcado en el giro: algo entre el 40 % y el 50 % sería razonable.

### V4. La validación local es optimista cuando diseñamos mirando errores de H

- **Qué pasa:** el verificador de cuentas y varias reglas se diseñaron mirando errores concretos de H, y luego se midieron sobre esos mismos casos.
- **Evidencia:** la validación de v4 daba +0,55 y el leaderboard 0,481.
- **Mitigación:** antes de diseñar una regla, separar un trozo de familias H que no se mire nunca y usarlo solo para la medición final.

### V5. Las sobregeneralizaciones sin palabra clave se escapan

- **Qué pasa:** respuestas como «las exportaciones de fruta crecieron un 18 % *en Escaleira*» o «concederá 180 becas *para alumnos de FP*» cambian el alcance sin usar «todos», «cualquier» ni nada parecido.
- **Evidencia:** en H se escapan 10 de 86 sobregeneralizaciones (12 %); es el tipo que más se escapa. Ni nuestros detectores ni el B4 de Candela las cazan.
- **Mitigación:** comparar el sujeto y los complementos de la respuesta con los de la frase de la fuente: qué se quita y qué se generaliza. Es difícil sin un modelo de lenguaje.

### V6. Errores del juez LLM que el bot da por buenos

- **Qué pasa:** cuando el LLM dice «contradicha» o «no respaldada», el bot casi siempre marca alucinación. En H acierta (118 de 128), pero los 10 fallos son falsas alarmas seguras. Por ejemplo, «Sierra Rasa perdió el 9 %, menos que Peñagrís» o «el 15 % se compró con fondos públicos».
- **Mitigación:** cuando el LLM dice «contradicha» y todas las cifras cuadran y el NLI no ve contradicción, bajar la probabilidad. Lo intentamos en v2 sin efecto; con el verificador actual podría funcionar mejor.

### V7. El verificador de cuentas puede cuadrar por casualidad

- **Qué pasa:** con muchas cifras en las fuentes, alguna resta o suma da el número buscado por azar. Ya pasó con «3 menos que en 2025», que daba bueno con 2 + 8. Se ha restringido mucho, pero:
  - en textos técnicos con más cifras el riesgo crece;
  - fuera de H, «cuenta comprobada y falsa» acierta peor (15 fieles frente a 6 alucinaciones en el resto del train), y la **regla fija** de «cuenta mal → alarma» depende de ella.
- **Mitigación:** limitar la regla fija a los casos H o similares, o pedirle más confianza (por ejemplo, que el error sea de más del 5 %).

### V8. Unidades y formatos no cubiertos

- **Qué pasa:** se cubren mm/cm/m/km, g/kg/t, ml/l, Wh/kWh/MWh/GWh, W/kW/MW y minutos/horas/días. No se cubren hectáreas y m², euros/céntimos en todas sus formas, ppm, porcentajes y puntos mezclados, fechas en formato numérico (10/03/2026) ni intervalos («del 10 al 12 de marzo»).
- **Mitigación:** ampliar la tabla de unidades con las del giro, revisando sus fuentes.

### V9. Dependencia de una sola señal

- **Qué pasa:** la característica más importante de los árboles es la confianza mínima del LLM al respaldar. En el giro esa confianza es alta (0,93), igual que en los casos fieles de H, así que el modelo pierde su señal principal y decide con las secundarias, que son las ruidosas.
- **Mitigación:** la misma que V3, más características que no dependan del LLM.

### V10. Riesgos de procedimiento

- **El modelo va escrito dentro de `mi_bot.py`.** Si alguien cambia las características sin volver a ejecutar `entrenar.py`, los árboles ya no corresponden. Hay que reentrenar siempre después de tocar `caracteristicas()`.
- **`evaluar_local.py` engaña con los árboles.** Da 0,74–0,83 en train por memorización. Para comparar versiones hay que usar la validación de `entrenar.py`.
- **Numeración de entregas:** el profesor no hace merge de los PR. Antes de entregar hay que mirar los PR abiertos del equipo para elegir el número.

## 4. Prioridades

1. **V3 y V1: reducir las falsas alarmas del giro.** Que la peor oración solo pueda superar el corte si tiene alguna prueba concreta, y comprobar que el porcentaje marcado en el giro baja hacia el 40–50 %. Es el cambio con más impacto esperado.
2. **Revisar a mano unas 20–30 respuestas del giro** para saber qué tipo de alucinaciones trae y si alguna comprobación nuestra las puede ver.
3. **V7:** endurecer la regla fija de cuentas.
4. **V4:** separar familias H solo para la medición final.

Mientras no se pueda medir el giro, **entregar dos variantes** (la actual y una más conservadora en respuestas largas) y quedarse con la que mejor nota pública saque.

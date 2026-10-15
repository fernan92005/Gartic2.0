# Prioridades de mejora — propuestas, no implementadas

Orden por coste de FN del Joseca OOF de desarrollo al corte actual. La estabilidad y el riesgo de FP se usan para orientar las pruebas, no para declarar que una propuesta funcionará. Los pesos de categorías siguen siendo iguales.

| Prioridad | Tipo | Problema principal | FN actuales | Coste | Mejora propuesta | Riesgo de FP |
|---|---|---|---:|---:|---|---|
| 1 | Distorsión | Interpretación aritmética y temporal con cobertura explícita | 7 | 28 | Separar suma, diferencia, porcentaje, fechas y cambio de unidad; contrastar resultados solo cuando la relación es inequívoca. | Alto si falta alineamiento o se castigan paráfrasis |
| 2 | Sobregeneralización | Alcance aceptado por MiniMax y señales próximas a fieles | 7 | 28 | Representar cuantificadores y modalidades sobre el mismo sujeto/intervalo, sin vetar palabras universales aisladas. | Alto si falta alineamiento o se castigan paráfrasis |
| 3 | No respaldado | Detalle local aceptado por MiniMax | 2 | 8 | Conservar relación fragmento→oración y comprobar soporte del detalle en la evidencia local citada. | Alto si falta alineamiento o se castigan paráfrasis |
| 4 | Contradicción | Contradicción en evidencia relevante | 2 | 8 | Combinar contradicción con la fuente que comparte entidad/atributo, sin maximizar sobre distractores. | Alto si falta alineamiento o se castigan paráfrasis |

Prioridad transversal: las **333 falsas alarmas** cuestan 333; antes de añadir nuevas alarmas debe comprobarse que cada FN corregido no produce más de cuatro FP, y que mejora el objetivo ponderado sin deteriorar H. Esta comparación de costes no justifica tratar FP como tipo de alucinación.

Las propuestas se vinculan a mecanismos observados, no a mejoras ya demostradas. La referencia E4 solo coincide con 0 FN del Joseca: su ganancia sobre otro HGB no puede trasladarse automáticamente.

En los empates de coste no hay una prioridad estadística demostrada. Distorsión concentra sus siete FN en dos folds, sobregeneralización en tres y contradicción en uno; consultar estabilidad_por_tipo.csv. Las cohortes de atribución cruzada, causa añadida y condición perdida detectada tienen cero FN: las propuestas sobre esos mecanismos son exploratorias y tienen menor respaldo que las basadas en FN observados.

Primer experimento transversal propuesto, aún sin implementar: reentrenar ablaciones de palabras nuevas, alineamiento citado y matices, una familia cada vez, con corte interno y mismo protocolo agrupado. Evidencia: 151 FP coinciden con vocabulario nuevo, 184 con poco respaldo citado y 116 con síntesis multifuente (cohortes solapadas). La intervención a cero de matices quita 37 FP pero añade 13 FN y empeora coste en 15: no justifica suprimirla. Hipótesis: mejorar la alineación y el trato de paráfrasis conserva sensibilidad con menos falsas alarmas. Riesgo: perder FN que dependen de esas mismas señales. Esfuerzo: bajo para la ablación, medio para cambiar alineamiento; cualquier aceptación requiere mejora pareada estable y futura evidencia no inspeccionada.

## Distorsión

Evidencia de diagnóstico: 7 FN; MiniMax respalda todas las oraciones en 7; riesgo NLI de oración >0,5 en 3. Distribuciones, cohortes y ejemplos están en diagnostico_tipos.md y diagnostico_nli_minimax.md.

Tres FN contienen un año y dos pertenecen a la cohorte de cuenta sospechosa (pueden solaparse); la atribución cruzada detectada tiene cero FN. Las operaciones/fechas tienen mayor respaldo como primera investigación. Esfuerzo medio; atribución completa alto.

1. **Hipótesis: Interpretación aritmética y temporal con cobertura explícita.** Mecanismo: Separar suma, diferencia, porcentaje, fechas y cambio de unidad; contrastar resultados solo cuando la relación es inequívoca. Componente: Extractor numérico; investigar signos y cruces aprendidos en el régimen MiniMax favorable. Ablación necesaria: Ablación por familia de operación y controles fieles de conversiones, tasas y redondeos. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

2. **Hipótesis: Atribución de cantidades al atributo y sujeto correctos.** Mecanismo: Restringir operandos por evidencia de atributo/entidad y unidad, con abstención si hay varias lecturas. Componente: Comprobaciones numéricas y cruces llmok; no un veto de cifra ausente. Ablación necesaria: Comparar extracción local frente a global, retirando solo esta restricción; medir ΔFP y ΔFN por distorsión. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

## Sobregeneralización

Evidencia de diagnóstico: 7 FN; MiniMax respalda todas las oraciones en 7; riesgo NLI de oración >0,5 en 2. Distribuciones, cohortes y ejemplos están en diagnostico_tipos.md y diagnostico_nli_minimax.md.

NLI por sí solo separa poco esta categoría de fieles (AUROC 0,608) y MiniMax respalda 54 de 97 positivos; los siete FN reciben su respaldo. La cohorte de condición perdida ya detectada tiene cero FN, así que ampliar condiciones es una hipótesis de cobertura, no una causa comprobada. Esfuerzo medio para alcance y alto para alineación de excepciones.

1. **Hipótesis: Alcance aceptado por MiniMax y señales próximas a fieles.** Mecanismo: Representar cuantificadores y modalidades sobre el mismo sujeto/intervalo, sin vetar palabras universales aisladas. Componente: Características de alcance/modalidad. Ablación necesaria: Retirar modalidad o cuantificadores por separado; incluir fieles con expresiones absolutas justificadas. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

2. **Hipótesis: Condición o excepción eliminada.** Mecanismo: Alinear el atributo con su restricción local; distinguir omisión irrelevante de endurecimiento de una afirmación. Componente: matiz_perdido_tramo/fuente y selección de evidencia. Ablación necesaria: Comparar tramo contra documento completo con mismo clasificador y cortes internos. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

## No respaldado

Evidencia de diagnóstico: 2 FN; MiniMax respalda todas las oraciones en 2; riesgo NLI de oración >0,5 en 1. Distribuciones, cohortes y ejemplos están en diagnostico_tipos.md y diagnostico_nli_minimax.md.

Los dos FN son aceptados por MiniMax; no hay FN en la cohorte de causa añadida detectada. Priorizar soporte del detalle antes de nuevas reglas causales. Esfuerzo medio; evidencia limitada a dos errores.

1. **Hipótesis: Detalle local aceptado por MiniMax.** Mecanismo: Conservar relación fragmento→oración y comprobar soporte del detalle en la evidencia local citada. Componente: Señales de detalle y desacuerdo; no confiar únicamente en confianza MiniMax. Ablación necesaria: Ablación por granularidad y por respaldo conjunto para comprobar que no se castiga síntesis fiel. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

2. **Hipótesis: Causa o consecuencia sin evidencia.** Mecanismo: Vincular conector causal a la relación que expresa y sus fuentes, en vez de buscar cualquier conector en el documento. Componente: anadido_causal y sus cruces. Ablación necesaria: Retirar componente causal; medir FN no respaldados y FP en causas que sí aparecen para el sujeto correcto. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

## Contradicción

Evidencia de diagnóstico: 2 FN; MiniMax respalda todas las oraciones en 2; riesgo NLI de oración >0,5 en 1. Distribuciones, cohortes y ejemplos están en diagnostico_tipos.md y diagnostico_nli_minimax.md.

Ambos FN contienen un verbo direccional y están en el mismo fold. Hay 111 FP con contradicción NLI en alguna fuente: maximizar contradicción indiscriminadamente tiene riesgo alto. Esfuerzo medio para verbos, alto para alineación robusta.

1. **Hipótesis: Contradicción en evidencia relevante.** Mecanismo: Combinar contradicción con la fuente que comparte entidad/atributo, sin maximizar sobre distractores. Componente: Características NLI/alineamiento y llm_contra_sin_apoyo. Ablación necesaria: Retirar solo la alineación y mantener el resto; comparar contradicción real frente a FP NLI. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

2. **Hipótesis: Oposición direccional y de relaciones.** Mecanismo: Comprobar pares verbales opuestos en cláusulas alineadas; abstenerse ante negación o sujeto ambiguo. Componente: Verificador textual limitado, candidato futuro. Ablación necesaria: Separar casos con/sin negación y evaluar pares fieles con otros sujetos antes de integrar. Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.

## Qué no hacer todavía

No desplegar las intervenciones de características a cero; sus pesos están correlacionados y no equivalen a modelos reentrenados. No endurecer el corte por observar FN aislados. No añadir el E4 de Candela al Joseca sin un experimento propio. No usar longitud, prefijos o categorías como predictores. Las ideas de granularidad, alineamiento y complementariedad son inspiraciones funcionales de fases anteriores, no reproducciones de modelos neuronales.

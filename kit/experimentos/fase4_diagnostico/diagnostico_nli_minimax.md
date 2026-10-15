# Diagnóstico NLI y MiniMax — OOF Joseca

Las señales se analizan con las etiquetas reales de desarrollo; no se recalculan NLI, similitud ni MiniMax. Riesgo NLI descriptivo = 1 − max(respaldo conjunto, mejor respaldo individual). Sus mínimos/máximos/medias se calculan por oración y fragmento, separados del score logístico de Joseca.

## Cuando MiniMax acepta alucinaciones

| Tipo | Positivos | MiniMax respalda todas | FN entre ellos | NLI conjunto/relevante >0,5 y MiniMax favorable | Ambos favorables NLI riesgo<=0,2 | FN entre ambos favorables |
|---|---:|---:|---:|---:|---:|---:|
| Distorsión | 274 | 73 | 7 | 15 | 18 | 3 |
| Contradicción | 120 | 6 | 2 | 1 | 2 | 0 |
| No respaldado | 178 | 41 | 2 | 5 | 7 | 1 |
| Sobregeneralización | 97 | 54 | 7 | 12 | 35 | 5 |

El veredicto favorable no garantiza fidelidad; el desacuerdo NLI puede aportar señal o proceder de una fuente distractora. Se distingue contradicción conjunta/alineada de máxima en cualquier fuente.

## Distribuciones principales: mediana [p25,p75]

| Tipo / resultado | N | Respaldo individual | Respaldo conjunto | Contradicción fuente | Contradicción conjunta | Similitud | Confianza MiniMax | Riesgo fragmento |
|---|---:|---|---|---|---|---|---|---|
| fiel/error | 333 | 0.312 [0.049, 0.908] | 0.743 [0.098, 0.956] | 0.157 [0.023, 0.729] | 0.030 [0.005, 0.183] | 0.805 [0.707, 0.941] | 0.980 [0.950, 1.000] | 0.104 [0.021, 0.534] |
| fiel/acierto | 278 | 0.843 [0.235, 0.984] | 0.857 [0.117, 0.984] | 0.131 [0.008, 0.692] | 0.020 [0.002, 0.329] | 0.814 [0.701, 0.919] | 0.980 [0.950, 1.000] | 0.052 [0.009, 0.410] |
| distorsion/error | 7 | 0.484 [0.339, 0.744] | 0.239 [0.003, 0.849] | 0.501 [0.273, 0.916] | 0.113 [0.003, 0.462] | 0.815 [0.579, 0.839] | 0.950 [0.935, 0.970] | 0.547 [0.266, 0.968] |
| distorsion/acierto | 267 | 0.031 [0.006, 0.177] | 0.005 [0.001, 0.058] | 0.930 [0.394, 0.992] | 0.972 [0.251, 0.996] | 0.800 [0.719, 0.888] | 0.980 [0.950, 0.990] | 0.944 [0.654, 0.991] |
| contradiccion/error | 2 | 0.092 [0.056, 0.129] | 0.276 [0.162, 0.391] | 0.496 [0.293, 0.700] | 0.699 [0.577, 0.821] | 0.858 [0.812, 0.904] | 0.944 [0.922, 0.966] | 0.519 [0.298, 0.741] |
| contradiccion/acierto | 118 | 0.009 [0.003, 0.050] | 0.003 [0.001, 0.028] | 0.865 [0.122, 0.993] | 0.867 [0.098, 0.996] | 0.806 [0.723, 0.882] | 0.980 [0.950, 0.990] | 0.986 [0.694, 0.996] |
| no_respaldado/error | 2 | 0.183 [0.143, 0.224] | 0.660 [0.493, 0.828] | 0.295 [0.148, 0.442] | 0.145 [0.073, 0.217] | 0.697 [0.588, 0.805] | 0.950 [0.950, 0.950] | 0.322 [0.164, 0.481] |
| no_respaldado/acierto | 176 | 0.007 [0.002, 0.036] | 0.003 [0.001, 0.021] | 0.063 [0.006, 0.622] | 0.004 [0.000, 0.054] | 0.871 [0.722, 0.969] | 0.950 [0.850, 0.970] | 0.988 [0.871, 0.997] |
| sobregeneralizacion/error | 7 | 0.803 [0.317, 0.873] | 0.973 [0.428, 0.990] | 0.927 [0.389, 0.927] | 0.009 [0.005, 0.309] | 0.796 [0.740, 0.883] | 0.950 [0.850, 0.950] | 0.327 [0.016, 0.943] |
| sobregeneralizacion/acierto | 90 | 0.231 [0.018, 0.811] | 0.354 [0.018, 0.909] | 0.506 [0.130, 0.950] | 0.142 [0.017, 0.747] | 0.806 [0.740, 0.886] | 0.950 [0.950, 0.970] | 0.267 [0.020, 0.927] |

El CSV `distribuciones_senales.csv` amplía estas tablas por H/no-H e incluye respaldo y contradicción individual/conjunto, similitud, veredicto, confianza, desacuerdo, riesgo mínimo/máximo/medio por oración/fragmento y su diferencia.

## Proximidad a señales de respuestas fieles

AUROC de cada señal por categoría frente a fieles; % positivos dentro del intervalo intercuartílico de fieles. Es discriminación de señales individuales, no el AUROC del bot.

| Tipo | AUROC riesgo NLI | AUROC riesgo MiniMax | % NLI en IQR fiel | % MiniMax en IQR fiel |
|---|---:|---:|---:|---:|
| Distorsión | 0.860 | 0.914 | 17.5% | 21.9% |
| Contradicción | 0.881 | 0.992 | 15.8% | 2.5% |
| No respaldado | 0.924 | 0.952 | 6.2% | 12.9% |
| Sobregeneralización | 0.608 | 0.841 | 41.2% | 45.4% |

## Qué cambia al usar fragmentos

Control descriptivo de máximo riesgo NLI a corte fijo 0,5: cambian **184** casos. Pasan de fiel a alerta 79: 35 positivos y 44 fieles. Pasan de alerta a fiel 105: 48 positivos y 57 fieles.

Esto no es una sustitución ni una calibración del bot: aísla el efecto de granularidad sobre las señales. Los casos concretos están en cambios_agregacion.csv. Un fragmento puede aportar evidencia local o perder sujeto/contexto. No se atribuye causalidad a una señal individual.

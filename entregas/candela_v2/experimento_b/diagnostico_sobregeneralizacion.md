# Diagnóstico de sobregeneralización

Se comparan los FN reales de B0 en los folds actuales. Los dos FN citados en el encargo no traen IDs ni un manifiesto verificable; la reproducción original permite mostrar cuántos hay con cada semilla, sin asumir identidad.

| Variante | Casos H | FN H | Recall H | Coste FN H | FN global | FN H B0 corregidos | FN H nuevos |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 63 | 2 | 0.9683 | 8 | 2 | 0 | 0 |
| B1 | 63 | 2 | 0.9683 | 8 | 2 | 0 | 0 |
| B2 | 63 | 3 | 0.9524 | 12 | 3 | 0 | 1 |
| B3 | 63 | 4 | 0.9365 | 16 | 4 | 0 | 2 |
| B4 | 63 | 2 | 0.9683 | 8 | 2 | 0 | 0 |
| B5 | 63 | 3 | 0.9524 | 12 | 3 | 0 | 1 |
| B6 | 63 | 5 | 0.9206 | 20 | 6 | 0 | 3 |
| B7 | 63 | 3 | 0.9524 | 12 | 4 | 0 | 1 |
| B1_norm | 63 | 2 | 0.9683 | 8 | 3 | 0 | 0 |
| B0_sin_NLI | 63 | 6 | 0.9048 | 24 | 7 | 0 | 4 |
| B3_sin_NLI | 63 | 5 | 0.9206 | 20 | 6 | 0 | 3 |
| B5_sustituye_matices | 63 | 5 | 0.9206 | 20 | 5 | 0 | 3 |

## Casos que escapan a B0 en desarrollo

| ID | Dominio | p B0 | Corte B0 | B4 sujeto | B4 confianza | Variantes que los detectan |
|---|---|---:|---:|---:|---:|---|
| H111::r4 | H | 0.1035 | 0.14 | 1 | 0.85 | ninguna |
| H422::r2 | H | 0.0971 | 0.14 | 1 | 0.85 | ninguna |

## Seguimiento de FN del protocolo original, semilla 0

| ID original reconstruido | En desarrollo | FN B0 actual | Variantes que lo detectan en el nuevo protocolo |
|---|---|---|---|
| H310::r3 | no (reserva histórica) | no evaluado OOF B actual | no se usa para elegir |
| H401::r8 | sí | False | B0, B1, B2, B3, B4, B5, B1_norm |
| H422::r2 | sí | True | ninguna |

B4 utiliza coincidencia de acción/propiedad y evidencia de sujeto singular, no el plural aislado. Se abstiene con colectivo respaldado, varios sujetos documentados, nombres enumerados o restricción explícita. La pregunta ayuda a interpretar el caso, pero su plural no demuestra respaldo. Esta heurística no resuelve correferencia general ni todos los cambios de alcance; los tests sintéticos prueban su mecanismo, no su recall en el giro.

La revisión B4 añade fracciones estrictas (n de N sujetos o de N sujetos, n...) y restricciones explícitas vinculadas a la misma propiedad. Se distingue el subconjunto del colectivo completo; una fuente que respalda el colectivo completo provoca abstención. Se describe su cobertura real en la tabla, sin tratar los casos del piloto como una prueba ciega.

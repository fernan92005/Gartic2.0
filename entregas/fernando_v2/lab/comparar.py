"""Compara cada variante con la v4, partición a partición (mismas particiones y semillas).

  python comparar.py [variantes ...]               # por defecto, todas las carpetas de lab/variantes/
  python comparar.py --base A1 A1_A2 A1_A6 ...      # diferencia pareada frente a otra variante (greedy)
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
VAR = os.path.join(AQUI, "variantes")


def cargar(nombre):
    return json.load(open(os.path.join(VAR, nombre, "resultado.json"), encoding="utf-8"))


def fila(r, base):
    assert all(a["ids"] == b["ids"] for a, b in zip(r["pliegues"], base["pliegues"])), "particiones distintas"
    d = np.array([a["puntuacion"] - b["puntuacion"] for a, b in zip(r["pliegues"], base["pliegues"])])
    k = base["fp"] - r["fp"]          # falsas alarmas que quita (suma de las 15 particiones)
    e = r["fn"] - base["fn"]          # escapes que añade
    ct = np.mean([(r["test_scores"][i] >= .5) != (base["test_scores"][i] >= .5) for i in base["test_scores"]])
    cg = np.mean([(r["giro_scores"][i] >= .5) != (base["giro_scores"][i] >= .5) for i in base["giro_scores"]])
    return {"variante": r["variante"], "cv": r["cv_media"], "desv": r["cv_desv"], "auroc": r["auroc"],
            "fp": r["fp"], "fn": r["fn"], "dif": d.mean(), "dif_desv": d.std(ddof=1), "ganadas": int((d > 1e-12).sum()),
            "empates": int((abs(d) <= 1e-12).sum()), "k": k, "e": e, "coste_ok": 4 * e < k,
            "giro": r["giro_marcado"], "giro3": r["giro_marcado_3"], "giro4": r["giro_marcado_4"],
            "cambia_test": ct, "cambia_giro": cg}


def main(argv=None):
    args = list(argv or sys.argv[1:])
    nombre_base = "v4"
    if args[:1] == ["--base"]:
        nombre_base, args = args[1], args[2:]
    nombres = args or sorted(d for d in os.listdir(VAR) if os.path.exists(os.path.join(VAR, d, "resultado.json")))
    base = cargar(nombre_base)
    print(f"Base: {nombre_base}")
    print("| variante | CV | AUROC | dif. pareada ± desv | ganadas/15 | FP | FN | k | e | 4e<k | giro % (3 / 4 or.) | cambia test | cambia giro |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in nombres:
        f = fila(cargar(n), base)
        print(f"| {f['variante']} | {f['cv']:+.3f} ± {f['desv']:.3f} | {f['auroc']:.3f} | {f['dif']:+.3f} ± {f['dif_desv']:.3f} | "
              f"{f['ganadas']} (={f['empates']}) | {f['fp']} | {f['fn']} | {f['k']} | {f['e']} | {'sí' if f['coste_ok'] else 'no'} | "
              f"{f['giro']:.1%} ({f['giro3']:.1%} / {f['giro4']:.1%}) | {f['cambia_test']:.1%} | {f['cambia_giro']:.1%} |")


if __name__ == "__main__":
    main()

"""Construye el bot final a partir de una variante: entrena con TODO el train (H, incluido el
hold-out ya usado, + no-H) con la config de la variante, embebe los árboles en el formato de
joseca_v4 (bloque # >>> MODELO ... # <<< MODELO) y comprueba Python puro frente a sklearn.

  python construir_final.py <carpeta_variante> <salida/mi_bot.py>
"""
import json
import re
import sys

import numpy as np

import banco as B


def main(carpeta, salida):
    v = B.Variante(carpeta)
    if v.cfg["modelo"] != "gb":
        raise SystemExit("solo se exporta GradientBoosting (formato de la v4)")
    train = B.leer("train.jsonl")
    casos = [B.para_bot(c) for c in train]
    y = [c["etiqueta"] for c in train]
    es_h = np.array([c["id"].startswith("H") for c in train])
    m = v.ajustar(casos, y, es_h)
    X = v.X(casos)
    arboles = [[[int(a) for a in t.tree_.feature], [float(a) for a in t.tree_.threshold],
                [int(a) for a in t.tree_.children_left], [int(a) for a in t.tree_.children_right],
                [round(float(a), 6) for a in t.tree_.value[:, 0, 0]]] for t in m.estimators_[:, 0]]
    tasa = float(m.learning_rate)
    inicial = float(m.decision_function(X[:1])[0] - tasa * sum(e.predict(X[:1])[0] for e in m.estimators_[:, 0]))
    bloque = ("# >>> MODELO (generado por fernando_v2/lab/construir_final.py; no editar a mano)\n"
              f"NOMBRES = {json.dumps(v.cols, ensure_ascii=False)}\n"
              f"INICIAL = {inicial:.6f}\nTASA = {tasa}\n"
              f"ARBOLES = {json.dumps(arboles, separators=(',', ':'))}\n"
              f"UMBRAL = {float(v.mod.UMBRAL):.2f}\n# <<< MODELO")
    src = open(f"{carpeta}/mi_bot.py", encoding="utf-8").read()
    assert src.count("# >>> MODELO") == 1, "falta el bloque MODELO"
    src = re.sub(r"# >>> MODELO.*?# <<< MODELO", lambda _: bloque, src, flags=re.S)
    open(salida, "w", encoding="utf-8", newline="").write(src)
    bot = B.cargar(salida)
    p_bot = np.array([bot.probabilidad(v.orig_caracteristicas(c)) for c in casos])
    dif = np.max(np.abs(p_bot - m.predict_proba(X)[:, 1]))
    print(f"{salida}: {len(arboles)} árboles, {len(v.cols)} features, UMBRAL {bot.UMBRAL}; "
          f"dif. máx. Python puro vs sklearn = {dif:.2e}")
    assert dif < 1e-3


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

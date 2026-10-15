"""¿Ayuda el reescalado del corte? CV ANIDADA: en cada partición externa se elige el
umbral con una CV interna (GroupKFold 5) SOLO sobre su entrenamiento, y se aplica a la
partición externa. Se compara con el corte fijo 0,5.

  python umbral_anidado.py
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import entrenar as E  # noqa: E402
import validar_cv as V  # noqa: E402


def reescalar(s, t):
    return np.where(s < t, 0.5 * s / t, 0.5 + 0.5 * (s - t) / (1 - t))


def main(variante=None):
    casos = [json.loads(l) for l in open(V.TRAIN, encoding="utf-8") if l.strip()]
    ad = V.Fernando(os.path.join(AQUI, "..", "mi_bot.py"), variante)
    ad.preparar(casos)
    var = ad.variante
    y = np.array([c["etiqueta"] for c in casos])
    idx_h = np.array([i for i, c in enumerate(casos) if c["id"].startswith("H")])
    idx_no = np.array([i for i, c in enumerate(casos) if not c["id"].startswith("H")])
    g = np.array([V.grupo(casos[i]["id"]) for i in idx_h])

    def oof(tr_h, te_h):
        ad.ajustar(np.concatenate([tr_h, idx_no]))
        return ad.predecir(te_h)

    fijo, anid, umbrales = [], [], []
    for semilla in V.SEMILLAS:
        for tr, te in GroupKFold(5, shuffle=True, random_state=semilla).split(idx_h, groups=g):
            tr_h, te_h = idx_h[tr], idx_h[te]
            # CV interna sobre el entrenamiento externo para elegir el umbral
            s_in = np.zeros(len(tr_h))
            for a, b in GroupKFold(5, shuffle=True, random_state=100 + semilla).split(tr_h, groups=g[tr]):
                s_in[b] = oof(tr_h[a], tr_h[b])
            _, t = E.mejor_corte(y[tr_h], s_in)
            t = t if abs(t - 0.5) > 0.05 else 0.5
            s = oof(tr_h, te_h)
            fijo.append(V.puntuacion(y[te_h], (s >= 0.5).astype(int))[0])
            anid.append(V.puntuacion(y[te_h], (reescalar(s, t) >= 0.5).astype(int))[0])
            umbrales.append(t)
    fijo, anid = np.array(fijo), np.array(anid)
    print(f"Variante {var}")
    print(f"Corte fijo 0,5          : {fijo.mean():+.3f} ± {fijo.std():.3f}")
    print(f"Umbral elegido anidado  : {anid.mean():+.3f} ± {anid.std():.3f}   umbrales {np.round(umbrales, 2).tolist()}")
    print(f"Diferencia por partición: media {np.mean(anid - fijo):+.3f}, mejora en {(anid > fijo).sum()}/15, "
          f"empeora en {(anid < fijo).sum()}/15")
    return float(fijo.mean()), float(anid.mean())


if __name__ == "__main__":
    main()

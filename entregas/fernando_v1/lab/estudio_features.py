"""Estudio de features de fernando_v1 en H: AUROC (todo H y zona gris), Pearson, VIF y pesos.

  python estudio_features.py
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import validar_cv as V  # noqa: E402
from entrenar import NOMBRES, SIGNO  # noqa: E402


def vif(X):
    """VIF_j = 1 / (1 - R²_j), regresando la columna j (estandarizada) sobre las demás + intercepto."""
    Xs = (X - X.mean(0)) / X.std(0)
    out = []
    for k in range(X.shape[1]):
        A = np.hstack([np.delete(Xs, k, 1), np.ones((len(Xs), 1))])
        b = Xs[:, k]
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        r2 = 1 - ((b - A @ beta) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        out.append(1 / (1 - r2) if r2 < 1 else float("inf"))
    return out


def main():
    M = V.cargar_modulo(os.path.join(AQUI, "..", "mi_bot.py"))
    casos = [c for c in map(json.loads, open(V.TRAIN, encoding="utf-8")) if c["id"].startswith("H")]
    g = M.Guardian()
    F = [g.extraer(V.caso_para_bot(c)) for c in casos]
    y = np.array([c["etiqueta"] for c in casos])
    gi = [i for i, f in enumerate(F) if f["_capa"] == 2]
    yg = y[gi]
    print(f"H: {len(casos)} casos ({y.sum()} alucinan). Zona gris: {len(gi)} ({yg.sum()} alucinan)")
    m = M.MODELO
    coef, esc = dict(zip(m["nombres"], m["coef"])), dict(zip(m["nombres"], m["escala"]))
    X = np.array([[F[i][n] for n in NOMBRES] for i in gi])
    vivas = [j for j in range(len(NOMBRES)) if X[:, j].std() > 1e-9]
    v = dict(zip([NOMBRES[j] for j in vivas], vif(X[:, vivas])))
    print("| feature | signo esperado | AUROC H | AUROC zona gris | media (gris) | VIF (todas) | peso (std) | peso/escala |")
    print("|---|---|---|---|---|---|---|---|")
    for j, n in enumerate(NOMBRES):
        xa = np.array([f[n] for f in F])
        w = f"{coef[n]:+.3f} | {coef[n] / esc[n]:+.2f}" if n in coef else "— | —"
        print(f"| {n} | {SIGNO[n]:+d} | {roc_auc_score(y, xa):.3f} | {roc_auc_score(yg, X[:, j]):.3f} | "
              f"{X[:, j].mean():.3f} | {v.get(n, float('nan')):.1f} | {w} |")
    C = np.corrcoef(X[:, vivas].T)
    print("\nPares con |r| > 0,8 (zona gris):")
    for a in range(len(vivas)):
        for b in range(a + 1, len(vivas)):
            if abs(C[a, b]) > 0.8:
                print(f"  {NOMBRES[vivas[a]]} ~ {NOMBRES[vivas[b]]}: r = {C[a, b]:+.3f}")
    Xm = np.array([[F[i][n] for n in m["nombres"]] for i in gi])
    Cm = np.corrcoef(Xm.T)
    print(f"\nModelo final ({m['variante']}, umbral {m['umbral']}): max |r| entre sus features = "
          f"{np.max(np.abs(Cm - np.eye(len(Cm)))):.3f}")
    for n, x in zip(m["nombres"], vif(Xm)):
        print(f"  VIF {n}: {x:.2f}")


if __name__ == "__main__":
    main()

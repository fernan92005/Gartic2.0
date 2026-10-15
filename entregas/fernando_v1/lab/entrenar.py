"""Entrena el especialista de la capa 2 de fernando_v1 y lo embebe en mi_bot.py.

  python entrenar.py              # compara variantes en CV agrupada, elige y escribe MODELO
  python entrenar.py --no-escribir

Variante = "tipo:clave=valor:...", p. ej. "logreg:C=0.1:pen=l2:noh=0.3" o
"hgb:depth=3:noh=1". noh = peso de los casos no-H (0 = solo H).
Dentro de ajustar() todo se decide SOLO con los casos que recibe (los de
entrenamiento de la partición): selección de features, signos y modelo.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import validar_cv as V  # noqa: E402

BOT = os.path.join(AQUI, "..", "mi_bot.py")

# Signo esperado: +1 = sospecha (más alto => alucina), -1 = respaldo
SIGNO = {"llm_conf_min": -1, "llm_null": +1, "llm_sin_fuente": +1, "nli_citada_min": -1,
         "nli_todas_min_orac": -1, "nli_todas_min_frag": -1, "nli_mejor_min_frag": -1,
         "nli_comb_min_frag": -1, "nli_contra_max": +1, "sim_min_frag": -1,
         "frag_sin_respaldo": +1, "num_cruzada": +1, "num_cruzada_llm": +1, "num_ausente": +1,
         "num_no_derivable_local": +1, "num_derivado_local": -1, "num_prop_fuera": +1, "matiz_num_perdido": +1,
         "matiz_modal_perdido": +1, "matiz_alcance_perdido": +1,
         "alcance_perdido_citada": +1, "modal_perdido_citada": +1, "universal_anadido": +1,
         "causal_anadido": +1, "comparativa_anadida": +1, "prop_raices_nuevas": +1,
         "nombre_nuevo": +1}
NOMBRES = list(SIGNO)


def parsear(variante):
    partes = variante.split(":")
    cfg = {"tipo": partes[0], "C": 0.1, "pen": "l2", "noh": 0.3, "depth": 3, "auc_min": 0.03, "rmax": 0.8}
    for p in partes[1:]:
        k, v = p.split("=")
        cfg[k] = v if k == "pen" else float(v)
    return cfg


def seleccionar(X, y, w, cfg):
    """Fuera AUROC ~0,5 (|AUROC-0,5| < auc_min) y una de cada par |r| > rmax."""
    aucs = {}
    for j, n in enumerate(NOMBRES):
        if np.std(X[:, j]) < 1e-9:
            continue
        a = roc_auc_score(y, X[:, j], sample_weight=w)
        if abs(a - 0.5) >= cfg["auc_min"]:
            aucs[n] = a
    keep = sorted(aucs, key=lambda n: -abs(aucs[n] - 0.5))
    elegidas = []
    for n in keep:
        j = NOMBRES.index(n)
        if all(abs(np.corrcoef(X[:, j], X[:, NOMBRES.index(m)])[0, 1]) <= cfg["rmax"] for m in elegidas):
            elegidas.append(n)
    return elegidas


def ajustar(casos, feats, variante="logreg"):
    cfg = parsear(variante)
    zona = [i for i, f in enumerate(feats) if f["_capa"] == 2.0]
    X = np.array([[feats[i].get(n, 0.0) for n in NOMBRES] for i in zona])
    y = np.array([casos[i]["etiqueta"] for i in zona])
    es_h = np.array([casos[i]["id"].startswith("H") for i in zona])
    w = np.where(es_h, 1.0, cfg["noh"]) * np.where(y == 1, 4.0, 1.0)  # coste 4:1
    m = w > 0
    X, y, w, es_h = X[m], y[m], w[m], es_h[m]
    # La selección mira solo la zona gris H de entrenamiento (la que se parece al test)
    elegidas = seleccionar(X[es_h], y[es_h], None, cfg)
    if cfg["tipo"] == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        cols = [NOMBRES.index(n) for n in elegidas]
        mod = HistGradientBoostingClassifier(
            max_depth=int(cfg["depth"]), learning_rate=0.05, max_iter=150, l2_regularization=1.0,
            min_samples_leaf=20, monotonic_cst=[SIGNO[n] for n in elegidas], random_state=0)
        mod.fit(X[:, cols], y, sample_weight=w)
        return {"tipo": "hgb", "obj": mod, "nombres": elegidas, "umbral": 0.5}
    while True:  # logística: ninguna feature con signo incoherente
        cols = [NOMBRES.index(n) for n in elegidas]
        mu, sd = X[:, cols].mean(0), X[:, cols].std(0)
        sd[sd < 1e-9] = 1.0
        lr = LogisticRegression(C=cfg["C"], penalty=cfg["pen"], solver="liblinear", max_iter=2000, random_state=0)
        lr.fit((X[:, cols] - mu) / sd, y, sample_weight=w)
        coef = lr.coef_[0]
        malas = [n for n, c in zip(elegidas, coef) if c * SIGNO[n] < 0]
        if not malas:
            break
        elegidas = [n for n in elegidas if n not in malas]
    vivas = [k for k, c in enumerate(coef) if abs(c) > 1e-9]
    return {"tipo": "logreg", "nombres": [elegidas[k] for k in vivas],
            "media": [float(mu[k]) for k in vivas], "escala": [float(sd[k]) for k in vivas],
            "coef": [float(coef[k]) for k in vivas], "sesgo": float(lr.intercept_[0]), "umbral": 0.5}


VARIANTES = ([f"logreg:C={c}:pen={p}:noh={h}" for c in (0.03, 0.1, 0.3, 1.0)
              for p in ("l2", "l1") for h in (0, 0.3, 1)] +
             [f"hgb:depth={d}:noh={h}" for d in (2, 3) for h in (0, 0.3, 1)])


def mejor_corte(y, s):
    best = (-9, 0.5)
    for t in np.arange(0.05, 0.951, 0.01):
        p = V.puntuacion(y, (s >= t).astype(int))[0]
        if p > best[0] + 1e-12:
            best = (p, float(t))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-escribir", action="store_true")
    ap.add_argument("--variantes", nargs="*")
    args = ap.parse_args(argv)
    casos = [json.loads(l) for l in open(V.TRAIN, encoding="utf-8") if l.strip()]
    res = []
    for var in args.variantes or VARIANTES:
        ad = V.Fernando(BOT, var)
        filas, glob, oof = V.validar(ad, casos)
        r = V.resumen(filas)
        res.append((r["puntuacion"][0], -r["puntuacion"][1], var, r, oof))
        print(f"{var:32s} {r['puntuacion'][0]:+.3f} ± {r['puntuacion'][1]:.3f}  AUROC "
              f"{r['auroc'][0]:.3f} ± {r['auroc'][1]:.3f}  FP {r['fp'][0]:.1f} FN {r['fn'][0]:.1f}",
              flush=True)
    # Regla de UN ERROR ESTÁNDAR (Breiman): las variantes cuya media está a menos de
    # 1 EE (= desv. entre particiones / raíz(15)) de la mejor son indistinguibles de ella;
    # entre ellas, la familia más simple (logística: coeficientes portables, sin pickle
    # atado a la versión de sklearn) y, dentro, la de mejor media y menor desviación.
    top = max(res, key=lambda x: x[0])
    ee = -top[1] / np.sqrt(15)
    cerca = [x for x in res if x[0] >= top[0] - ee]
    p, _, var, r, oof = min(cerca, key=lambda x: (not x[2].startswith("logreg"), -round(x[0], 3), -x[1]))
    print(f"Mejor media: {top[2]} {top[0]:+.3f} (EE {ee:.3f}); {len(cerca)} variantes a menos de 1 EE")
    ids = sorted(oof)
    y = np.array([next(c["etiqueta"] for c in casos if c["id"] == i) for i in ids])
    s = np.array([oof[i] for i in ids])
    p_opt, t_opt = mejor_corte(y, s)
    p_05 = V.puntuacion(y, (s >= 0.5).astype(int))[0]
    print(f"\nElegida: {var}  (CV {p:+.3f} ± {r['puntuacion'][1]:.3f})")
    print(f"Corte óptimo sobre las predicciones de VALIDACIÓN (media de semillas): {t_opt:.2f} "
          f"(puntuación {p_opt:+.3f}) frente a 0,50 ({p_05:+.3f})")
    ad = V.Fernando(BOT, var)
    ad.preparar(casos)
    modelo = ajustar(casos, ad.feats, var)
    # Reescalado solo si ayuda FUERA de muestra: CV anidada (umbral elegido con una CV
    # interna dentro de cada partición). Si no gana > 0,01, el corte queda en 0,5.
    import umbral_anidado
    p_fijo, p_anid = umbral_anidado.main(var)
    umbral = t_opt if (abs(t_opt - 0.5) > 0.05 and p_anid - p_fijo > 0.01) else 0.5
    print(f"Umbral final: {umbral:.2f} (anidado {p_anid:+.3f} frente a fijo {p_fijo:+.3f})")
    modelo["umbral"] = umbral
    modelo["variante"] = var
    print("Modelo final:", json.dumps({k: v for k, v in modelo.items() if k != "obj"}, indent=1))
    if modelo["tipo"] == "hgb":
        print("HGB elegido: hay que embeberlo serializado (no implementado: ver INFORME)")
        return
    neutro = {n: float(np.mean([f[n] for f in ad.feats if f["_capa"] == 2.0])) for n in NOMBRES}
    if not args.no_escribir:
        src = open(BOT, encoding="utf-8").read()
        bloque = ("# >>> MODELO\n# Generado por lab/entrenar.py (variante " + var + "). No editar a mano.\n"
                  "MODELO = " + json.dumps({k: v for k, v in modelo.items()}) + "\n# <<< MODELO")
        src = re.sub(r"# >>> MODELO.*?# <<< MODELO", lambda _: bloque, src, flags=re.S)
        src = re.sub(r"NEUTRO = .*?\n", lambda _: "NEUTRO = " + json.dumps(
            {k: round(v, 4) for k, v in neutro.items()}) + "  # medias de la zona gris (train)\n", src, count=1)
        open(BOT, "w", encoding="utf-8").write(src)
        print("Escrito en", os.path.abspath(BOT))


if __name__ == "__main__":
    main()

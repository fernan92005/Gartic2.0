"""Banco común de fernando_v2: evalúa una VARIANTE de joseca_v4 con CV agrupada.

  python banco.py <carpeta_variante>            # CV 5x3 en el 80 % de H + giro/test
  python banco.py <carpeta_variante> --holdout  # SOLO la sesión principal, fase 4

Una variante es una carpeta con:
  mi_bot.py    copia de entregas/joseca_v4/mi_bot.py con UN cambio (mismas funciones:
               caracteristicas, NOMBRES, probabilidad, reescalar, UMBRAL, Guardian)
  config.json  (opcional) cómo se entrena el modelo; por defecto, lo de joseca_v4:
     {"modelo": "gb",            # "gb" = GradientBoostingClassifier (v4) | "hgb" = HistGB (admite monotonic_cst)
      "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
      "peso_h": 5.0,            # peso de los casos H (no-H pesan 1)
      "peso_pos": 1.0,          # peso extra de las alucinaciones (B5: 4.0)
      "excluir": [],            # features de NOMBRES que no entran al modelo
      "monotonic": {},          # solo "hgb": {"feature": +1 sospecha | -1 respaldo}
      "poda": null}             # B2: {"signos": {"feature": +1|-1}, "auc_min": 0.03, "rmax": 0.8}
                                #     la poda se decide DENTRO de cada partición, solo con su entrenamiento

Cómo se puntúa: el banco SUSTITUYE mod.probabilidad por el modelo de la partición y
llama a mod.Guardian().score(caso). Así cualquier lógica de la variante (capa 1, reglas,
agregación por oraciones, correcciones) se evalúa tal cual, con el UMBRAL de la variante.

Datos: CV GroupKFold(5, shuffle) x semillas (0, 1, 2) SOLO sobre las familias H que NO
están en lab/holdout_familias.txt (solo lectura). Los no-H siempre entrenan. El hold-out
no se toca salvo con --holdout. Escribe <carpeta>/resultado.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))
KIT = os.path.join(RAIZ, "kit")
SEMILLAS = (0, 1, 2)
DEFECTO = {"modelo": "gb", "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
           "peso_h": 5.0, "peso_pos": 1.0, "excluir": [], "monotonic": {}, "poda": None}


def leer(nombre):
    return [json.loads(l) for l in open(os.path.join(KIT, nombre), encoding="utf-8") if l.strip()]


def grupo(cid):
    return cid.split("::")[0]


def para_bot(c):
    return {k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales") if k in c}


def puntuacion(y, s):
    y, p = np.asarray(y), (np.asarray(s) >= 0.5)
    fn, fp = int(((y == 1) & ~p).sum()), int(((y == 0) & p).sum())
    return 1.0 - (4 * fn + fp) / int((y == 0).sum()), fp, fn


def cargar(ruta):
    nombre = "var_" + str(abs(hash(ruta)))
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


class Variante:
    def __init__(self, carpeta):
        self.carpeta = os.path.abspath(carpeta)
        self.mod = cargar(os.path.join(self.carpeta, "mi_bot.py"))
        cfg = os.path.join(self.carpeta, "config.json")
        self.cfg = dict(DEFECTO, **(json.load(open(cfg, encoding="utf-8")) if os.path.exists(cfg) else {}))
        self.cols = [n for n in self.mod.NOMBRES if n not in self.cfg["excluir"]]
        # caché de características por objeto caso (los casos se reutilizan entre particiones)
        orig, cache = self.mod.caracteristicas, {}

        def memo(caso):
            k = id(caso)
            if k not in cache:
                cache[k] = (caso, orig(caso))  # se guarda el caso para que su id no se recicle
            return cache[k][1]
        self.mod.caracteristicas = memo
        self.orig_caracteristicas = orig

    def X(self, casos):
        return np.array([[float(self.mod.caracteristicas(c).get(n, 0.0)) for n in self.cols] for c in casos],
                        dtype=np.float32)

    def podar(self, X, y, es_h):
        """Criterio de fernando_v1 sobre la zona gris H del ENTRENAMIENTO: fuera AUROC ~0,5,
        signo contrario al esperado y una de cada par |r| > rmax (se queda la más discriminante)."""
        pc = self.cfg["poda"]
        llm_neg = [n for n in ("llm_no_respaldada", "llm_contradicha") if n in self.cols]
        gris = es_h.copy()
        for n in llm_neg:
            gris &= X[:, self.cols.index(n)] == 0
        Xg, yg = X[gris], y[gris]
        aucs = {}
        for j, n in enumerate(self.cols):
            if np.std(Xg[:, j]) < 1e-9:
                continue
            a = roc_auc_score(yg, Xg[:, j])
            sg = pc.get("signos", {}).get(n, 0)
            if abs(a - 0.5) < pc.get("auc_min", 0.03) or (sg and (a - 0.5) * sg < 0):
                continue
            aucs[n] = a
        keep = []
        for n in sorted(aucs, key=lambda n: -abs(aucs[n] - 0.5)):
            j = self.cols.index(n)
            if all(abs(np.corrcoef(Xg[:, j], Xg[:, self.cols.index(m)])[0, 1]) <= pc.get("rmax", 0.8) for m in keep):
                keep.append(n)
        keep += [n for n in llm_neg if n not in keep]  # las del LLM negativo no se evalúan en la zona gris
        return [n for n in self.cols if n in keep]

    def ajustar(self, casos, y, es_h):
        cfg = self.cfg
        self.cols = [n for n in self.mod.NOMBRES if n not in cfg["excluir"]]
        X = self.X(casos)
        y = np.asarray(y)
        if cfg["poda"]:
            self.cols = self.podar(X, y, np.asarray(es_h))
            self.podadas = self.cols
            X = self.X(casos)
        w = np.where(es_h, cfg["peso_h"], 1.0) * np.where(y == 1, cfg["peso_pos"], 1.0)
        if cfg["modelo"] == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(
                max_iter=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"],
                min_samples_leaf=20, l2_regularization=0.0, early_stopping=False, random_state=0,
                monotonic_cst=[int(cfg["monotonic"].get(n, 0)) for n in self.cols])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                           learning_rate=cfg["learning_rate"], subsample=0.8,
                                           min_samples_leaf=20, random_state=0)
        m.fit(X, y, sample_weight=w)
        cols = self.cols

        def probabilidad(feats):
            x = np.array([[float(feats.get(n, 0.0)) for n in cols]], dtype=np.float32)
            return float(m.predict_proba(x)[0, 1])
        self.mod.probabilidad = probabilidad
        return m

    def puntuar(self, casos):
        g = self.mod.Guardian()
        return np.array([min(1.0, max(0.0, float(g.score(c)))) for c in casos])


def particiones(familias):
    """GroupKFold con shuffle real; comprueba que las semillas dan repartos distintos."""
    out, firmas = [], []
    fam = np.array(familias)
    for s in SEMILLAS:
        partes = list(GroupKFold(5, shuffle=True, random_state=s).split(fam, groups=fam))
        firmas.append(tuple(sorted(tuple(sorted(set(fam[te]))) for _, te in partes)))
        out.append(partes)
    assert len(set(firmas)) == len(SEMILLAS), "las semillas dan las mismas particiones"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("variante")
    ap.add_argument("--holdout", action="store_true", help="solo la sesión principal (fase 4)")
    args = ap.parse_args(argv)
    t0 = time.time()
    v = Variante(args.variante)
    train = leer("train.jsonl")
    for c in train:
        c["_b"] = para_bot(c)
    hold = {l.strip() for l in open(os.path.join(AQUI, "holdout_familias.txt")) if l.strip()}
    H = [c for c in train if c["id"].startswith("H") and grupo(c["id"]) not in hold]
    HO = [c for c in train if c["id"].startswith("H") and grupo(c["id"]) in hold]
    noH = [c for c in train if not c["id"].startswith("H")]

    def fit(casos_h):
        casos = casos_h + noH
        v.ajustar([c["_b"] for c in casos], [c["etiqueta"] for c in casos],
                  np.array([c["id"].startswith("H") for c in casos]))

    if args.holdout:
        fit(H)
        y = [c["etiqueta"] for c in HO]
        s = v.puntuar([c["_b"] for c in HO])
        p, fp, fn = puntuacion(y, s)
        r = {"n": len(HO), "familias": len(hold), "puntuacion": p, "auroc": roc_auc_score(y, s), "fp": fp, "fn": fn,
             "scores": {c["id"]: float(x) for c, x in zip(HO, s)}}
        json.dump(r, open(os.path.join(v.carpeta, "holdout.json"), "w"), indent=1)
        print(f"HOLD-OUT {os.path.basename(v.carpeta)}: n={len(HO)} punt {p:+.3f} AUROC {r['auroc']:.3f} FP {fp} FN {fn}")
        return

    fam = sorted({grupo(c["id"]) for c in H})
    por_fam = {f: [c for c in H if grupo(c["id"]) == f] for f in fam}
    pliegues = []
    for partes in particiones(fam):
        for tr, te in partes:
            tr_c = [c for i in tr for c in por_fam[fam[i]]]
            te_c = [c for i in te for c in por_fam[fam[i]]]
            fit(tr_c)
            y = [c["etiqueta"] for c in te_c]
            s = v.puntuar([c["_b"] for c in te_c])
            p, fp, fn = puntuacion(y, s)
            pliegues.append({"puntuacion": p, "auroc": roc_auc_score(y, s), "fp": fp, "fn": fn,
                             "ids": [c["id"] for c in te_c], "scores": [float(x) for x in s]})
    # modelo con todo el 80 % de H + no-H (nunca el hold-out) para test y giro
    fit(H)
    test, giro = leer("test_publico.jsonl"), leer("giro_publico.jsonl")
    st = v.puntuar([para_bot(c) for c in test])
    sg = v.puntuar([para_bot(c) for c in giro])
    nor = np.array([len(c["senales"]["oraciones"]) for c in giro])
    P = np.array([f["puntuacion"] for f in pliegues])
    res = {"variante": os.path.basename(v.carpeta), "config": v.cfg, "n_features": len(v.cols),
           "cv_media": float(P.mean()), "cv_desv": float(P.std()),
           "auroc": float(np.mean([f["auroc"] for f in pliegues])),
           "fp": int(sum(f["fp"] for f in pliegues)), "fn": int(sum(f["fn"] for f in pliegues)),
           "pliegues": pliegues,
           "giro_marcado": float(np.mean(sg >= .5)), "giro_marcado_3": float(np.mean(sg[nor == 3] >= .5)),
           "giro_marcado_4": float(np.mean(sg[nor == 4] >= .5)), "test_marcado": float(np.mean(st >= .5)),
           "test_scores": {c["id"]: float(x) for c, x in zip(test, st)},
           "giro_scores": {c["id"]: float(x) for c, x in zip(giro, sg)},
           "segundos": round(time.time() - t0)}
    json.dump(res, open(os.path.join(v.carpeta, "resultado.json"), "w"))
    print(f"{res['variante']}: CV {P.mean():+.3f} ± {P.std():.3f} AUROC {res['auroc']:.3f} FP {res['fp']} FN {res['fn']} | "
          f"giro marcado {res['giro_marcado']:.1%} (3 or. {res['giro_marcado_3']:.1%}, 4 or. {res['giro_marcado_4']:.1%}) | "
          f"test marcado {res['test_marcado']:.1%} | {res['segundos']} s")


if __name__ == "__main__":
    main()

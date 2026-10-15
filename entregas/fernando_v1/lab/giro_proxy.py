"""Banco de pruebas del GIRO con etiquetas: respuestas largas sintéticas, fuera de muestra.

El giro trae respuestas de 3-4 oraciones y no tiene etiquetas. Lo imitamos juntando
k = 3 o 4 respuestas H del MISMO documento (mismas fuentes):
  - k fieles                      -> fiel
  - 1 alucinada + (k-1) fieles    -> alucina
En cada partición (GroupKFold por documento, 5 x 3 semillas, como validar_cv.py) el
modelo se entrena SIN los documentos con los que se construyen los casos sintéticos.

Para cada bot se mide:
  - "completo": el bot puntúa la respuesta larga entera;
  - "por_oracion": se puntúa cada oración como un caso propio (sus fragmentos, las
    mismas fuentes) y el score es el MÁXIMO (basta una oración falsa).
Métricas: % falsas alarmas sobre las fieles (f), recall sobre las alucinadas (r) y la
puntuación esperada con una prevalencia pi de alucinaciones (la del test es 0,40):
    puntuación = 1 - f - 4 * pi / (1 - pi) * (1 - r)

  python giro_proxy.py [--bot ruta ...] [--curva]
"""
import argparse
import json
import os
import random
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import validar_cv as V  # noqa: E402
from estres_giro import concatenar  # noqa: E402

PI = 0.40


def sinteticos(H, docs, rng, por_doc=4):
    out = []
    for d in docs:
        cs = [c for c in H if V.grupo(c["id"]) == d]
        fie = [c for c in cs if c["etiqueta"] == 0]
        mal = [c for c in cs if c["etiqueta"] == 1]
        for k in (3, 4):
            for _ in range(por_doc):
                if len(fie) >= k:
                    out.append(concatenar(rng.sample(fie, k)))
                if mal and len(fie) >= k - 1:
                    g = rng.sample(fie, k - 1) + [rng.choice(mal)]
                    rng.shuffle(g)
                    out.append(concatenar(g))
    return out


def oraciones(c):
    """Cada oración como caso propio (con sus fragmentos)."""
    ors = c["senales"]["oraciones"]
    return [{"pregunta": c.get("pregunta", ""), "fuentes": c["fuentes"], "respuesta": o.get("texto", ""),
             "senales": {"oraciones": [o], "fragmentos": [dict(f, oracion=0) for f in c["senales"]["fragmentos"]
                                                         if f.get("oracion") == k]}}
            for k, o in enumerate(ors)]


class Puntuador:
    """Puntúa casos NUEVOS con el modelo que el adaptador acaba de ajustar."""

    def __init__(self, ad):
        self.ad, self.cache = ad, {}

    def _x(self, c):
        k = id(c)
        if k not in self.cache:
            if isinstance(self.ad, V.Fernando):
                self.cache[k] = self.ad.mod.Guardian().extraer(c)
            else:
                f = self.ad.mod.caracteristicas(c)
                self.cache[k] = [f.get(n, 0.0) for n in self.ad.mod.NOMBRES]
        return self.cache[k]

    def puntuar(self, casos):
        if isinstance(self.ad, V.Fernando):
            g = self.ad.mod.Guardian(modelo=self.ad.modelo)
            return np.array([g.puntuar_feats(self._x(c)) for c in casos])
        X = np.array([self._x(c) for c in casos])
        p = self.ad.modelo.predict_proba(X)[:, 1] if self.ad.gb else self.ad.ent.predecir(X, self.ad.modelo)
        return np.array([self.ad.mod.reescalar(v, self.ad.mod.UMBRAL) for v in p])


def puntuacion(f, r, pi=PI):
    return 1 - f - 4 * pi / (1 - pi) * (1 - r)


def evaluar(ruta, casos, H, curva=False):
    ad = V.elegir_adaptador(ruta, None)
    ad.preparar(casos)
    P = Puntuador(ad)
    idx_h = np.array([i for i, c in enumerate(casos) if c["id"].startswith("H")])
    idx_no = np.array([i for i, c in enumerate(casos) if not c["id"].startswith("H")])
    g = np.array([V.grupo(casos[i]["id"]) for i in idx_h])
    y_all = np.array([c["etiqueta"] for c in casos])
    res = {"completo": [], "por_oracion": [], "una_oracion": []}
    curvas = []
    for semilla in V.SEMILLAS:
        rng = random.Random(semilla)
        for tr, te in GroupKFold(5, shuffle=True, random_state=semilla).split(idx_h, groups=g):
            ad.ajustar(np.concatenate([idx_h[tr], idx_no]))
            s1 = ad.predecir(idx_h[te])
            res["una_oracion"].append(V.puntuacion(y_all[idx_h[te]], (s1 >= .5).astype(int))[0])
            sint = sinteticos(H, sorted(set(g[te])), rng)
            y = np.array([c["etiqueta"] for c in sint])
            sc = P.puntuar(sint)
            subs = [oraciones(c) for c in sint]
            plano = [o for s in subs for o in s]
            so = P.puntuar(plano)
            cortes = np.cumsum([0] + [len(s) for s in subs])
            sm = np.array([so[a:b].max() for a, b in zip(cortes[:-1], cortes[1:])])
            for nom, s in (("completo", sc), ("por_oracion", sm)):
                f, r = np.mean(s[y == 0] >= .5), np.mean(s[y == 1] >= .5)
                res[nom].append((f, r, puntuacion(f, r)))
            if curva:
                curvas.append([(t, np.mean(sm[y == 0] >= t), np.mean(sm[y == 1] >= t))
                               for t in np.arange(0.5, 0.96, 0.05)])
    return res, curvas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", nargs="*", default=[os.path.join(V.RAIZ, "entregas", b, "mi_bot.py")
                                                 for b in ("fernando_v1", "joseca_v3", "joseca_v4")])
    ap.add_argument("--curva", action="store_true", help="umbral del máximo por oración (solo informativo)")
    args = ap.parse_args(argv)
    casos = [json.loads(l) for l in open(V.TRAIN, encoding="utf-8") if l.strip()]
    H = [c for c in casos if c["id"].startswith("H")]
    print(f"Prevalencia supuesta en el giro: {PI:.0%} (la del test). Fuera de muestra, 5x3 particiones.")
    print(f"{'bot':14s} {'test-like (1 oración)':>22s} | {'giro: modo':12s} {'falsas al.':>10s} {'recall':>7s} "
          f"{'puntuación giro':>16s} | {'combinada*':>10s}")
    for ruta in args.bot:
        nombre = os.path.basename(os.path.dirname(ruta))
        res, curvas = evaluar(ruta, casos, H, args.curva)
        u = np.array(res["una_oracion"])
        for modo in ("completo", "por_oracion"):
            a = np.array(res[modo])
            comb = (265 * u.mean() + 132 * a[:, 2].mean()) / 397
            print(f"{nombre:14s} {u.mean():+.3f} ± {u.std():.3f}        | {modo:12s} {a[:, 0].mean():9.1%} "
                  f"{a[:, 1].mean():7.1%} {a[:, 2].mean():+9.3f} ± {a[:, 2].std():.3f} | {comb:+10.3f}")
        if args.curva:
            c = np.array(curvas)  # particiones x cortes x (t, f, r)
            for j in range(c.shape[1]):
                t, f, r = c[:, j, 0].mean(), c[:, j, 1].mean(), c[:, j, 2].mean()
                print(f"    corte del máximo {t:.2f}: falsas al. {f:5.1%} recall {r:5.1%} -> {puntuacion(f, r):+.3f}")
    print("* combinada = media ponderada por nº de fieles: test (265 fieles) + giro (~132 si pi = 0,40).")


if __name__ == "__main__":
    main()

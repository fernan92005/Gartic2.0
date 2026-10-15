"""Estrés para el giro (respuestas más largas, documentos técnicos/legales).

  python estres_giro.py --bot ../mi_bot.py [--bot ...] [--giro ../../../kit/giro_publico.jsonl]

1. Subgrupos de H con predicciones FUERA DE MUESTRA (lab/resultados/oof_*.json, que
   escribe validar_cv.py --guardar-oof): 1 oración vs >=2, y recall por tipo.
2. Respuestas largas sintéticas: concatena 2-4 respuestas H FIELES del mismo
   documento (mismas fuentes; siguen siendo fieles) y mide cuántas marca el bot.
   Con una alucinación dentro, mide cuántas detecta (escape = ×4).
3. Deriva: proporción marcada (score >= 0,5) e histograma por deciles en train-H
   frente a test_publico (y giro si existe). Una diferencia grande es una alarma.
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import validar_cv as V  # noqa: E402


def concatenar(grupo):
    base = grupo[0]
    sen = {"oraciones": [], "fragmentos": []}
    off = 0
    for c in grupo:
        sen["oraciones"] += c["senales"]["oraciones"]
        for f in c["senales"]["fragmentos"]:
            sen["fragmentos"].append(dict(f, oracion=f.get("oracion", 0) + off))
        off += len(c["senales"]["oraciones"])
    return {"pregunta": base["pregunta"], "fuentes": base["fuentes"],
            "respuesta": " ".join(c["respuesta"].rstrip(".") + "." for c in grupo),
            "senales": sen, "etiqueta": int(any(c["etiqueta"] for c in grupo))}


def sinteticos(H, k, con_alucinacion, rng, n=200):
    por_doc = {}
    for c in H:
        por_doc.setdefault(V.grupo(c["id"]), []).append(c)
    out = []
    for doc, cs in sorted(por_doc.items()):
        fieles = [c for c in cs if c["etiqueta"] == 0 and c["fuentes"] == cs[0]["fuentes"]]
        malas = [c for c in cs if c["etiqueta"] == 1 and c["fuentes"] == cs[0]["fuentes"]]
        need = k - 1 if con_alucinacion else k
        if len(fieles) < need or (con_alucinacion and not malas):
            continue
        g = rng.sample(fieles, need) + ([rng.choice(malas)] if con_alucinacion else [])
        rng.shuffle(g)
        out.append(concatenar(g))
    return out[:n]


def deciles(s):
    h, _ = np.histogram(s, bins=np.linspace(0, 1, 11))
    return " ".join(f"{100 * v / len(s):4.1f}" for v in h)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", action="append", required=True)
    ap.add_argument("--giro", default=os.path.join(V.RAIZ, "kit", "giro_publico.jsonl"))
    args = ap.parse_args(argv)
    casos = [json.loads(l) for l in open(V.TRAIN, encoding="utf-8") if l.strip()]
    H = [c for c in casos if c["id"].startswith("H")]
    test = [json.loads(l) for l in open(os.path.join(V.RAIZ, "kit", "test_publico.jsonl"), encoding="utf-8")]
    giro = [json.loads(l) for l in open(args.giro, encoding="utf-8")] if os.path.exists(args.giro) else []
    for ruta in args.bot:
        ruta = os.path.abspath(ruta)
        nombre = os.path.relpath(ruta, V.RAIZ).replace("\\", "/").replace("/mi_bot.py", "")
        print(f"\n===== {nombre}")
        bot = V.cargar_modulo(ruta).Guardian()
        exc = [0]
        # 1. subgrupos con OOF
        fich = os.path.join(AQUI, "resultados", "oof_" + re.sub(r"[^A-Za-z0-9_.-]", "_", nombre) + ".json")
        if os.path.exists(fich):
            oof = json.load(open(fich))
            for nom, sub in (("1 oración", [c for c in H if len(c["senales"]["oraciones"]) == 1]),
                             (">=2 oraciones", [c for c in H if len(c["senales"]["oraciones"]) >= 2])):
                y = np.array([c["etiqueta"] for c in sub])
                s = np.array([oof[c["id"]] for c in sub])
                m = V.metricas(y, s) if len(set(y)) > 1 else None
                print(f"  OOF {nom:14s} n={len(sub):3d} " + (f"punt {m['puntuacion']:+.3f} AUROC {m['auroc']:.3f} "
                      f"FP {m['fp']} FN {m['fn']}" if m else f"recall {np.mean(s[y == 1] >= .5):.2f}"))
            for t in ("distorsion", "sobregeneralizacion", "no_respaldado", "contradiccion"):
                s = np.array([oof[c["id"]] for c in H if c.get("tipo") == t])
                print(f"  OOF recall {t:20s} {np.mean(s >= .5):.2f} (n={len(s)})")
        else:
            print("  (sin OOF: ejecuta validar_cv.py --guardar-oof)")
        # 2. respuestas largas sintéticas (score del bot final: dentro de muestra para H)
        rng = random.Random(0)
        for k in (2, 3, 4):
            fie = sinteticos(H, k, False, rng)
            mal = sinteticos(H, k, True, rng)
            sf = V.puntuar_seguro(bot, fie, exc)
            sm = V.puntuar_seguro(bot, mal, exc)
            print(f"  {k} resp. concatenadas: fieles marcadas {100 * np.mean(sf >= .5):5.1f}% (n={len(fie)}) · "
                  f"con 1 alucinación detectadas {100 * np.mean(sm >= .5):5.1f}% (n={len(mal)})")
        s1 = V.puntuar_seguro(bot, [c for c in H if c["etiqueta"] == 0], exc)
        print(f"  referencia 1 resp.: fieles marcadas {100 * np.mean(s1 >= .5):5.1f}% (dentro de muestra)")
        # 3. deriva
        for nom, data in (("train-H", H), ("test_publico", test), ("giro", giro)):
            if not data:
                continue
            s = V.puntuar_seguro(bot, data, exc)
            print(f"  {nom:12s} n={len(data):3d} marcados {100 * np.mean(s >= .5):5.1f}%  deciles: {deciles(s)}")
        print(f"  excepciones: {exc[0]}" + (f" · contador interno del bot: {bot.excepciones}"
                                           if hasattr(bot, "excepciones") else ""))


if __name__ == "__main__":
    main()

"""Calcula CDF empiricas POR ELEMENTO (101 cuantiles) y n0 (mediana de nº de elementos) en H del train,
EXCLUYENDO las familias de lab/holdout_familias.txt y sin usar etiquetas. Escribe cdf_const.py y comprueba que
las extracciones reproducen las features originales de joseca_v4 (min/max)."""
import json, os, sys, importlib.util
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, *[".."] * 5))
spec = importlib.util.spec_from_file_location("v4", os.path.join(RAIZ, "entregas", "joseca_v4", "mi_bot.py"))
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
hold = {l.strip() for l in open(os.path.join(AQUI, "..", "..", "holdout_familias.txt")) if l.strip()}
train = [json.loads(l) for l in open(os.path.join(RAIZ, "kit", "train.jsonl"), encoding="utf-8") if l.strip()]
H = [c for c in train if c["id"].startswith("H") and c["id"].split("::")[0] not in hold]


def elementos(caso):
    """dict poblacion -> lista de valores por elemento (misma logica que caracteristicas())."""
    s = caso.get("senales") or {}
    ors, frs = s.get("oraciones") or [], s.get("fragmentos") or []
    nf = len(caso.get("fuentes") or [])
    ver = [((o.get("llm") or {}).get("veredicto")) for o in ors]
    con = [((o.get("llm") or {}).get("confianza") or 0.0) for o in ors]
    citada = []
    for o in ors:
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < nf:
            rpf = o.get("respaldo_por_fuente") or []
            if i < len(rpf):
                citada.append(rpf[i])
    return {
        "llm_conf": [c for v, c in zip(ver, con) if v == "respaldada"],
        "orac_resp": [o.get("respaldo_todas", 0.0) for o in ors],
        "frag_resp": [x.get("respaldo_todas", 0.0) for x in frs],
        "orac_contr": [o.get("contradiccion_todas", 0.0) for o in ors],
        "frag_contr": [x.get("contradiccion_todas", 0.0) for x in frs],
        "orac_mejor": [max(o.get("respaldo_por_fuente") or [0.0]) for o in ors],
        "frag_contr_fuente": [max(x.get("contradiccion_por_fuente") or [0.0]) for x in frs],
        "citada": citada,
        "frag_mejor": [max([x.get("respaldo_todas", 0.0)] + list(x.get("respaldo_por_fuente") or [])) for x in frs],
        "frag_sim": [max(x.get("similitud_por_fuente") or [0.0]) for x in frs],
    }

pob, cuentas = {}, {}
for c in H:
    for k, v in elementos(c).items():
        pob.setdefault(k, []).extend(v)
        cuentas.setdefault(k, []).append(len(v))
# comprobacion: min/max de las poblaciones por caso == features originales
chk = {"llm_conf": ("llm_conf_min_respaldada", min), "orac_resp": ("nli_resp_todas_min_orac", min),
       "orac_contr": ("nli_contr_todas_max_orac", max), "frag_contr": ("nli_contr_todas_max_frag", max),
       "orac_mejor": ("nli_mejor_fuente_min_orac", min), "frag_contr_fuente": ("nli_contr_fuente_max_frag", max),
       "citada": ("nli_resp_fuente_citada_min", min), "frag_mejor": ("frag_respaldo_min", min),
       "frag_sim": ("frag_similitud_min", min)}
for c in H[:400]:
    f, e = v4.caracteristicas(c), elementos(c)
    for k, (nom, fn) in chk.items():
        esperado = fn(e[k]) if e[k] else 0.0
        assert abs(f[nom] - esperado) < 1e-9, (c["id"], nom, f[nom], esperado)
Q = {k: [round(float(x), 4) for x in np.quantile(np.array(v), np.linspace(0, 1, 101))] for k, v in pob.items()}
N0 = {k: float(np.median(v)) for k, v in cuentas.items()}
print("n casos H:", len(H), "N0:", N0, {k: len(v) for k, v in pob.items()})
open(os.path.join(AQUI, "cdf_const.py"), "w", encoding="utf-8").write(
    "CDF_Q = " + repr(Q) + "\nCDF_N0 = " + repr(N0) + "\n")

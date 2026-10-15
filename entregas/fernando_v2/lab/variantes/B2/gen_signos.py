import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "joseca_v4"))
src = open(os.path.join(os.path.dirname(__file__), "mi_bot.py"), encoding="utf-8").read()
NOM = eval(re.search(r"^NOMBRES = (\[.*?\])$", src, re.M).group(1))
# signo por SIGNIFICADO (+1 sospecha, -1 respaldo, 0 ambiguo); llmok_X hereda el de X
BASE = {
 "llm_no_respaldada": 1, "llm_contradicha": 1, "llm_conf_min_respaldada": -1,
 "nli_resp_todas_min_frag": -1, "nli_contr_todas_max_frag": 1, "nli_mejor_fuente_min_orac": -1,
 "nli_contr_fuente_max_frag": 1, "nli_resp_fuente_citada_min": -1,
 "frag_respaldo_min": -1, "frag_respaldo_2o": -1, "frag_frac_bajo_05": 1, "frag_frac_bajo_02": 1,
 "frag_similitud_min": -1,
 "num_inexplicados": 1, "num_derivado_casual": 1, "num_sospechosos_grandes": 1,
 "num_derivado_local": -1, "num_derivado_anclado": -1, "num_cuenta_falsa": 1, "num_cuenta_correcta": -1,
 "num_hay_cifras": 0, "verif_cuenta_mal": 1, "verif_cuenta_ok": -1,
 "entidad_ajena": 1, "entidad_ajena_fuerte": 1, "nombre_inventado": 1,
 "fecha_plazo_falso": 1, "fecha_plazo_correcto": -1, "hora_inventada": 1,
 "huecos_tramo_max": 0, "matiz_oracion_perdido": 1,
 "anadido_universal": 1, "anadido_causal": 1, "anadido_causal_todas": 1,
 "palabras_nuevas_frac": 0, "llm_ok_y_num_sospechoso": 1, "llm_ok_y_nli_bajo": 1,
 "llm_contra_sin_apoyo": 0, "n_oraciones": 0,
}
def signo(n):
    if n.startswith("matiz_perdido_"): return 1
    if n in BASE: return BASE[n]
    if n.startswith("llmok_"): return signo(n[6:])
    raise KeyError(n)
S = {n: signo(n) for n in NOM}
json.dump(S, open(os.path.join(os.path.dirname(__file__), "signos.json"), "w"), indent=1)
v = list(S.values()); print(len(S), "+1:", v.count(1), "-1:", v.count(-1), "0:", v.count(0))
print("sin signo 0:", [n for n in S if S[n] == 0])

"""Tests de fernando_v1:  python -m pytest entregas/fernando_v1/lab/tests -q"""
import ast
import copy
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import pytest

AQUI = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(AQUI)
BOT_DIR = os.path.dirname(LAB)
BOT = os.path.join(BOT_DIR, "mi_bot.py")
CSV = os.path.join(BOT_DIR, "predicciones.csv")
RAIZ = os.path.abspath(os.path.join(BOT_DIR, "..", ".."))
KIT = os.path.join(RAIZ, "kit")
GIRO = os.path.join(KIT, "giro_publico.jsonl")
CV_MINIMO = 0.30  # versión congelada: +0,362 ± 0,095 en CV agrupada; margen ≈ 0,6 desviaciones


def leer(ruta):
    return [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]


def cargar():
    spec = importlib.util.spec_from_file_location("bot_fernando_test", BOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return cargar()


@pytest.fixture(scope="module")
def train():
    return leer(os.path.join(KIT, "train.jsonl"))


@pytest.fixture(scope="module")
def test():
    casos = leer(os.path.join(KIT, "test_publico.jsonl"))
    return casos + (leer(GIRO) if os.path.exists(GIRO) else [])


def para_bot(c):
    return {k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales") if k in c}


# ------------------------------------------------------------------ CSV
def test_csv_formato(test):
    filas = list(csv.reader(open(CSV, encoding="utf-8")))
    assert filas[0] == ["id", "score"]
    ids = [f[0] for f in filas[1:]]
    assert len(ids) == len(set(ids)), "ids repetidos"
    assert set(ids) == {c["id"] for c in test}, "faltan o sobran ids (¿regeneraste tras el giro?)"
    for cid, s in filas[1:]:
        assert re.fullmatch(r"\d+\.\d+", s), f"formato de score raro en {cid}: {s}"
        assert 0.0 <= float(s) <= 1.0
    assert ";" not in open(CSV, encoding="utf-8").read()


def test_csv_coincide_con_el_bot(mod, test):
    g = mod.Guardian()
    esperado = {c["id"]: f"{float(g.score(para_bot(c))):.6f}" for c in test}
    leido = {f[0]: f[1] for f in list(csv.reader(open(CSV, encoding="utf-8")))[1:]}
    assert leido == esperado


# ------------------------------------------------------------------ scores
def test_scores_en_rango_y_sin_excepciones(mod, train, test):
    g = mod.Guardian()
    for c in train + test:
        s = g.score(para_bot(c))
        assert isinstance(s, float) and 0.0 <= s <= 1.0
    assert g.excepciones == 0


def test_determinista(tmp_path, test):
    salidas = []
    for k in range(2):
        out = tmp_path / f"p{k}.csv"
        rutas = [os.path.join(KIT, "test_publico.jsonl")] + ([GIRO] if os.path.exists(GIRO) else [])
        subprocess.run([sys.executable, "generar_predicciones.py", "--bot", BOT, "--test", *rutas,
                        "--salida", str(out)], cwd=KIT, check=True, capture_output=True)
        salidas.append(out.read_text(encoding="utf-8"))
    assert salidas[0] == salidas[1]


def test_funciona_desde_la_raiz():
    r = subprocess.run([sys.executable, "-c",
                        "import importlib.util as u,json;s=u.spec_from_file_location('b','entregas/fernando_v1/mi_bot.py');"
                        "m=u.module_from_spec(s);s.loader.exec_module(m);"
                        "c=json.loads(open('kit/test_publico.jsonl',encoding='utf-8').readline());"
                        "print(m.Guardian().score(c))"], cwd=RAIZ, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert 0 <= float(r.stdout) <= 1


# ------------------------------------------------------------------ reglas
STDLIB = set(getattr(sys, "stdlib_module_names", ())) or {"math", "re", "unicodedata"}


def test_imports_permitidos():
    arbol = ast.parse(open(BOT, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
        elif isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("__import__", "eval", "exec"):
            pytest.fail(f"llamada prohibida: {n.func.id}")
    malos = {m for m in mods if m not in STDLIB and m not in ("numpy", "sklearn")}
    assert not malos, malos


def test_sin_ficheros_ni_red():
    src = open(BOT, encoding="utf-8").read()
    for prohibido in ("open(", "urllib", "socket", "subprocess", "train.jsonl", "__file__"):
        assert prohibido not in src, prohibido


def test_sin_caracteres_de_control():
    src = open(BOT, encoding="utf-8").read()
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", src)


def test_sin_ids_ni_frases_del_test():
    src = open(BOT, encoding="utf-8").read()
    assert not re.search(r"\bt0\d+", src)
    texto_bot = " ".join(re.findall(r"\w+", src.lower()))
    casos = leer(os.path.join(KIT, "test_publico.jsonl")) + (leer(GIRO) if os.path.exists(GIRO) else [])
    copiados = set()
    for c in casos:
        for t in [c["respuesta"], c["pregunta"]] + c["fuentes"]:
            w = re.findall(r"\w+", t.lower())
            for i in range(len(w) - 5):
                ng = " ".join(w[i:i + 6])
                if ng in texto_bot:
                    copiados.add(ng)
    assert not copiados, sorted(copiados)[:5]


def test_modelo_coherente(mod):
    m = mod.MODELO
    assert m and m["tipo"] == "logreg" and 0.0 < m["umbral"] < 1.0
    sys.path.insert(0, LAB)
    from entrenar import SIGNO
    for n, c in zip(m["nombres"], m["coef"]):
        assert c * SIGNO[n] >= 0, f"{n} con signo incoherente ({c:+.3f})"


# ------------------------------------------------------------------ robustez
def base(train):
    return copy.deepcopy(next(c for c in train if c["id"].startswith("H")))


def sinteticos(train):
    c0 = para_bot(base(train))
    out = {}
    c = copy.deepcopy(c0)
    for o in c["senales"]["oraciones"]:
        o["llm"] = None
    out["llm_none"] = c
    c = copy.deepcopy(c0)
    for o in c["senales"]["oraciones"]:
        o["llm"] = {"veredicto": None, "fuente": None, "confianza": None}
    out["veredicto_null"] = c
    c = copy.deepcopy(c0)
    c["senales"]["fragmentos"] = []
    out["sin_fragmentos"] = c
    c = copy.deepcopy(c0)
    c["respuesta"] = "La agrupación tiene muchos miembros y una sede propia."
    for o in c["senales"]["oraciones"] + c["senales"]["fragmentos"]:
        o["texto"] = c["respuesta"]
    out["sin_cifras"] = c
    c = copy.deepcopy(c0)
    c["fuentes"] = c["fuentes"][:1]
    for o in c["senales"]["oraciones"] + c["senales"]["fragmentos"]:
        for k in ("respaldo_por_fuente", "contradiccion_por_fuente", "similitud_por_fuente"):
            o[k] = o[k][:1]
        if o.get("llm"):
            o["llm"]["fuente"] = 0
    out["una_fuente"] = c
    c = copy.deepcopy(c0)
    o, f = c["senales"]["oraciones"][0], c["senales"]["fragmentos"][0]
    c["senales"]["oraciones"] = [dict(o) for _ in range(10)]
    c["senales"]["fragmentos"] = [dict(f, oracion=i) for i in range(10)]
    c["respuesta"] = " ".join([c["respuesta"]] * 10)
    out["diez_oraciones"] = c
    c = copy.deepcopy(c0)
    legal = ("No se podrá solicitar la ayuda si no se acredita la residencia, salvo que el artículo 12 "
             "disponga lo contrario; si no hay resolución en 3 meses, no se entenderá estimada.")
    c["fuentes"] = [legal, "El apartado 2 no se aplica si el solicitante no es titular."]
    c["respuesta"] = "Si no se acredita la residencia no se puede pedir la ayuda, salvo lo que diga el artículo 12."
    for o in c["senales"]["oraciones"] + c["senales"]["fragmentos"]:
        o["texto"] = c["respuesta"]
        for k in ("respaldo_por_fuente", "contradiccion_por_fuente", "similitud_por_fuente"):
            o[k] = [0.5, 0.2]
        if o.get("llm"):
            o["llm"]["fuente"] = 1
    out["legal_no_si"] = c
    out["vacio"] = {}
    out["senales_rotas"] = {"respuesta": "x", "fuentes": ["y"], "senales": {"oraciones": [{"texto": "x"}],
                                                                           "fragmentos": [None, 3]}}
    return out


def test_robustez_sinteticos(mod, train):
    g = mod.Guardian()
    for nombre, c in sinteticos(train).items():
        s = g.score(c)
        assert 0.0 <= s <= 1.0, nombre
    assert g.excepciones == 0, f"{g.excepciones} excepciones internas"


def test_null_es_sospechoso(mod, train):
    s = mod.Guardian().score(sinteticos(train)["veredicto_null"])
    assert s >= 0.5


# ------------------------------------------------------------------ rendimiento y regresión
def test_rendimiento(mod, test):
    g = mod.Guardian()
    t = time.perf_counter()
    for c in test:
        g.score(para_bot(c))
    ms = 1000 * (time.perf_counter() - t) / len(test)
    print(f"\n{ms:.2f} ms por caso")
    assert ms < 50


def test_regresion_cv():
    sys.path.insert(0, LAB)
    import validar_cv as V
    casos = leer(V.TRAIN)
    filas, _, _ = V.validar(V.Fernando(BOT), casos)
    media = V.resumen(filas)["puntuacion"][0]
    print(f"\nCV agrupada H: {media:+.3f}")
    assert media >= CV_MINIMO


def test_lexico_con_soporte_en_train(mod, train):
    """Cada patrón de las listas de palabras aparece en algún texto del train: nada de
    vocabulario que solo se haya podido sacar del test."""
    textos = [mod.norm(t) for c in train for t in c["fuentes"] + [c["respuesta"]]]
    for nombre in ("UNIVERSALES", "CAUSALES", "COMPARATIVAS", "MATIZ_MODAL", "MATIZ_ALCANCE"):
        for p in getattr(mod, nombre):
            assert any(p.search(t) for t in textos), f"{nombre}: {p.pattern} no aparece en el train"

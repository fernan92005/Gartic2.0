"""Tests de fernando_v2:  python -m pytest entregas/fernando_v2/lab/tests -q -p no:cacheprovider"""
import ast
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

AQUI = os.path.dirname(os.path.abspath(__file__))
BOT_DIR = os.path.abspath(os.path.join(AQUI, "..", ".."))
BOT = os.path.join(BOT_DIR, "mi_bot.py")
CSV = os.path.join(BOT_DIR, "predicciones.csv")
RAIZ = os.path.abspath(os.path.join(BOT_DIR, "..", ".."))
KIT = os.path.join(RAIZ, "kit")


def leer(nombre):
    return [json.loads(l) for l in open(os.path.join(KIT, nombre), encoding="utf-8") if l.strip()]


def para_bot(c):
    return {k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales") if k in c}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bot_fv2", BOT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def casos():
    return {"train": leer("train.jsonl"), "test": leer("test_publico.jsonl"), "giro": leer("giro_publico.jsonl")}


def test_csv_formato(casos):
    filas = list(csv.reader(open(CSV, encoding="utf-8")))
    assert filas[0] == ["id", "score"]
    ids = [f[0] for f in filas[1:]]
    assert len(ids) == len(set(ids)) == 660
    assert set(ids) == {c["id"] for c in casos["test"] + casos["giro"]}
    for cid, s in filas[1:]:
        assert re.fullmatch(r"\d+\.\d+", s) and 0.0 <= float(s) <= 1.0, (cid, s)
    assert ";" not in open(CSV, encoding="utf-8").read()


def test_csv_coincide_con_el_bot(mod, casos):
    g = mod.Guardian()
    esperado = {c["id"]: f"{float(g.score(para_bot(c))):.6f}" for c in casos["test"] + casos["giro"]}
    assert {f[0]: f[1] for f in list(csv.reader(open(CSV, encoding="utf-8")))[1:]} == esperado


def test_determinista(tmp_path):
    salidas = []
    for k in range(2):
        out = tmp_path / f"p{k}.csv"
        subprocess.run([sys.executable, "generar_predicciones.py", "--bot", BOT, "--test", "test_publico.jsonl",
                        "giro_publico.jsonl", "--salida", str(out)], cwd=KIT, check=True, capture_output=True)
        salidas.append(out.read_text(encoding="utf-8"))
    assert salidas[0] == salidas[1]


def test_cero_excepciones(mod, casos):
    """La v4 devuelve 1.0 en silencio si algo falla: comprobamos caracteristicas() y score() a mano."""
    fallos = 0
    for c in casos["train"] + casos["test"] + casos["giro"]:
        try:
            mod.caracteristicas(para_bot(c))
            s = mod.Guardian().score(para_bot(c))
            assert 0.0 <= s <= 1.0
        except Exception:
            fallos += 1
    assert fallos == 0


def test_imports_permitidos():
    arbol = ast.parse(open(BOT, encoding="utf-8").read())
    mods = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    permitidos = set(sys.stdlib_module_names) | {"numpy", "sklearn"}
    assert mods <= permitidos, mods - permitidos


def test_no_lee_ficheros_ni_red():
    src = open(BOT, encoding="utf-8").read()
    for prohibido in ("open(", "urllib", "socket", "subprocess", "train.jsonl", "requests"):
        assert prohibido not in src, prohibido


def test_sin_ids_ni_frases_del_test_ni_del_giro(casos):
    src = open(BOT, encoding="utf-8").read()
    assert not re.search(r"\b[tg]0\d{3}\b", src)
    texto_bot = " ".join(re.findall(r"\w+", src.lower()))

    def ngramas(lista):
        out = set()
        for c in lista:
            for t in [c["respuesta"], c["pregunta"]] + c["fuentes"]:
                w = re.findall(r"\w+", t.lower())
                out |= {" ".join(w[i:i + 6]) for i in range(len(w) - 5)}
        return out
    del_train = ngramas(casos["train"])
    copiados = {g for g in ngramas(casos["test"] + casos["giro"]) - del_train if g in texto_bot}
    assert not copiados, sorted(copiados)[:5]


def test_documentacion_sin_otros_equipos():
    src = open(BOT, encoding="utf-8").read().lower()
    assert "otros equipos" not in src


def test_c1_no_se_usa(mod):
    """C1 (corrección de estadísticos de orden) no entró en fernando_v2; su comprobación
    n <= n0 => sin cambios está en lab/variantes/C1/test_c1.py."""
    assert not hasattr(mod, "CDF_Q")

"""Validación cruzada agrupada (GroupKFold 5 x 3 semillas) SOLO sobre los casos H.

  python validar_cv.py                                  # todos los mi_bot.py de entregas/
  python validar_cv.py --bot ../../joseca_v1/mi_bot.py  # uno concreto
  python validar_cv.py --bot ../mi_bot.py@hgb           # variante de un adaptador

Grupo = documento = prefijo del id antes de "::" (o el id entero). Las particiones
reparten documentos de H; los casos no-H nunca se evalúan, pero los bots que
entrenan pueden usarlos siempre como entrenamiento (--sin-no-h para quitarlos).

Cómo se adapta cada bot (ADAPTADORES, abajo):
  - estatico: el bot se usa tal cual. Si trae pesos embebidos entrenados con TODO
    el train, su CV es OPTIMISTA (ha visto la partición evaluada) y se marca con *.
  - inyectar: para bots que leen train.jsonl en __init__: se copia el bot a una
    carpeta temporal con un train.jsonl que solo contiene los casos de
    entrenamiento de la partición y se instancia desde allí (cwd = esa carpeta).
  - joseca: reentrena por partición con las mismas funciones de su entrenar.py
    (caracteristicas -> estandarizar -> logística L2 por Newton, peso_h, l2) y
    mantiene su UMBRAL/reescalado tal cual.
  - candela: reentrena por partición un HistGradientBoosting de sklearn con su
    configuración embebida (7 hojas, 120 iteraciones, l2=5, peso no-H 0,35, sus
    mismas features) y conserva su regla sci_relation_risk y su reescalado.
  - fernando: reentrena con lab/entrenar.py (variantes: logreg, hgb).
"""
import argparse
import importlib.util
import json
import re
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))
TRAIN = os.path.join(RAIZ, "kit", "train.jsonl")
SEMILLAS = (0, 1, 2)


def cargar_modulo(ruta, nombre=None):
    nombre = nombre or "bot_" + str(abs(hash(os.path.abspath(ruta))))
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def caso_para_bot(c):
    return {k: c[k] for k in ("pregunta", "fuentes", "respuesta", "senales") if k in c}


def grupo(cid):
    return cid.split("::")[0]


def puntuacion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    n_neg = int((y == 0).sum())
    return 1.0 - (4 * fn + fp) / n_neg, fp, fn


def metricas(y, s):
    pred = (np.asarray(s) >= 0.5).astype(int)
    p, fp, fn = puntuacion(y, pred)
    auc = roc_auc_score(y, s) if len(set(y)) > 1 else 0.5
    return {"puntuacion": p, "auroc": auc, "fp": fp, "fn": fn, "marcados": 100 * pred.mean()}


def puntuar_seguro(bot, casos, contador):
    out = []
    for c in casos:
        try:
            out.append(min(1.0, max(0.0, float(bot.score(caso_para_bot(c))))))
        except Exception:
            contador[0] += 1
            out.append(1.0)
    return np.array(out)


# ---------------------------------------------------------------- adaptadores
class Estatico:
    optimista = False

    def __init__(self, ruta, variante=None):
        self.ruta = ruta
        src = open(ruta, encoding="utf-8").read()
        # Pesos embebidos = entrenados con todo el train (no podemos apartarlos)
        self.optimista = any(k in src for k in ("PESOS =", "_PARAMETERS", "MODELO_B64", "COEF"))

    def preparar(self, casos):
        self.excepciones = [0]
        bot = cargar_modulo(self.ruta).Guardian()
        t = time.perf_counter()
        self.scores = puntuar_seguro(bot, casos, self.excepciones)
        self.ms_caso = 1000 * (time.perf_counter() - t) / len(casos)

    def ajustar(self, idx):
        pass

    def predecir(self, idx):
        return self.scores[idx]


class Inyectar(Estatico):
    """Bot que entrena en __init__ leyendo train.jsonl: se le da solo la partición."""

    def preparar(self, casos):
        self.casos, self.excepciones, self.ms_caso = casos, [0], float("nan")

    def ajustar(self, idx):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "kit"))
        for destino in (self.tmp, os.path.join(self.tmp, "kit")):
            with open(os.path.join(destino, "train.jsonl"), "w", encoding="utf-8") as f:
                for i in idx:
                    f.write(json.dumps(self.casos[i], ensure_ascii=False) + "\n")
        copia = os.path.join(self.tmp, os.path.basename(self.ruta))
        shutil.copy(self.ruta, copia)
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.bot = cargar_modulo(copia).Guardian()
        finally:
            os.chdir(cwd)

    def predecir(self, idx):
        s = puntuar_seguro(self.bot, [self.casos[i] for i in idx], self.excepciones)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return s


class Joseca(Estatico):
    def __init__(self, ruta, variante=None):
        self.ruta = ruta
        self.mod = cargar_modulo(ruta)
        self.ent = cargar_modulo(os.path.join(os.path.dirname(ruta), "entrenar.py"))
        # Hiperparámetros que eligió su entrenar.py (verificado reproduciendo sus pesos
        # embebidos con todo el train): v1 peso_h=5 l2=10; v2 peso_h=5 l2=3;
        # v3 GradientBoosting(300 árboles, prof. 3, lr 0,03, subsample 0,8, hoja>=20), peso_h=5
        self.gb = hasattr(self.mod, "ARBOLES")
        defecto = "5,3" if "joseca_v2" in ruta.replace("\\", "/") else "5,10"
        self.peso_h, self.l2 = (float(v) for v in (variante or defecto).split(","))

    def preparar(self, casos):
        self.excepciones = [0]
        t = time.perf_counter()
        feats = [self.mod.caracteristicas(caso_para_bot(c)) for c in casos]
        self.ms_caso = 1000 * (time.perf_counter() - t) / len(casos)
        self.X = np.array([[f.get(n, 0.0) for n in self.mod.NOMBRES] for f in feats])
        self.y = np.array([c["etiqueta"] for c in casos], float)
        self.es_h = np.array([c["id"].startswith("H") for c in casos])

    def ajustar(self, idx):
        w = np.where(self.es_h[idx], self.peso_h, 1.0)
        if self.gb:
            from sklearn.ensemble import GradientBoostingClassifier
            self.modelo = GradientBoostingClassifier(
                n_estimators=len(self.mod.ARBOLES), max_depth=3, learning_rate=self.mod.TASA,
                subsample=0.8, min_samples_leaf=20, random_state=0).fit(self.X[idx], self.y[idx], sample_weight=w)
        else:
            self.modelo = self.ent.ajustar(self.X[idx], self.y[idx], w, self.l2)

    def predecir(self, idx):
        if self.gb:
            p = self.modelo.predict_proba(self.X[idx])[:, 1]
        else:
            p = self.ent.predecir(self.X[idx], self.modelo)
        return np.array([self.mod.reescalar(v, self.mod.UMBRAL) for v in p])


class Candela(Estatico):
    def __init__(self, ruta, variante=None):
        self.ruta = ruta
        self.mod = cargar_modulo(ruta)

    def preparar(self, casos):
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa
        P = self.mod._PARAMETERS
        self.excepciones = [0]
        t = time.perf_counter()
        feats = [self.mod.features_for(caso_para_bot(c), P["config"]["ablation"]) for c in casos]
        self.regla = np.array([self.mod.sci_relation_risk(caso_para_bot(c)) for c in casos])
        self.ms_caso = 1000 * (time.perf_counter() - t) / len(casos)
        self.X = np.array([[f[k] for k in P["features"]] for f in feats], float)
        self.y = np.array([c["etiqueta"] for c in casos])
        self.es_h = np.array([c["id"].startswith("H") for c in casos])

    def ajustar(self, idx):
        from sklearn.ensemble import HistGradientBoostingClassifier
        cfg = self.mod._PARAMETERS["config"]
        self.m = HistGradientBoostingClassifier(
            max_leaf_nodes=cfg["leaves"], max_iter=cfg["iterations"],
            l2_regularization=cfg["l2"], random_state=0)
        w = np.where(self.es_h[idx], 1.0, cfg["non_h_weight"])
        self.m.fit(self.X[idx], self.y[idx], sample_weight=w)

    def predecir(self, idx):
        raw = np.maximum(self.m.predict_proba(self.X[idx])[:, 1], self.regla[idx])
        t = self.mod._SCI_THRESHOLD
        return np.where(raw < t, 0.5 * raw / t, 0.5 + 0.5 * (raw - t) / (1 - t))


class Fernando(Estatico):
    def __init__(self, ruta, variante=None):
        self.ruta = ruta
        self.mod = cargar_modulo(ruta)
        self.variante = variante or (self.mod.MODELO or {}).get("variante", "logreg")
        self.ent = cargar_modulo(os.path.join(AQUI, "entrenar.py"))

    _cache = {}

    def preparar(self, casos):
        self.casos = casos
        clave = (self.ruta, len(casos), casos[0]["id"])
        if clave not in Fernando._cache:
            t = time.perf_counter()
            bot = self.mod.Guardian()
            feats = [bot.extraer(caso_para_bot(c)) for c in casos]
            Fernando._cache[clave] = (feats, 1000 * (time.perf_counter() - t) / len(casos), bot.excepciones)
        self.feats, self.ms_caso, exc = Fernando._cache[clave]
        self.excepciones = [exc]

    def ajustar(self, idx):
        self.modelo = self.ent.ajustar([self.casos[i] for i in idx],
                                       [self.feats[i] for i in idx], self.variante)

    def predecir(self, idx):
        bot = self.mod.Guardian(modelo=self.modelo)
        return np.array([bot.puntuar_feats(self.feats[i]) for i in idx])


def elegir_adaptador(ruta, variante):
    r = ruta.replace("\\", "/")
    if "joseca" in r:
        return Joseca(ruta, variante)
    if "candela" in r:
        return Candela(ruta, variante)
    if "fernando" in r:
        return Fernando(ruta, variante)
    src = open(ruta, encoding="utf-8").read()
    if "train.jsonl" in src and "open(" in src:
        return Inyectar(ruta)
    return Estatico(ruta)


# ---------------------------------------------------------------- validación
def validar(ad, casos, con_no_h=True, n_part=5):
    ad.preparar(casos)
    idx_h = np.array([i for i, c in enumerate(casos) if c["id"].startswith("H")])
    idx_no_h = np.array([i for i, c in enumerate(casos) if not c["id"].startswith("H")])
    y = np.array([c["etiqueta"] for c in casos])
    grupos_h = np.array([grupo(casos[i]["id"]) for i in idx_h])
    filas, oof = [], np.zeros((len(SEMILLAS), len(casos)))
    for k, semilla in enumerate(SEMILLAS):
        # GroupKFold sin shuffle reparte por tamaño de grupo e ignora el orden: con
        # shuffle=True (sklearn >= 1.6) cada semilla da un reparto distinto de documentos
        orden = np.arange(len(idx_h))
        gkf = GroupKFold(n_splits=n_part, shuffle=True, random_state=semilla)
        for tr, te in gkf.split(idx_h[orden], groups=grupos_h[orden]):
            tr_idx, te_idx = idx_h[orden][tr], idx_h[orden][te]
            assert not set(grupos_h[orden][tr]) & set(grupos_h[orden][te])
            entreno = np.concatenate([tr_idx, idx_no_h]) if con_no_h else tr_idx
            ad.ajustar(entreno)
            s = ad.predecir(te_idx)
            oof[k, te_idx] = s
            filas.append(metricas(y[te_idx], s))
    glob = [metricas(y[idx_h], oof[k, idx_h]) for k in range(len(SEMILLAS))]
    return filas, glob, {casos[i]["id"]: float(oof[:, i].mean()) for i in idx_h}


def resumen(filas):
    out = {}
    for k in ("puntuacion", "auroc", "fp", "fn", "marcados"):
        v = np.array([f[k] for f in filas], float)
        out[k] = (v.mean(), v.std())
    return out


def buscar_bots():
    base = os.path.join(RAIZ, "entregas")
    for d, _, fs in os.walk(base):
        if "mi_bot.py" in fs and "lab" not in d:
            yield os.path.join(d, "mi_bot.py")
    yield os.path.join(RAIZ, "kit", "ejemplo_bot.py")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", nargs="*", help="ruta[@variante]; por defecto todos")
    ap.add_argument("--datos", default=TRAIN)
    ap.add_argument("--sin-no-h", action="store_true")
    ap.add_argument("--detalle", action="store_true", help="una línea por partición")
    ap.add_argument("--guardar-oof", action="store_true",
                    help="guarda lab/resultados/oof_<nombre>.json (para estres_giro.py)")
    args = ap.parse_args(argv)
    casos = [json.loads(l) for l in open(args.datos, encoding="utf-8") if l.strip()]
    bots = args.bot or list(buscar_bots())
    print(f"{'bot':38s} {'puntuación':>15s} {'AUROC':>13s} {'FP':>11s} {'FN':>11s} "
          f"{'%marc':>11s} {'glob/semilla':>13s} {'ms':>6s} exc")
    resultados = {}
    for spec in bots:
        ruta, _, variante = spec.partition("@")
        ruta = os.path.abspath(ruta)
        ad = elegir_adaptador(ruta, variante or None)
        filas, glob, oof = validar(ad, casos, con_no_h=not args.sin_no_h)
        r = resumen(filas)
        nombre = os.path.relpath(ruta, RAIZ).replace("\\", "/").replace("/mi_bot.py", "")
        nombre += ("@" + variante if variante else "") + ("*" if ad.optimista else "")
        g = np.mean([x["puntuacion"] for x in glob])
        print(f"{nombre:38s} {r['puntuacion'][0]:+.3f} ± {r['puntuacion'][1]:.3f}  "
              f"{r['auroc'][0]:.3f} ± {r['auroc'][1]:.3f} {r['fp'][0]:5.1f} ± {r['fp'][1]:3.1f} "
              f"{r['fn'][0]:5.1f} ± {r['fn'][1]:3.1f} {r['marcados'][0]:5.1f} ± {r['marcados'][1]:3.1f} "
              f"{g:+13.3f} {ad.ms_caso:6.2f} {ad.excepciones[0]}")
        if args.detalle:
            for i, f in enumerate(filas):
                print(f"    semilla {SEMILLAS[i // 5]} part {i % 5}: {f['puntuacion']:+.3f} "
                      f"AUROC {f['auroc']:.3f} FP {f['fp']} FN {f['fn']} marcados {f['marcados']:.1f}%")
        resultados[nombre] = {"resumen": r, "global": glob}
        if args.guardar_oof:
            os.makedirs(os.path.join(AQUI, "resultados"), exist_ok=True)
            fich = re.sub(r"[^A-Za-z0-9_.-]", "_", nombre.replace("*", ""))
            with open(os.path.join(AQUI, "resultados", f"oof_{fich}.json"), "w") as f:
                json.dump(oof, f)
    print("* = pesos embebidos entrenados con todo el train: su CV es optimista (no se pudo apartar la partición).")
    return resultados


if __name__ == "__main__":
    main()

"""Entrena los pesos de mi_bot.py (de esta misma carpeta) y mide el bot de forma honesta.

  python entrenar.py                # valida, analiza errores y escribe los pesos en mi_bot.py
  python entrenar.py --no-escribir  # solo valida y analiza, sin tocar mi_bot.py
  python entrenar.py --ablacion     # cuánto aporta cada característica

Qué hace:
1. Calcula las características de todos los casos de train.jsonl (con las
   funciones de mi_bot.py, para que entrenamiento y bot usen exactamente lo mismo).
2. VALIDACIÓN CRUZADA POR FAMILIAS: los casos vienen en familias (H103::r1,
   H103::r2... comparten fuentes), así que una familia entera va siempre al
   mismo lado. Se entrena sin ella y se predice. Se repite con 3 repartos.
   Medimos sobre dos grupos:
     - H: los casos escritos como el test (la medida principal);
     - VARIAS ORACIONES (G, R, g con más de una oración): lo más parecido que
       tenemos al giro de las 15:30 («respuestas más largas»).
3. Elige el corte que maximiza la métrica oficial en H y lo guarda como UMBRAL
   (el bot lo reescala a 0,5).
4. Imprime un análisis de errores y escribe los pesos en mi_bot.py.

v2: valida todas las familias (no solo H), añade la medida «varias oraciones»
y busca train.jsonl también en kit/ (esta carpeta está en entregas/joseca_v2).

Solo el entrenamiento usa numpy; el bot entregado funciona con Python puro.
"""
import argparse
import collections
import importlib.util
import json
import os
import re

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
# train.jsonl puede estar aquí al lado o en kit/ (dos carpetas más arriba)
CANDIDATOS_TRAIN = [os.path.join(AQUI, "train.jsonl"),
                    os.path.join(AQUI, "..", "..", "kit", "train.jsonl")]

SEMILLAS = (0, 1, 2)  # repetimos la validación con 3 repartos distintos de familias

# Características que se calculan pero NO entran en el modelo, porque la
# ablación (python entrenar.py --ablacion) mostró que no aportan o empeoran.
EXCLUIR = [
    "palabras_nuevas",  # aprendía «más palabras nuevas = fiel» (las paráfrasis): atajo que no generaliza
    # v2, según la ablación (quitarlas mejora en H y en varias oraciones):
    # - versiones «por oración» del NLI, redundantes con las «por fragmento»
    "nli_resp_todas_min_orac", "nli_contr_todas_max_orac",
    # - características que se activan en poquísimos casos: el modelo les da
    #   pesos que dependen del azar de esos pocos casos
    "llm_sin_veredicto", "num_proporcion_falsa", "llmok_num_proporcion_falsa",
    "llmok_num_inexplicados", "llmok_matiz_perdido_tramo_aproximado",
]


def cargar_bot(ruta):
    spec = importlib.util.spec_from_file_location("mi_bot", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def puntuacion(y, pred):
    """Métrica oficial: 1 - (4·escapes + falsas alarmas) / nº de fieles."""
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    return 1.0 - (4 * fn + fp) / max(1, int(np.sum(y == 0))), fn, fp


def auroc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    mayor = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return mayor / (len(pos) * len(neg))


def mejor_corte(y, p):
    """Corte que maximiza la métrica, sobre una rejilla fija (más estable que
    probar cada score individual, que se ajusta demasiado a estos datos)."""
    mejor = (-1e9, 0.5)
    for t in np.arange(0.02, 0.80, 0.01):
        s, _, _ = puntuacion(y, (p >= t).astype(int))
        if s > mejor[0]:
            mejor = (s, float(t))
    return mejor


def entrenar_logistica(X, y, w_muestra, l2=1.0, iteraciones=50):
    """Regresión logística con regularización L2, resuelta con el método de
    Newton. X ya viene estandarizada. Devuelve (pesos, sesgo)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0  # el sesgo no se regulariza
    for _ in range(iteraciones):
        z = np.clip(Xb @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (w_muestra * (p - y)) + reg * beta
        W = w_muestra * p * (1 - p)
        H = (Xb * W[:, None]).T @ Xb + np.diag(reg)
        paso = np.linalg.solve(H, grad)
        beta -= paso
        if np.max(np.abs(paso)) < 1e-6:
            break
    return beta[:-1], beta[-1]


def ajustar(X, y, w, l2):
    media = X.mean(axis=0)
    escala = X.std(axis=0)
    escala[escala < 1e-9] = 1.0
    pesos, sesgo = entrenar_logistica((X - media) / escala, y, w, l2)
    return media, escala, pesos, sesgo


def predecir(X, modelo):
    media, escala, pesos, sesgo = modelo
    z = np.clip(((X - media) / escala) @ pesos + sesgo, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def validacion_por_familias(X, y, grupos, w, l2, n_pliegues=5, semilla=0):
    """Predicciones fuera de muestra para TODOS los casos, apartando familias enteras."""
    familias = sorted(set(grupos))
    rng = np.random.RandomState(semilla)
    rng.shuffle(familias)
    pliegue_de = {g: k % n_pliegues for k, g in enumerate(familias)}
    pliegue = np.array([pliegue_de[g] for g in grupos])
    oof = np.zeros(len(y))
    for k in range(n_pliegues):
        test = pliegue == k
        modelo = ajustar(X[~test], y[~test], w[~test], l2)
        oof[test] = predecir(X[test], modelo)
    return oof


class Validacion:
    """Resultado de validar una configuración: medidas en H y en varias oraciones."""

    def __init__(self, X, y, grupos, w, l2, es_h, es_multi):
        oofs = [validacion_por_familias(X, y, grupos, w, l2, semilla=s) for s in SEMILLAS]
        yh = y[es_h].astype(int)
        # Un único corte para todas las repeticiones (más estable), elegido en H
        _, self.corte = mejor_corte(np.concatenate([yh] * len(oofs)),
                                    np.concatenate([o[es_h] for o in oofs]))
        self.punt_h = float(np.mean([puntuacion(yh, (o[es_h] >= self.corte).astype(int))[0] for o in oofs]))
        self.auroc_h = float(np.mean([auroc(yh, o[es_h]) for o in oofs]))
        ym = y[es_multi].astype(int)
        self.punt_multi = float(np.mean([puntuacion(ym, (o[es_multi] >= self.corte).astype(int))[0] for o in oofs]))
        self.auroc_multi = float(np.mean([auroc(ym, o[es_multi]) for o in oofs]))
        self.oof = oofs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=os.path.join(AQUI, "mi_bot.py"))
    ap.add_argument("--datos", default=next((c for c in CANDIDATOS_TRAIN if os.path.exists(c)), CANDIDATOS_TRAIN[0]))
    ap.add_argument("--no-escribir", action="store_true")
    ap.add_argument("--ablacion", action="store_true",
                    help="mide cuánto aporta cada característica quitándolas de una en una")
    args = ap.parse_args()

    bot = cargar_bot(args.bot)
    casos = [json.loads(l) for l in open(args.datos, encoding="utf-8") if l.strip()]
    print(f"Calculando características de {len(casos)} casos ({os.path.normpath(args.datos)})...")
    feats = [bot.caracteristicas(c) for c in casos]
    nombres = [n for n in feats[0].keys() if n not in EXCLUIR]
    X = np.array([[f[n] for n in nombres] for f in feats], dtype=float)
    y = np.array([c["etiqueta"] for c in casos], dtype=float)
    ids = [c["id"] for c in casos]
    grupos = np.array([i.split("::")[0] for i in ids])
    tipos = [c.get("tipo") for c in casos]
    es_h = np.array([i.startswith("H") for i in ids])
    es_multi = np.array([i[0] in "GRg" and len(c["senales"]["oraciones"]) > 1
                         for i, c in zip(ids, casos)])
    yh = y[es_h].astype(int)
    print(f"Casos H: {es_h.sum()} · casos de varias oraciones (proxy del giro): {es_multi.sum()}")

    # --- Elegimos configuración (peso de los casos H y regularización) por validación en H ---
    print(f"\n=== Validación por familias ({len(SEMILLAS)} repeticiones, casos nunca vistos) ===")
    resultados = []
    for peso_h in (3.0, 5.0):
        for l2 in (3.0, 10.0, 30.0):
            w = np.where(es_h, peso_h, 1.0)
            v = Validacion(X, y, grupos, w, l2, es_h, es_multi)
            resultados.append((v.punt_h, peso_h, l2, v))
            print(f"peso H={peso_h:<3} L2={l2:<5}: H {v.punt_h:+.3f} (AUROC {v.auroc_h:.3f}) · "
                  f"varias oraciones {v.punt_multi:+.3f} (AUROC {v.auroc_multi:.3f}) · corte {v.corte:.2f}")
    _, peso_h, l2, val = max(resultados, key=lambda r: r[0])
    umbral = val.corte
    w = np.where(es_h, peso_h, 1.0)
    print(f"\nElegido: peso H={peso_h}, L2={l2}, UMBRAL={umbral:.2f} · H {val.punt_h:+.3f} · "
          f"varias oraciones {val.punt_multi:+.3f}")

    if args.ablacion:
        print("\n=== Ablación: qué pasa si quitamos cada característica ===")
        print("  (Δ positivo = el bot MEJORA sin ella -> candidata a EXCLUIR)")
        filas = []
        for k, n in enumerate(nombres):
            cols = [j for j in range(len(nombres)) if j != k]
            v = Validacion(X[:, cols], y, grupos, w, l2, es_h, es_multi)
            filas.append((v.punt_h - val.punt_h, v.punt_multi - val.punt_multi, n))
        for dh, dm, n in sorted(filas, reverse=True):
            print(f"  Δ H {dh:+.3f} · Δ varias oraciones {dm:+.3f} · {n}")
        return

    # --- Análisis de errores con las predicciones fuera de muestra (H) ---
    ph = val.oof[es_h]
    pred = (ph >= umbral).astype(int)
    idx_h = np.where(es_h)[0]
    print("\n=== ¿Qué se escapa? (alucinaciones H no detectadas, por tipo) ===")
    total, escapes = collections.Counter(), collections.Counter()
    for j, i in enumerate(idx_h):
        if yh[j] == 1:
            total[tipos[i]] += 1
            if pred[j] == 0:
                escapes[tipos[i]] += 1
    for t in sorted(total, key=lambda k: -total[k]):
        print(f"  {t:<20} se escapan {escapes[t]:>3} de {total[t]:>3}  ({100 * escapes[t] / total[t]:.0f} %)")
    print(f"  Falsas alarmas: {int(np.sum((yh == 0) & (pred == 1)))} de {int(np.sum(yh == 0))} fieles")

    def mostrar(j, i):
        c = casos[i]
        f = feats[i]
        activas = [f"{n}={f[n]:g}" for n in nombres
                   if f[n] and n.startswith(("num_", "matiz", "anadido", "llm_no", "llm_contr", "fecha", "entidad"))]
        print(f"  [{ph[j]:.2f}] {c['id']} ({c.get('tipo') or 'fiel'}): {c['respuesta'][:150]}")
        if activas:
            print(f"         señales: {', '.join(activas)}")

    print("\n=== Escapes más «seguros» (el bot dice fiel con más convicción) ===")
    for j in [j for j in np.argsort(ph) if yh[j] == 1 and pred[j] == 0][:10]:
        mostrar(j, idx_h[j])
    print("\n=== Falsas alarmas más «seguras» ===")
    for j in [j for j in np.argsort(-ph) if yh[j] == 0 and pred[j] == 1][:10]:
        mostrar(j, idx_h[j])

    # --- Modelo final con todo el train ---
    media, escala, pesos, sesgo = ajustar(X, y, w, l2)
    print("\n=== Pesos del modelo final (sobre características estandarizadas) ===")
    for n, p in sorted(zip(nombres, pesos), key=lambda t: -abs(t[1])):
        print(f"  {p:+.3f}  {n}")

    if args.no_escribir:
        return
    bloque = (
        "# >>> PESOS (generado automáticamente por entrenar.py; no editar a mano)\n"
        f"NOMBRES = {json.dumps(nombres, ensure_ascii=False)}\n"
        f"MEDIAS = {json.dumps([round(float(v), 6) for v in media])}\n"
        f"ESCALAS = {json.dumps([round(float(v), 6) for v in escala])}\n"
        f"PESOS = {json.dumps([round(float(v), 6) for v in pesos])}\n"
        f"SESGO = {float(sesgo):.6f}\n"
        f"UMBRAL = {umbral:.2f}\n"
        "# <<< PESOS"
    )
    codigo = open(args.bot, encoding="utf-8").read()
    codigo = re.sub(r"# >>> PESOS.*?# <<< PESOS", lambda _: bloque, codigo, flags=re.S)
    with open(args.bot, "w", encoding="utf-8", newline="") as fh:
        fh.write(codigo)
    print(f"\nPesos escritos en {args.bot}")


if __name__ == "__main__":
    main()

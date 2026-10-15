"""Entrena los pesos de mi_bot.py y mide el bot de forma honesta.

  python entrenar.py              # valida, analiza errores y escribe los pesos en mi_bot.py
  python entrenar.py --no-escribir  # solo valida y analiza, sin tocar mi_bot.py

Qué hace:
1. Calcula las características de todos los casos de train.jsonl (con las
   funciones de mi_bot.py, para que entrenamiento y bot usen exactamente lo mismo).
2. VALIDACIÓN CRUZADA POR GRUPOS sobre los casos H (los parecidos al test):
   los casos H vienen en familias (H103::r1, H103::r2... comparten fuentes),
   así que una familia entera va siempre al mismo lado. En cada vuelta se
   entrena con todo lo demás (resto del train + otras familias H) y se predice
   la familia apartada. Así medimos casos que el modelo no ha visto nunca.
3. Elige el corte que maximiza la métrica oficial en esas predicciones
   «fuera de muestra» y lo guarda como UMBRAL (el bot lo reescala a 0,5).
4. Imprime un análisis de errores: qué tipos de alucinación se escapan, qué
   falsas alarmas da y qué características pesan más.
5. Entrena el modelo final con todo el train y escribe los pesos en mi_bot.py.

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
    """Corte que maximiza la métrica. Probamos una rejilla fija (más estable que
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


def estandarizar(X):
    media = X.mean(axis=0)
    escala = X.std(axis=0)
    escala[escala < 1e-9] = 1.0
    return media, escala


def ajustar(X, y, w, l2):
    media, escala = estandarizar(X)
    pesos, sesgo = entrenar_logistica((X - media) / escala, y, w, l2)
    return media, escala, pesos, sesgo


def predecir(X, modelo):
    media, escala, pesos, sesgo = modelo
    z = np.clip(((X - media) / escala) @ pesos + sesgo, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def validacion_por_grupos(X, y, ids, es_h, peso_h, l2, n_pliegues=5, semilla=0):
    """Predicciones fuera de muestra para los casos H, apartando familias enteras."""
    grupos = np.array([i.split("::")[0] for i in ids])
    familias = sorted(set(grupos[es_h]))
    rng = np.random.RandomState(semilla)
    rng.shuffle(familias)
    pliegue_de = {g: k % n_pliegues for k, g in enumerate(familias)}
    oof = np.full(len(y), np.nan)
    for k in range(n_pliegues):
        test = es_h & np.array([pliegue_de.get(g, -1) == k for g in grupos])
        train = ~test
        w = np.where(es_h, peso_h, 1.0)[train]
        modelo = ajustar(X[train], y[train], w, l2)
        oof[test] = predecir(X[test], modelo)
    return oof


SEMILLAS = (0, 1, 2)  # repetimos la validación con 3 repartos distintos de familias

# Características que se calculan pero NO entran en el modelo, porque la
# ablación (python entrenar.py --ablacion) mostró que quitarlas no empeora o mejora.
EXCLUIR = [
    "palabras_nuevas",  # aprendía «más palabras nuevas = fiel» (las paráfrasis): atajo que no generaliza
    # Atribución cruzada: salta igual en fieles (19) que en alucinaciones (17)
    # en H, así que tal como está no distingue nada. Pendiente de rehacer.
    "num_entidad_ajena", "llmok_num_entidad_ajena",
]


def evaluar(X, y, ids, es_h, peso_h, l2):
    """Valida con varias semillas. Devuelve (AUROC medio, puntuación media con el
    mejor corte común, corte, predicciones fuera de muestra de la 1ª semilla)."""
    yh = y[es_h].astype(int)
    oofs = [validacion_por_grupos(X, y, ids, es_h, peso_h, l2, semilla=s)[es_h] for s in SEMILLAS]
    # Un único corte para todas las repeticiones (más estable)
    _, corte = mejor_corte(np.concatenate([yh] * len(oofs)), np.concatenate(oofs))
    puntos = [puntuacion(yh, (o >= corte).astype(int))[0] for o in oofs]
    return float(np.mean([auroc(yh, o) for o in oofs])), float(np.mean(puntos)), corte, oofs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=os.path.join(AQUI, "mi_bot.py"))
    ap.add_argument("--datos", default=os.path.join(AQUI, "train.jsonl"))
    ap.add_argument("--no-escribir", action="store_true")
    ap.add_argument("--ablacion", action="store_true",
                    help="mide cuánto aporta cada característica quitándolas de una en una")
    args = ap.parse_args()

    bot = cargar_bot(args.bot)
    casos = [json.loads(l) for l in open(args.datos, encoding="utf-8") if l.strip()]
    print(f"Calculando características de {len(casos)} casos...")
    feats = [bot.caracteristicas(c) for c in casos]
    nombres = [n for n in feats[0].keys() if n not in EXCLUIR]
    X = np.array([[f[n] for n in nombres] for f in feats], dtype=float)
    y = np.array([c["etiqueta"] for c in casos], dtype=float)
    ids = [c["id"] for c in casos]
    tipos = [c.get("tipo") for c in casos]
    es_h = np.array([i.startswith("H") for i in ids])
    yh = y[es_h].astype(int)

    # --- Referencias para comparar ---
    print("\n=== Referencias sobre los 631 casos H ===")
    llm_malo = ((X[:, nombres.index("llm_no_respaldada")] + X[:, nombres.index("llm_contradicha")]) > 0)
    s, fn, fp = puntuacion(yh, llm_malo[es_h].astype(int))
    print(f"Fiarse del juez LLM        : puntuación {s:+.3f} · {fn} escapes · {fp} falsas alarmas")

    # --- Elegimos configuración (peso de los casos H y regularización) por validación ---
    print(f"\n=== Validación cruzada por familias sobre H (casos nunca vistos, {len(SEMILLAS)} repeticiones) ===")
    resultados = []
    for peso_h in (1.0, 3.0, 5.0):
        for l2 in (3.0, 10.0, 30.0):
            a, s_opt, t_opt, ph = evaluar(X, y, ids, es_h, peso_h, l2)
            resultados.append((s_opt, peso_h, l2, t_opt, ph))
            print(f"peso H={peso_h:<3} L2={l2:<5}: AUROC {a:.3f} · con corte {t_opt:.2f}: {s_opt:+.3f}")
    s_opt, peso_h, l2, umbral, ph = max(resultados, key=lambda r: r[0])
    print(f"\nElegido: peso H={peso_h}, L2={l2}, UMBRAL={umbral:.2f} (puntuación validada {s_opt:+.3f})")
    print("Ojo: el corte se elige sobre estas mismas predicciones, así que la cifra real")
    print("en el test será algo más baja. Es la mejor estimación que tenemos.")

    if args.ablacion:
        print("\n=== Ablación: qué pasa si quitamos cada característica ===")
        print("  (Δ positivo = el bot MEJORA sin ella -> candidata a EXCLUIR)")
        filas = []
        for k, n in enumerate(nombres):
            cols = [j for j in range(len(nombres)) if j != k]
            a_k, s_k, _, _ = evaluar(X[:, cols], y, ids, es_h, peso_h, l2)
            filas.append((s_k - s_opt, a_k, n))
        for d, a_k, n in sorted(filas, reverse=True):
            print(f"  Δ puntuación {d:+.3f} · AUROC sin ella {a_k:.3f} · {n}")
        return

    # --- Análisis de errores con las predicciones fuera de muestra ---
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
    nfp = int(np.sum((yh == 0) & (pred == 1)))
    print(f"  Falsas alarmas: {nfp} de {int(np.sum(yh == 0))} fieles")

    def mostrar(j, i):
        c = casos[i]
        f = feats[i]
        activas = [f"{n}={f[n]:g}" for n in nombres
                   if f[n] and (n.startswith(("num_", "matiz", "anadido", "llm_no", "llm_contr")))]
        print(f"  [{ph[j]:.2f}] {c['id']} ({c.get('tipo') or 'fiel'}): {c['respuesta'][:150]}")
        if activas:
            print(f"         señales: {', '.join(activas)}")

    print("\n=== Escapes más «seguros» (el bot dice fiel con más convicción) ===")
    orden = [j for j in np.argsort(ph) if yh[j] == 1 and pred[j] == 0][:10]
    for j in orden:
        mostrar(j, idx_h[j])
    print("\n=== Falsas alarmas más «seguras» ===")
    orden = [j for j in np.argsort(-ph) if yh[j] == 0 and pred[j] == 1][:8]
    for j in orden:
        mostrar(j, idx_h[j])

    # --- Modelo final con todo el train ---
    w = np.where(es_h, peso_h, 1.0)
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
    with open(args.bot, "w", encoding="utf-8") as fh:
        fh.write(codigo)
    print(f"\nPesos escritos en {args.bot}")


if __name__ == "__main__":
    main()

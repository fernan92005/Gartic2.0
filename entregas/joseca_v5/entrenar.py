"""Entrena el modelo de mi_bot.py (de esta misma carpeta) y mide el bot de forma honesta.

  python entrenar.py                # valida, analiza errores y escribe los árboles en mi_bot.py
  python entrenar.py --no-escribir  # solo valida y analiza, sin tocar mi_bot.py

Necesita numpy y scikit-learn (pip install scikit-learn). El bot que se entrega
NO los necesita: los árboles se escriben dentro de mi_bot.py como datos.

Qué hace:
1. Calcula las características de todos los casos de train.jsonl con las
   funciones de mi_bot.py (entrenamiento y bot usan exactamente lo mismo).
2. VALIDACIÓN POR FAMILIAS: los casos vienen en familias (H103::r1, H103::r2...
   comparten fuentes), así que una familia entera va siempre al mismo lado; se
   entrena sin ella y se predice. Se repite con 3 repartos. Se mide en:
     - H: los casos escritos como el test (la medida principal);
     - VARIAS ORACIONES (G, R, g con más de una oración): lo más parecido que
       tenemos al giro de las 15:30 («respuestas más largas»).
3. Compara la regresión logística (v2) con gradient boosting (v3) y elige la
   configuración de gradient boosting con mejor puntuación en H.
4. Elige el corte (UMBRAL) que maximiza la métrica oficial en H; el bot lo
   reescala a 0,5.
5. Entrena el modelo final con todo el train y escribe los árboles en mi_bot.py.
   Comprueba que el bot en Python puro da lo mismo que scikit-learn.
"""
import argparse
import collections
import importlib.util
import json
import os
import re

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

AQUI = os.path.dirname(os.path.abspath(__file__))
# train.jsonl puede estar aquí al lado o en kit/ (dos carpetas más arriba)
CANDIDATOS_TRAIN = [os.path.join(AQUI, "train.jsonl"),
                    os.path.join(AQUI, "..", "..", "kit", "train.jsonl")]

SEMILLAS = (0, 1, 2)  # repetimos la validación con 3 repartos distintos de familias
PESO_H = 5.0          # los casos H (parecidos al test) pesan más al entrenar

# Características que se calculan pero NO entran en el modelo (ver ablación de v2)
EXCLUIR = [
    # Tiempos verbales: «previsto -> hecho» metía ruido (H -0,012, pseudo-giro
    # -0,014); se queda solo «plan -> seguro» (34 alucinaciones y 1 fiel en H)
    "tiempo_previsto_a_hecho", "llmok_tiempo_previsto_a_hecho",
    "palabras_nuevas",  # aprendía «más palabras nuevas = fiel» (las paráfrasis): atajo que no generaliza
    "nli_resp_todas_min_orac", "nli_contr_todas_max_orac",  # redundantes con las «por fragmento»
    "llm_sin_veredicto", "num_proporcion_falsa", "llmok_num_proporcion_falsa",  # casi nunca se activan
    "llmok_num_inexplicados", "llmok_matiz_perdido_tramo_aproximado",
    # v4: la cobertura (qué parte de la fuente citada / de la respuesta queda
    # fuera) empeora la validación en H (-0,015) y en varias oraciones: fuera
    "cobertura_citada_fuera", "resp_fuera_de_usadas",
    # v5, según la ablación por grupos:
    # - «todo cuadra» empeoraba (H -0,029, varias oraciones -0,041): los árboles
    #   ya combinan solos esas señales y la versión explícita sobreajustaba
    "todo_cuadra", "todo_cuadra_con_cuenta_ok",
    # - neutras (±0,003): fuera para no complicar el modelo
    "sujeto_generalizado", "llmok_sujeto_generalizado",
    "matiz_oracion_perdido_v5", "llmok_matiz_oracion_perdido_v5",
    # v6: de las características de Candela usamos B1 + B2 + B4; las
    # proporciones de B1 no las probamos en el banco de pruebas
    "cb_matiz_clausula_ratio_restrictivo", "cb_matiz_clausula_ratio_condicional",
    "cb_matiz_clausula_ratio_aproximado", "cb_matiz_clausula_ratio_parcial",
    # v6: idea de Fernando («todas las cifras nuevas explicadas»): subía el
    # AUROC pero bajaba la puntuación en la validación honesta
    "cifras_nuevas_todas_explicadas", "llmok_cifras_nuevas_todas_explicadas",
]

# Configuraciones de gradient boosting que probamos (todas con árboles pequeños
# y submuestreo, para no memorizar el train)
CONFIGS_GB = [
    dict(n_estimators=200, max_depth=3, learning_rate=0.05),
    dict(n_estimators=300, max_depth=3, learning_rate=0.03),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05),
]
COMUN_GB = dict(subsample=0.8, min_samples_leaf=20, random_state=0)


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


# --- Modelos: cada «fábrica» entrena con (X, y, pesos) y devuelve una función que predice ---

def pseudo_giro(casos, semilla=11):
    """Respuestas de 3-4 oraciones (como el giro) hechas juntando casos H de una
    oración de la MISMA familia (mismas fuentes): todas fieles, o con UNA
    alucinación dentro. Así tenemos casos con forma de giro y con etiqueta."""
    import random
    rng = random.Random(semilla)
    por_familia = collections.defaultdict(list)
    for c in casos:
        if c["id"].startswith("H") and len(c["senales"]["oraciones"]) == 1:
            por_familia[c["id"].split("::")[0]].append(c)
    salida = []
    for g in sorted(por_familia):
        cs = [c for c in por_familia[g] if c["fuentes"] == por_familia[g][0]["fuentes"]]
        fieles = [c for c in cs if c["etiqueta"] == 0]
        malas = [c for c in cs if c["etiqueta"] == 1]
        for _ in range(3):
            n = 3 if rng.random() < 0.7 else 4
            if len(fieles) >= n:
                salida.append(_juntar(rng.sample(fieles, n)))
            if len(fieles) >= n - 1 and malas:
                grupo = rng.sample(fieles, n - 1) + [rng.choice(malas)]
                rng.shuffle(grupo)
                salida.append(_juntar(grupo))
    return salida


def _juntar(grupo):
    senales = {"oraciones": [], "fragmentos": []}
    desplaz = 0
    for c in grupo:
        senales["oraciones"] += c["senales"]["oraciones"]
        for fr in c["senales"]["fragmentos"]:
            senales["fragmentos"].append(dict(fr, oracion=fr.get("oracion", 0) + desplaz))
        desplaz += len(c["senales"]["oraciones"])
    mala = [c for c in grupo if c["etiqueta"] == 1]
    return {"id": grupo[0]["id"].split("::")[0] + "::pseudo", "pregunta": grupo[0]["pregunta"],
            "fuentes": grupo[0]["fuentes"], "senales": senales,
            "respuesta": " ".join(c["respuesta"].rstrip(".") + "." for c in grupo),
            "etiqueta": int(bool(mala)), "tipo": mala[0]["tipo"] if mala else None}


def logistica(X, y, w, l2=3.0, iteraciones=50):
    """Regresión logística con L2 (método de Newton). Solo como referencia (era v2)."""
    media, escala = X.mean(axis=0), X.std(axis=0)
    escala[escala < 1e-9] = 1.0
    Xb = np.hstack([(X - media) / escala, np.ones((len(X), 1))])
    beta = np.zeros(Xb.shape[1])
    reg = np.full(Xb.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iteraciones):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        H = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + np.diag(reg)
        paso = np.linalg.solve(H, Xb.T @ (w * (p - y)) + reg * beta)
        beta -= paso
        if np.max(np.abs(paso)) < 1e-6:
            break

    def predecir(Xn):
        Xnb = np.hstack([(Xn - media) / escala, np.ones((len(Xn), 1))])
        return 1.0 / (1.0 + np.exp(-np.clip(Xnb @ beta, -30, 30)))
    return predecir


def fabrica_gb(config):
    def entrenar(X, y, w):
        m = GradientBoostingClassifier(**config, **COMUN_GB)
        m.fit(X, y, sample_weight=w)
        return lambda Xn: m.predict_proba(Xn)[:, 1]
    return entrenar


def validacion_por_familias(fabrica, X, y, grupos, w, n_pliegues=5, semilla=0):
    """Predicciones fuera de muestra para TODOS los casos, apartando familias enteras."""
    familias = sorted(set(grupos))
    rng = np.random.RandomState(semilla)
    rng.shuffle(familias)
    pliegue_de = {g: k % n_pliegues for k, g in enumerate(familias)}
    pliegue = np.array([pliegue_de[g] for g in grupos])
    oof = np.zeros(len(y))
    for k in range(n_pliegues):
        test = pliegue == k
        oof[test] = fabrica(X[~test], y[~test], w[~test])(X[test])
    return oof


class Validacion:
    """Resultado de validar un modelo: medidas en H y en varias oraciones."""

    def __init__(self, fabrica, X, y, grupos, w, es_h, es_multi, regla=None):
        oofs = [validacion_por_familias(fabrica, X, y, grupos, w, semilla=s) for s in SEMILLAS]
        if regla is not None:  # v5: las reglas del bot se aplican igual que en mi_bot.py
            oofs = [regla(o) for o in oofs]
        yh, ym = y[es_h].astype(int), y[es_multi].astype(int)
        # Un único corte para todas las repeticiones (más estable), elegido en H
        _, self.corte = mejor_corte(np.concatenate([yh] * len(oofs)),
                                    np.concatenate([o[es_h] for o in oofs]))
        self.punt_h = float(np.mean([puntuacion(yh, (o[es_h] >= self.corte).astype(int))[0] for o in oofs]))
        self.auroc_h = float(np.mean([auroc(yh, o[es_h]) for o in oofs]))
        self.punt_multi = float(np.mean([puntuacion(ym, (o[es_multi] >= self.corte).astype(int))[0] for o in oofs]))
        self.auroc_multi = float(np.mean([auroc(ym, o[es_multi]) for o in oofs]))
        self.oof = oofs[0]

    def linea(self):
        return (f"H {self.punt_h:+.3f} (AUROC {self.auroc_h:.3f}) · varias oraciones "
                f"{self.punt_multi:+.3f} (AUROC {self.auroc_multi:.3f}) · corte {self.corte:.2f}")


def exportar_arboles(modelo):
    """Convierte los árboles de scikit-learn en listas de Python para mi_bot.py."""
    arboles = []
    for est in modelo.estimators_[:, 0]:
        t = est.tree_
        arboles.append([
            [int(v) for v in t.feature],
            [float(v) for v in t.threshold],  # sin redondear: el bot debe dar lo mismo que sklearn
            [int(v) for v in t.children_left],
            [int(v) for v in t.children_right],
            [round(float(v), 6) for v in t.value[:, 0, 0]],
        ])
    return arboles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=os.path.join(AQUI, "mi_bot.py"))
    ap.add_argument("--datos", default=next((c for c in CANDIDATOS_TRAIN if os.path.exists(c)), CANDIDATOS_TRAIN[0]))
    ap.add_argument("--no-escribir", action="store_true")
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
    w = np.where(es_h, PESO_H, 1.0)
    yh = y[es_h].astype(int)
    print(f"Casos H: {es_h.sum()} · casos de varias oraciones (proxy del giro): {es_multi.sum()}")

    # v5: las reglas fijas del bot (aplicar_reglas) se aplican también aquí
    def regla(oof):
        return np.array([bot.aplicar_reglas(p, f) for p, f in zip(oof, feats)])

    print(f"\n=== Validación por familias ({len(SEMILLAS)} repeticiones, casos nunca vistos) ===")
    print(f"Regresión logística (v2)          : {Validacion(logistica, X, y, grupos, w, es_h, es_multi).linea()}")
    resultados = []
    for config in CONFIGS_GB:
        v = Validacion(fabrica_gb(config), X, y, grupos, w, es_h, es_multi, regla=regla)
        resultados.append((v.punt_h, config, v))
        print(f"GB {config['n_estimators']} árboles prof={config['max_depth']} lr={config['learning_rate']:<5}: {v.linea()}")
    _, config, val = max(resultados, key=lambda r: r[0])
    umbral = val.corte
    sin_regla = Validacion(fabrica_gb(config), X, y, grupos, w, es_h, es_multi)
    print(f"  (la misma configuración SIN la regla de cuentas: {sin_regla.linea()})")
    print(f"\nElegido: gradient boosting {config} · UMBRAL={umbral:.2f}")

    # --- Análisis de errores con las predicciones fuera de muestra (H) ---
    ph = val.oof[es_h]
    pred = (ph >= umbral).astype(int)
    idx_h = np.where(es_h)[0]
    print("\n=== ¿Qué se escapa? (alucinaciones H no detectadas, por tipo) ===")
    total, escapes = collections.Counter(), collections.Counter()
    for j, i in enumerate(idx_h):
        if yh[j] == 1:
            total[tipos[i]] += 1
            escapes[tipos[i]] += int(pred[j] == 0)
    for t in sorted(total, key=lambda k: -total[k]):
        print(f"  {t:<20} se escapan {escapes[t]:>3} de {total[t]:>3}  ({100 * escapes[t] / total[t]:.0f} %)")
    print(f"  Falsas alarmas: {int(np.sum((yh == 0) & (pred == 1)))} de {int(np.sum(yh == 0))} fieles")

    def mostrar(j, i):
        c, f = casos[i], feats[i]
        activas = [f"{n}={f[n]:g}" for n in nombres
                   if f[n] and n.startswith(("num_", "matiz", "anadido", "llm_no", "llm_contr", "fecha", "entidad"))]
        print(f"  [{ph[j]:.2f}] {c['id']} ({c.get('tipo') or 'fiel'}): {c['respuesta'][:150]}")
        if activas:
            print(f"         señales: {', '.join(activas)}")

    print("\n=== Escapes más «seguros» (el bot dice fiel con más convicción) ===")
    for j in [j for j in np.argsort(ph) if yh[j] == 1 and pred[j] == 0][:8]:
        mostrar(j, idx_h[j])
    print("\n=== Falsas alarmas más «seguras» ===")
    for j in [j for j in np.argsort(-ph) if yh[j] == 0 and pred[j] == 1][:8]:
        mostrar(j, idx_h[j])

    # --- Corte para respuestas LARGAS (el giro: 3-4 oraciones) ---
    # El corte de arriba se elige con casos H de una oración. Una respuesta
    # larga tiene 3-4 ocasiones de parecer sospechosa, así que necesita su
    # propio corte. Lo elegimos con el PSEUDO-GIRO: respuestas de 3-4 oraciones
    # hechas juntando casos H de una misma familia (todas fieles, o con una
    # alucinación). Cada una la puntúa un modelo que no vio esa familia, con la
    # misma regla que el bot (la peor de sus oraciones).
    pseudo = pseudo_giro(casos)
    fam_p = np.array([c["id"].split("::")[0] for c in pseudo])
    y_p = np.array([c["etiqueta"] for c in pseudo])
    sub_p = [[bot.caracteristicas(sc) for sc in bot.por_oracion(c)] for c in pseudo]
    oofs_p = []
    for s in SEMILLAS:
        familias = sorted(set(grupos)); rng = np.random.RandomState(s); rng.shuffle(familias)
        pl = {g: k % 5 for k, g in enumerate(familias)}
        pliegue = np.array([pl[g] for g in grupos]); pliegue_p = np.array([pl[g] for g in fam_p])
        oof_p = np.zeros(len(y_p))
        for k in range(5):
            t = pliegue == k
            pred = fabrica_gb(config)(X[~t], y[~t], w[~t])
            for i in np.where(pliegue_p == k)[0]:
                ps = pred(np.array([[f[n] for n in nombres] for f in sub_p[i]], dtype=float))
                oof_p[i] = max(bot.aplicar_reglas(a, f) for a, f in zip(ps, sub_p[i]))
        oofs_p.append(oof_p)
    _, umbral_largas = mejor_corte(np.concatenate([y_p] * len(oofs_p)), np.concatenate(oofs_p))
    punt_p = np.mean([puntuacion(y_p, (o >= umbral_largas).astype(int))[0] for o in oofs_p])
    print(f"\nPseudo-giro ({len(pseudo)} respuestas de 3-4 oraciones): UMBRAL_LARGAS={umbral_largas:.2f} · "
          f"puntuación {punt_p:+.3f} · AUROC {np.mean([auroc(y_p, o) for o in oofs_p]):.3f}")

    # --- Modelo final con todo el train ---
    modelo = GradientBoostingClassifier(**config, **COMUN_GB)
    modelo.fit(X, y, sample_weight=w)
    print("\n=== Características más usadas por los árboles (importancia) ===")
    for n, imp in sorted(zip(nombres, modelo.feature_importances_), key=lambda t: -t[1])[:15]:
        print(f"  {imp:.3f}  {n}")

    # Valor inicial del modelo (antes de sumar árboles): lo sacamos de sklearn
    tasa = float(modelo.learning_rate)
    arboles = exportar_arboles(modelo)
    suma_arboles = sum(est.predict(X[:1])[0] for est in modelo.estimators_[:, 0])
    inicial = float(modelo.decision_function(X[:1])[0] - tasa * suma_arboles)

    if args.no_escribir:
        return
    bloque = (
        "# >>> MODELO (generado automáticamente por entrenar.py; no editar a mano)\n"
        f"NOMBRES = {json.dumps(nombres, ensure_ascii=False)}\n"
        f"INICIAL = {inicial:.6f}\n"
        f"TASA = {tasa}\n"
        f"ARBOLES = {json.dumps(arboles, separators=(',', ':'))}\n"
        f"UMBRAL = {umbral:.2f}\n"
        f"UMBRAL_LARGAS = {umbral_largas:.2f}\n"
        "# <<< MODELO"
    )
    codigo = open(args.bot, encoding="utf-8").read()
    codigo = re.sub(r"# >>> MODELO.*?# <<< MODELO", lambda _: bloque, codigo, flags=re.S)
    with open(args.bot, "w", encoding="utf-8", newline="") as fh:
        fh.write(codigo)
    print(f"\nÁrboles escritos en {args.bot} ({len(arboles)} árboles)")

    # Comprobación: el bot en Python puro tiene que dar lo mismo que scikit-learn
    bot = cargar_bot(args.bot)
    p_bot = np.array([bot.probabilidad(f) for f in feats])
    p_sk = modelo.predict_proba(X)[:, 1]
    print(f"Diferencia máxima bot (Python puro) vs scikit-learn: {np.max(np.abs(p_bot - p_sk)):.2e}")


if __name__ == "__main__":
    main()

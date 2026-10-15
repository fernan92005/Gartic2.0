"""El Guardián — bot de joseca (versión 8).

v8 = v4 (la mejor en el leaderboard: 0,497 con test + giro) con un único
cambio, en el ENTRENAMIENTO: las respuestas de varias oraciones del train
(las más parecidas al giro) pesan como los casos H. Las características y la
forma de puntuar son las de v4. En el banco de pruebas: H honesto +0,509 ->
+0,536 y respuestas de varias oraciones +0,714 -> +0,735.

--- Notas de la versión 4 ---


v4 (propuesta A): verificador de cuentas DENTRO de la frase
(verificar_cuentas_frase): recalcula la operación concreta que hace la
respuesta con sus propias cifras («de A a B, un X %», «A por B ... fue de X»,
«A y B sumaron X», «X más que [otra entidad]», «perdió A, un X % de su
población») en vez de aceptar cualquier combinación casual de cifras.
Además: «el 30 % de 18.400» entre fuentes distintas, y comparación en float32
para que el bot dé exactamente lo mismo que scikit-learn.
En la validación por familias: H +0,42 -> +0,51, AUROC 0,907 -> 0,931.
Después, dos ideas generales (programadas por nosotros, tras ver qué
señales aprovechaban otros equipos): «matiz perdido por oración» (la oración
no conserva ninguna cautela y su fuente sí) y el reparto del respaldo entre
fragmentos (segundo peor, fracción de fragmentos flojos). La cobertura se
probó y se descartó (empeoraba). Resultado: H +0,55, AUROC 0,941.

--- Notas de la versión 3 ---

v3: mismas características que v2, pero el modelo pasa de regresión logística
a gradient boosting (árboles de decisión), embebido en este fichero como datos.
En la validación por familias: H +0,36 -> +0,42, AUROC 0,877 -> 0,907.

--- Notas de la versión 2 ---

Cambios respecto a joseca_v1 (los comentarios con «v2» marcan cada uno):
  - cuentas anunciadas («9 % más», «X menos que»): si la cifra está tal cual
    en la fuente, es un dato y no una cuenta; y las diferencias sin otra cifra
    en la respuesta solo se buscan en fuentes que nombran a alguien de ella;
  - atribución cruzada rehecha: entidad principal frente a secundarias, y solo
    cuentan las fuentes que hablan de OTRA entidad comparable («la vecina...»);
  - nombres propios al principio de frase («Lusel retiene...»);
  - plazos («nueve meses después») y horas («10:05») comprobados;
  - recuentos limitados a 2, para que las respuestas largas (el giro) no
    disparen el score solo por tener más oraciones;
  - entrenar.py excluye características redundantes o rarísimas (ablación).

Solo usa la librería estándar de Python: no instala ni ejecuta ningún modelo.

Idea general
------------
1. Para cada caso calculamos un conjunto de CARACTERÍSTICAS (números entre 0 y
   algo) que miden sospechas concretas:
     - qué dicen las señales precalculadas (juez LLM y NLI), cogiendo siempre
       la PEOR oración / fragmento, nunca la media;
     - si las cifras de la respuesta salen de las fuentes (directamente o
       haciendo cuentas: sumas, restas, porcentajes, proporciones);
     - si una cifra correcta se atribuye a otra entidad;
     - si la respuesta pierde matices de la fuente («solo», «prevé», «si se
       aprueba», «de los 14»...) o añade alcance («cualquier», «todos») o
       causas («gracias a», «porque»...);
     - cuántas palabras de la respuesta no aparecen en ninguna fuente.
2. Combinamos esas características con una regresión logística cuyos pesos se
   han aprendido con train.jsonl (ver entrenar.py, que los escribe abajo).
3. Reescalamos la probabilidad para que el corte óptimo (un escape cuesta 4
   veces más que una falsa alarma) caiga justo en 0,5, que es donde corta el
   organizador.
"""
import collections
import math
import re
import struct
import unicodedata
from itertools import combinations

# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def normalizar(texto):
    """Minúsculas y sin tildes, para comparar palabras sin que molesten acentos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")


# Palabras vacías: no aportan contenido y no las usamos para comparar.
VACIAS = set("""
a al algo ante antes asi aun bajo cada como con contra cual cuando de del desde
donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso
esos esta estaba estan estas este esto estos fue fueron ha han hasta hay la las
le les lo los mas me mientras muy ni no nos o otra otras otro otros para pero
por que quien se sea segun ser si sido sin sobre son su sus tambien tan tanto
te tiene tienen todo tras tu un una uno unos unas y ya yo año años
""".split())


def palabras(texto):
    """Lista de palabras normalizadas (solo letras)."""
    return re.findall(r"[a-zñ]+", normalizar(texto))


def raiz(palabra):
    """'Raíz' muy simple: los 5 primeros caracteres. Así 'plantó' y 'plantaron'
    o 'destinatario' y 'destinatarios' cuentan como la misma palabra."""
    return palabra[:5]


def raices_contenido(texto):
    """Conjunto de raíces de las palabras con contenido (sin palabras vacías)."""
    return {raiz(p) for p in palabras(texto) if p not in VACIAS and len(p) > 2}


def nombres_propios(texto):
    """Palabras en mayúscula que NO van al principio de frase: nombres de
    empresas, pueblos, organismos... Sirven para saber de QUIÉN se habla."""
    nombres = set()
    for frase in re.split(r"(?<=[.!?;:])\s+", texto):
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", frase)
        for t in tokens[1:]:
            if t[0].isupper() and len(t) > 2:
                nombres.add(normalizar(t))
    return nombres


# ---------------------------------------------------------------------------
# Cifras: extraerlas y comprobar si salen de las fuentes
# ---------------------------------------------------------------------------

UNIDADES = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}

# Frases que anuncian que una fuente habla de OTRA entidad comparable (v2)
RE_OTRA_ENTIDAD = re.compile(r"\b(vecin[oa]s?|competidor[ae]?s?|rival(es)?|por su parte|en cambio|"
                             r"mientras que|a diferencia|otr[oa]s? (empresa|entidad|ciudad|municipio|"
                             r"comarca|asociacion|ong|fundacion|cooperativa|banco|ayuntamiento|"
                             r"hospital|universidad|operadora|region|provincia))\b")
# Una hora: 10:05, 8:30
RE_HORA = re.compile(r"\b\d{1,2}:\d{2}\b")
# Una cifra escrita con dígitos al estilo español: 1.380 / 2,6 / 18.400 / 45
RE_CIFRA = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?")
# Una palabra-número (seis, treinta y dos...)
RE_PALABRA_NUM = re.compile(
    r"\b(" + "|".join(sorted(UNIDADES, key=len, reverse=True)) +
    r")(?:\s+y\s+(uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))?\b")


def _valor(cifra):
    return float(cifra.replace(".", "").replace(",", "."))


def extraer_numeros(texto):
    """Devuelve una lista de cifras del texto. Cada cifra es un dict con:
       valores: posibles valores (p. ej. 5,3 millones -> {5.3, 5300000})
       pct:     si va seguida de % o «por ciento»
       anio:    si parece un año (1900-2100)
       aprox:   si es una proporción «X de cada Y» (comparación aproximada)
    """
    t = normalizar(texto)
    # v2: las horas («10:05») no son dos cifras sueltas; las tapamos con espacios
    # (misma longitud, para no mover las posiciones) y se comprueban aparte.
    t = RE_HORA.sub(lambda m: " " * len(m.group()), t)
    encontrados = []  # (posicion_inicio, posicion_fin, valor)
    for m in RE_CIFRA.finditer(t):
        encontrados.append((m.start(), m.end(), _valor(m.group())))
    for m in RE_PALABRA_NUM.finditer(t):
        v = UNIDADES[m.group(1)]
        if m.group(2):
            v += UNIDADES.get(m.group(2), 0) if m.group(2) != "uno" else 1
        encontrados.append((m.start(), m.end(), float(v)))
    encontrados.sort()

    numeros = []
    usados = set()
    for i, (ini, fin, v) in enumerate(encontrados):
        if i in usados:
            continue
        resto = t[fin:fin + 25]
        # «seis de cada diez» -> 60 % aproximado
        m = re.match(r"\s+de\s+cada\s+", resto)
        if m and i + 1 < len(encontrados) and encontrados[i + 1][0] == fin + m.end():
            total = encontrados[i + 1][2]
            usados.add(i + 1)
            if total:
                numeros.append({"valores": {100.0 * v / total}, "pct": True,
                                "anio": False, "aprox": True, "ini": ini,
                                "fin": encontrados[i + 1][1]})
            continue
        valores = {v}
        m = re.match(r"\s*(mil\s+)?millones", resto)
        if m:
            valores.add(v * (1e9 if m.group(1) else 1e6))
        elif re.match(r"\s*mil\b", resto):
            valores.add(v * 1e3)
        if re.match(r"\s*centimos", resto):
            valores.add(v / 100.0)
        pct = bool(re.match(r"\s*(%|por\s*ciento)", resto))
        anio = v.is_integer() and 1900 <= v <= 2100 and not pct
        numeros.append({"valores": valores, "pct": pct, "anio": anio, "aprox": False,
                        "ini": ini, "fin": fin})
    return numeros


# Palabras que anuncian que la respuesta HACE UNA CUENTA con sus cifras. Si la
# respuesta dice «suman 1.980» o «cuatro menos que en 2024», ese número tiene
# que salir de ESA operación concreta, no de cualquier combinación.
RE_CUENTA_SUMA = re.compile(r"\b(suman|sumaron|suma|sumo|sumaba|sumando|en total|entre (los|las) (dos|tres|cuatro)"
                            r"|entre ambos|entre ambas|juntos|juntas|en conjunto)\b")
RE_CUENTA_DIF_DESPUES = re.compile(r"^\s*((%\s*|por ciento\s*|puntos?\s*|[a-z]+\s+)?(mas|menos)\b"
                                   r"|puntos?\b)")  # «25 puntos» = diferencia de porcentajes
RE_CUENTA_DIF_ANTES = re.compile(r"(diferencia|aumento|subida|incremento|rebaja|descenso|caida|"
                                 r"ganancia|perdida|ahorro|crecio|subio|bajo|aumento|cayo|mejoro|"
                                 r"empeoro|gano|perdio|redujo|anadio)\s+(en\s+|de\s+)?(unos?\s+|el\s+|un\s+)?$")
RE_CUENTA_CALCULO = re.compile(r"(es decir|lo que supone|lo que significa|lo que equivale|equivale a|"
                               r"o sea|lo que representa)[\s,]*(a\s+)?(unos?\s+|unas\s+|el\s+|la\s+|un\s+|una\s+)?$")


def tipo_de_cuenta(texto_norm, num):
    """'suma', 'diferencia', 'calculo' o None según las palabras alrededor de la cifra."""
    antes = texto_norm[max(0, num["ini"] - 40):num["ini"]]
    despues = texto_norm[num["fin"]:num["fin"] + 30]
    if RE_CUENTA_CALCULO.search(antes) or re.match(r"\s*veces\b", despues):
        return "calculo"  # «es decir, 11.600» / «seis veces más»: producto o proporción
    if RE_CUENTA_DIF_DESPUES.match(despues) or RE_CUENTA_DIF_ANTES.search(antes):
        return "diferencia"
    if RE_CUENTA_SUMA.search(antes[-25:] + " " + despues[:25]):
        return "suma"
    return None


def cuenta_cuadra(tipo, num, anclas, operandos):
    """¿Sale la cifra de una operación del tipo anunciado?
    anclas:    las otras cifras de la respuesta (si las hay, la cuenta debe usar una)
    operandos: cifras de las fuentes en las que se apoya la respuesta"""
    # Las anclas siempre entran (antes un recorte a 30 cifras tiraba las grandes)
    vals = sorted(set(anclas) | set(sorted(operandos)[:60]))
    candidatos = set()
    if tipo == "suma":
        for a, b in combinations(vals, 2):
            candidatos.update({a + b, a * b})  # «en total» también puede ser un producto
        for a, b, c in combinations(vals[:30], 3):
            candidatos.add(a + b + c)
    else:
        bases = anclas if anclas else vals
        for a in bases:
            for b in vals:
                if a == b:
                    continue
                candidatos.add(abs(a - b))
                if tipo == "calculo":
                    candidatos.update({a * b / 100.0, a * b})
                    if b:
                        candidatos.update({100.0 * a / b, a / b})
                    if a:
                        candidatos.update({100.0 * b / a, b / a})
    return any(_cerca(v, c) for v in num["valores"] for c in candidatos)


# ---------------------------------------------------------------------------
# Fechas y plazos (v2)
# ---------------------------------------------------------------------------

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
         "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
         "noviembre": 11, "diciembre": 12}
RE_FECHA = re.compile(r"(?:(\d{1,2}) de )?(" + "|".join(MESES) + r")(?: de(?:l)? (\d{4}))?")
RE_PLAZO = re.compile(r"\b(\d+|" + "|".join(UNIDADES) + r"|un|una)\s+(dias|semanas|meses|anos)\s+"
                      r"(despues|antes|mas tarde|mas)\b")


def _fechas(texto_norm):
    """Fechas del texto como (año, mes, día); año o día pueden faltar (None)."""
    fechas = []
    for m in RE_FECHA.finditer(texto_norm):
        dia = int(m.group(1)) if m.group(1) else None
        anio = int(m.group(3)) if m.group(3) else None
        fechas.append((anio, MESES[m.group(2)], dia))
    return fechas


def comprobar_plazos(respuesta, texto_fuentes):
    """Cuenta plazos («N meses después») que (no) salen de restar dos fechas.
    Devuelve (falsos, correctos)."""
    r = normalizar(respuesta)
    s = normalizar(texto_fuentes)
    falsos = correctos = 0
    for m in RE_PLAZO.finditer(r):
        palabra, unidad = m.group(1), m.group(2)
        n = 1 if palabra in ("un", "una") else (UNIDADES.get(palabra) or int(palabra))
        # Fechas disponibles: las de las fuentes y las que cita la propia respuesta.
        # Si la respuesta cita fechas, el plazo debe contarse DESDE una de ellas
        # («el 30 de junio de 2027, y nueve meses después»); si no, entre dos
        # fechas cualesquiera de las fuentes.
        f_fuentes, f_resp = _fechas(s), _fechas(r)
        a_fuentes = {int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", s)}
        a_resp = {int(a) for a in re.findall(r"\b(19\d\d|20\d\d)\b", r)}
        diferencias = set()
        if unidad == "anos":
            origen = a_resp or a_fuentes
            diferencias = {abs(a - b) for a in origen for b in a_fuentes | a_resp}
        elif unidad == "meses":
            # Completamos el año que falte con el de otra fecha («junio» -> «junio de 2027»)
            con_anio = [a for a, _, _ in f_resp + f_fuentes if a]
            defecto = con_anio[0] if con_anio else 2000
            m_fuentes = {(a or defecto) * 12 + mes for a, mes, _ in f_fuentes}
            m_resp = {(a or defecto) * 12 + mes for a, mes, _ in f_resp}
            origen = m_resp or m_fuentes
            diferencias = {abs(x - y) for x in origen for y in m_fuentes | m_resp}
        else:
            # Días o semanas: si no podemos calcularlo, no opinamos
            continue
        if not diferencias - {0}:
            continue
        if n in diferencias:
            correctos += 1
        else:
            falsos += 1
    return falsos, correctos


# ---------------------------------------------------------------------------
# Verificador de cuentas DENTRO de la frase (v4)
# ---------------------------------------------------------------------------
# Hasta v3, para saber si una cifra derivada era correcta buscábamos CUALQUIER
# combinación de cifras de las fuentes que la diera. Con tantas cifras, casi
# siempre aparecía alguna por casualidad, y así se escapaban cuentas falsas
# como «de 212 a 258 millones, un 12 %» (en realidad un 21,7 %).
# Aquí reconocemos la forma concreta de la cuenta que hace la frase y
# recalculamos ESA operación con las cifras que la propia frase menciona:
#   A. cambio:        «de A a B, un X %» / «X puntos» / «X más»
#   B. producto:      «A habitantes y B euros por habitante ... fue de X»
#   C. suma nombrada: «el fútbol sala y el baloncesto sumaron X»
#   D. comparación:   «X más/menos que [otra entidad / otro año]»
#   E. parte del total: «perdió A, un X % de su población»
# Cada cifra comprobada queda marcada como «ok» o «mal».

MESES_RE = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
RE_DIA_MES = re.compile(r"\s+de\s+(" + MESES_RE + r")\b")
RE_NUMERO_ORDEN = re.compile(r"(n\.?\s*[o\u00ba\u00b0]\.?|numero)\s*$")
RE_SUMA_NOMBRADA = re.compile(r"\b(suman|sumaron|suma|sumo|sumaban|sumarian|entre (los|las) (dos|tres)|en total)\b")
RE_VERBO_CAMBIO = re.compile(r"\b(aumento|aumentaron|crecio|crecieron|subio|subieron|bajo|bajaron|cayo|"
                             r"cayeron|redujo|mejoro|empeoro|gano|perdio|un|una)\s*([a-z]+\s+)?$")
RE_DESDE = re.compile(r"\b(de|del|desde)\s*$")
RE_HASTA = re.compile(r"\b(a|al|hasta)\s+(los\s+|las\s+|el\s+|la\s+)?$")
RE_TASA = re.compile(r"\bpor (habitante|persona|vecino|alumno|escolar|metro|kilo|litro|hora|dia|mes|"
                     r"ano|unidad|plaza|cama|hectarea|trabajador|empleado|socio|cliente|vivienda|"
                     r"hogar|familia|tripulante|deportista|participante|beneficiario|usuario)|"
                     r"\bal (mes|ano|dia)\b|\bmensual|\banual")
RE_RESULTADO = re.compile(r"(fue de|es de|sera de|seran|asciende a|ascendio a|supone|suponen|"
                          r"pagara|pagaran|paga|pagan|cuesta|costara|costo|ingresara|ingresa|"
                          r"recibira|recibe|total de|en total|es decir|lo que da|resulta|"
                          r"alcanza|alcanzo|suma|suman)\s+(unos?\s+|unas\s+|el\s+|los\s+|de\s+)?$")
RE_COMPARA = re.compile(r"^\s*(%\s*|por ciento\s*|puntos?\s*)?([a-z]+\s+){0,2}?(mas|menos|mayor|menor|"
                        r"superior|inferior)\s+(que|a|al)\s+")
RE_PARTE = re.compile(r"^[\s,]*(lo que supone\s+|es decir,?\s+)?(un|el)\s*$")
RE_PERDIDA = re.compile(r"\b(perdio|perdieron|pierde|bajo|bajaron|cayo|cayeron|redujo|redujeron|"
                        r"disminuyo|descendio|menos)\b")
RE_GANANCIA = re.compile(r"\b(gano|ganaron|gana|subio|subieron|crecio|crecieron|aumento|aumentaron)\b")
GENERICAS = {"total", "entre", "ambos", "ambas", "sumar", "suman", "sumar"}


def _valores_no_anio(nums):
    return [v for n in nums if not n["anio"] for v in n["valores"]]


def _casi(real, dicho, texto_antes):
    """¿Cuadra la cifra «dicha» con la calculada («real»)? Respeta «más de»,
    «casi», «unos»... que la frase ponga delante."""
    if real is None:
        return False
    if re.search(r"(mas del?|por encima del?|superior al?)\s*(un\s+|el\s+)?$", texto_antes):
        return dicho * 0.98 <= real <= dicho * 1.6 + 0.5
    if re.search(r"(menos del?|por debajo del?|inferior al?)\s*(un\s+|el\s+)?$", texto_antes):
        return dicho * 0.4 - 0.5 <= real <= dicho * 1.02
    if re.search(r"(casi|cerca de|alrededor de|unos|unas|aproximadamente)\s*(un\s+|el\s+)?$", texto_antes):
        return abs(real - dicho) <= max(0.1 * abs(dicho), 0.6)
    return abs(real - dicho) <= max(0.03 * abs(dicho), 0.6 if dicho >= 3 else 0.06)


def _fuentes_que_hablan_de(texto_norm, fuentes_norm):
    """Fuentes que más palabras (o años) comparten con un trozo de texto."""
    raices = raices_contenido(texto_norm) - GENERICAS
    anios = set(re.findall(r"\b(?:19|20)\d\d\b", texto_norm))
    puntos = []
    for i, fn in enumerate(fuentes_norm):
        p = len(raices & raices_contenido(fn)) + 2 * sum(1 for a in anios if a in fn)
        puntos.append(p)
    mejor = max(puntos) if puntos else 0
    return [i for i, p in enumerate(puntos) if p == mejor and p > 0]


def verificar_cuentas_frase(resp_norm, nums_resp, fuentes, nums_por_fuente):
    """Devuelve {índice de la cifra en nums_resp: ("ok"|"mal", tipo_de_cuenta)}."""
    fuentes_norm = [normalizar(x) for x in fuentes]
    todas_fuente = _valores_no_anio([n for ns in nums_por_fuente for n in ns])
    resultado = {}
    # Trocear la respuesta en frases, respetando posiciones
    cortes = [0] + [m.end() for m in re.finditer(r"[.;](\s|$)", resp_norm)] + [len(resp_norm)]
    for ini_f, fin_f in zip(cortes, cortes[1:]):
        en_frase = [k for k, n in enumerate(nums_resp) if ini_f <= n["ini"] < fin_f]
        # No cuentan los años, los días de una fecha («31 de julio») ni «el n.º 2»
        utiles = [k for k in en_frase if not nums_resp[k]["anio"]
                  and not RE_DIA_MES.match(resp_norm[nums_resp[k]["fin"]:nums_resp[k]["fin"] + 15])
                  and not RE_NUMERO_ORDEN.search(resp_norm[max(0, nums_resp[k]["ini"] - 8):nums_resp[k]["ini"]])]

        def copiada(k):
            """La cifra está tal cual en una fuente que habla de lo mismo (mismo
            tipo, % o no, y al menos 2 palabras en común con la frase): es un
            dato copiado, no una cuenta de la respuesta. No la juzgamos como «mal»."""
            raices_frase = raices_contenido(resp_norm[ini_f:fin_f])
            for i, ns in enumerate(nums_por_fuente):
                for n in ns:
                    if n["pct"] == nums_resp[k]["pct"] and any(
                            _cerca(v, w, 0.001) for v in n["valores"] for w in nums_resp[k]["valores"]):
                        if len(raices_frase & raices_contenido(fuentes_norm[i])) >= 2:
                            return True
            return False

        def anotar(k, ok, tipo):
            if ok:
                resultado[k] = ("ok", tipo)
            elif not copiada(k):
                resultado[k] = ("mal", tipo)

        def antes(k, ancho=30):
            return resp_norm[max(ini_f, nums_resp[k]["ini"] - ancho):nums_resp[k]["ini"]]

        def despues(k, ancho=40):
            return resp_norm[nums_resp[k]["fin"]:min(fin_f, nums_resp[k]["fin"] + ancho)]

        def base(k):
            return min(nums_resp[k]["valores"])  # el valor «tal cual» (sin millones)

        # --- A. Cambio «de A a B» + cifra que lo resume ---
        for x, a in enumerate(utiles):
            if not RE_DESDE.search(antes(a, 12)):
                continue
            siguientes = [k for k in utiles if nums_resp[k]["ini"] > nums_resp[a]["fin"]]
            if not siguientes:
                continue
            b = siguientes[0]
            if not RE_HASTA.search(resp_norm[nums_resp[a]["fin"]:nums_resp[b]["ini"]]):
                continue
            va, vb = base(a), base(b)
            for c in utiles:
                if c in (a, b) or c in resultado:
                    continue
                # Pegada detrás («de 6,4 a 6,8 millones, un 8 % más») o delante
                # («aumentó un 12 % en 2025, de 212 a 258»), sin saltar a otra cláusula
                if nums_resp[c]["ini"] > nums_resp[b]["fin"]:
                    hueco = resp_norm[nums_resp[b]["fin"]:nums_resp[c]["ini"]]
                    pegada = (len(hueco) <= 45 and ";" not in hueco
                              and re.search(r",\s*([a-z]+\s+){0,6}$", hueco))
                else:
                    hueco = resp_norm[nums_resp[c]["fin"]:nums_resp[a]["ini"]]
                    pegada = (nums_resp[c]["fin"] <= nums_resp[a]["ini"] and len(hueco) <= 35
                              and ";" not in hueco and RE_VERBO_CAMBIO.search(antes(c, 25)))
                if not pegada:
                    continue
                tras = despues(c, 25)
                dicho = base(c)
                if re.match(r"\s*puntos?\b", tras):
                    real = abs(vb - va)
                elif nums_resp[c]["pct"]:
                    reales = [abs(vb - va) / va * 100 if va else None]
                    if nums_resp[a]["pct"] and nums_resp[b]["pct"]:
                        reales.append(abs(vb - va))  # «subió 15 puntos» dicho como «un 15 %»
                    anotar(c, any(_casi(r, dicho, antes(c)) for r in reales), "cambio")
                    continue
                elif re.match(r"\s*([a-z]+\s+){0,2}(mas|menos)\b", tras):
                    real = abs(vb - va)
                elif re.match(r"\s*veces\b", tras):
                    real = max(va, vb) / min(va, vb) if min(va, vb) else None
                else:
                    continue
                anotar(c, _casi(real, dicho, antes(c)), "cambio")

        # --- B. Producto con una tasa («por habitante», «al mes») ---
        frase = resp_norm[ini_f:fin_f]
        if RE_TASA.search(frase) and len(utiles) >= 3:
            r = utiles[-1]
            if r not in resultado and RE_RESULTADO.search(antes(r, 35)):
                factores = [1.0]
                if re.search(r"\bal mes\b|\bmensual", frase) and re.search(r"\bal ano\b|\banual", frase):
                    factores += [12.0, 1 / 12.0]
                ops = [nums_resp[k]["valores"] for k in utiles[:-1]]
                ok = False
                for tam in (2, 3):
                    for combo in combinations(ops, tam):
                        prods = [1.0]
                        for vals in combo:
                            prods = [p * v for p in prods for v in vals]
                        if any(_cerca(rv, p * fct, 0.02) for p in prods for fct in factores
                               for rv in nums_resp[r]["valores"]):
                            ok = True
                anotar(r, ok, "producto")

        # --- C. Suma de cosas con nombre: «A y B sumaron X» ---
        m = RE_SUMA_NOMBRADA.search(frase)
        if m:
            pos_marca = ini_f + m.start()
            # (el «dos» de «entre los dos» no es el resultado de la suma)
            tras_marca = [k for k in utiles if ini_f + m.end() <= nums_resp[k]["ini"] <= ini_f + m.end() + 25
                          and not re.search(r"entre (los|las|ambos)\s*$", antes(k, 15))]
            if tras_marca and tras_marca[0] not in resultado:
                x = tras_marca[0]
                sujeto = resp_norm[ini_f:pos_marca]
                # Partes unidas por «y»; si hay «y», las comas también separan
                # («marroquí, rumana y colombiana»). Sin «y» no hay suma de cosas
                # («el verso, un endecasílabo, suma once sílabas»).
                trozos = re.split(r"\s(?:y|e)\s", sujeto)
                if len(trozos) >= 2:
                    trozos = re.split(r",", trozos[0])[-3:] + trozos[1:]
                partes = [p for p in trozos if raices_contenido(p) - GENERICAS]
                if len(partes) >= 2:
                    candidatos = []
                    for p in partes:
                        idx = _fuentes_que_hablan_de(p, fuentes_norm)
                        # Si la parte nombra una cifra en la propia respuesta, también vale
                        vals = set(_valores_no_anio([n for k2, n in enumerate(nums_resp)
                                                     if k2 in utiles and n["ini"] < pos_marca
                                                     and p.strip() and p.strip()[:8] in resp_norm]))
                        for i in idx:
                            # Solo las cifras que están junto a las palabras de esa parte
                            for n in nums_por_fuente[i]:
                                if n["anio"]:
                                    continue
                                cerca_txt = fuentes_norm[i][max(0, n["ini"] - 60):n["fin"] + 20]
                                if raices_contenido(p) & raices_contenido(cerca_txt):
                                    vals.update(n["valores"])
                        candidatos.append(sorted(vals)[:25])
                    if all(candidatos) and len(candidatos) <= 4:
                        sumas = [0.0]
                        for vals in candidatos:
                            sumas = [s + v for s in sumas for v in vals][:20000]
                        ok = any(_cerca(xv, s, 0.01) for s in sumas for xv in nums_resp[x]["valores"])
                        anotar(x, ok, "suma")

        # --- D. Comparación: «X más/menos que [otra entidad / otro año]» ---
        for c in utiles:
            if c in resultado:
                continue
            tras = resp_norm[nums_resp[c]["fin"]:fin_f]
            m = RE_COMPARA.match(tras)
            if not m:
                continue
            objetivo = re.split(r"[,;]|\sy\s", tras[m.end():m.end() + 60])[0]
            anclas = [k for k in utiles if k != c and nums_resp[k]["ini"] < nums_resp[c]["ini"]]
            idx = set(_fuentes_que_hablan_de(objetivo, fuentes_norm))
            idx |= {i for i, ns in enumerate(nums_por_fuente) for k in anclas
                    if any(_cerca(v, w, 0.001) for n in ns for v in n["valores"] for w in nums_resp[k]["valores"])}
            if not anclas or not idx:
                continue
            otras = _valores_no_anio([n for i in idx for n in nums_por_fuente[i]])
            # La otra cifra puede venir en la propia frase («... que las 4 horas de Lusel»)
            fin_obj = nums_resp[c]["fin"] + m.end() + len(objetivo)
            otras += _valores_no_anio([nums_resp[k] for k in utiles
                                       if nums_resp[c]["fin"] < nums_resp[k]["ini"] < fin_obj])
            dicho = base(c)
            pares = [(av, o) for k in anclas for av in nums_resp[k]["valores"] for o in otras if o]
            if re.match(r"\s*veces\b", tras):
                # «seis veces más que»: proporción, no resta
                ok = any(_casi(max(av, o) / min(av, o), dicho, antes(c)) for av, o in pares if min(av, o))
            elif nums_resp[c]["pct"] and not any(nums_resp[k]["pct"] for k in anclas):
                # «9.800 préstamos, un 15 % más que en 2024»: cambio relativo
                ok = any(_casi(abs(av - o) / o * 100, dicho, antes(c)) for av, o in pares)
            else:
                ok = any(_casi(abs(av - o), dicho, antes(c)) for av, o in pares)
            anotar(c, ok, "comparacion")

        # --- E. Parte del total: «perdió 2.410 habitantes, un 18 % de su población» ---
        for c in utiles:
            if c in resultado or not nums_resp[c]["pct"]:
                continue
            previas = [k for k in utiles if nums_resp[k]["fin"] < nums_resp[c]["ini"]
                       and not nums_resp[k]["pct"]]
            if not previas:
                continue
            x = previas[-1]
            entre = resp_norm[nums_resp[x]["fin"]:nums_resp[c]["ini"]]
            if not (re.search(r",\s*(lo que supone\s+|es decir,?\s+)?(un|el)\s*$", entre)
                    and re.match(r"\s*(%|por ciento)\s+(de|del)\b", resp_norm[nums_resp[c]["fin"]:fin_f])):
                continue
            vx = base(x) if base(x) > 1000 or not any(v > 1000 for v in nums_resp[x]["valores"]) else max(nums_resp[x]["valores"])
            # La base del porcentaje es la cifra de PARTIDA: si perdió X, la
            # base B cumple que B - X también está en las fuentes (si ganó, B + X)
            if RE_PERDIDA.search(frase):
                bases = [b for b in todas_fuente if any(_cerca(b - vx, o) for o in todas_fuente)]
            elif RE_GANANCIA.search(frase):
                bases = [b for b in todas_fuente if any(_cerca(b + vx, o) for o in todas_fuente)]
            else:
                bases = [b for b in todas_fuente if b > vx]  # «119.040 TEU, el 64 % del total»
            if not bases:
                continue
            dicho = base(c)
            anotar(c, any(_casi(100.0 * vx / b, dicho, antes(c)) for b in bases if b), "parte")
    return resultado


def _cerca(a, b, tolerancia_rel=0.01):
    """¿Son a y b el mismo número salvo redondeo?"""
    return abs(a - b) <= max(tolerancia_rel * abs(b), 0.051 if not float(a).is_integer() else 0.5)


def valores_derivados(valores_fuente):
    """Todo lo que se puede calcular con dos (o tres) cifras de las fuentes:
    sumas, restas, porcentajes de una cantidad, proporciones..."""
    vals = sorted(valores_fuente)[:60]  # límite por si hay muchísimas cifras
    derivados = set()
    for a, b in combinations(vals, 2):
        derivados.update({a + b, abs(a - b), a * b / 100.0, a * b})
        if b:
            derivados.update({100.0 * a / b, a / b})
        if a:
            derivados.update({100.0 * b / a, b / a})
    for a, b, c in combinations(vals[:30], 3):
        derivados.add(a + b + c)
    return derivados


def clasificar_numero(num, valores_fuente, derivados):
    """'directo' si la cifra está en las fuentes, 'derivado' si sale de hacer
    cuentas con ellas y 'inexplicado' si no sale de ningún sitio."""
    tol = 0.12 if num["aprox"] else 0.01
    for v in num["valores"]:
        if any(_cerca(v, f, tol) for f in valores_fuente):
            return "directo"
    for v in num["valores"]:
        if any(_cerca(v, d, tol) for d in derivados):
            return "derivado"
    return "inexplicado"


# Afirmaciones de proporción: «el doble», «cuadruplica», «seis veces», «un tercio»...
PROPORCIONES = [
    (r"\bdoble\b|\bduplica", 2.0), (r"\btriple\b|\btriplica", 3.0),
    (r"\bcuadruplica|\bcuadruple\b", 4.0), (r"\bquintuplica", 5.0),
    (r"\bla mitad\b", 0.5), (r"\bun tercio\b|\btercera parte\b", 1 / 3.0),
    (r"\bdos tercios\b", 2 / 3.0), (r"\bun cuarto\b|\bcuarta parte\b", 0.25),
    (r"\btres cuartos\b", 0.75),
]


def proporciones_falsas(texto_resp, numeros_resp, valores_fuente):
    """Cuenta afirmaciones de proporción que NO cuadran con ningún par de cifras
    (al menos una de las dos debe aparecer en la respuesta)."""
    t = normalizar(texto_resp)
    afirmaciones = []  # (k, es_minimo): es_minimo si dice «más de(l) ...»

    def es_minimo(inicio):
        return bool(re.search(r"(mas|algo mas) del?\s*(un|una|la|el)?\s*$", t[max(0, inicio - 14):inicio]))

    for patron, k in PROPORCIONES:
        m = re.search(patron, t)
        if m:
            afirmaciones.append((k, es_minimo(m.start())))
    for m in re.finditer(r"(\d+(?:,\d+)?|" + "|".join(UNIDADES) + r")\s+veces\b", t):
        palabra = m.group(1)
        k = UNIDADES.get(palabra) or _valor(palabra)
        afirmaciones.append((float(k), False))
    if not afirmaciones:
        return 0
    de_resp = {v for n in numeros_resp for v in n["valores"] if v}
    todos = {v for v in de_resp | set(valores_fuente) if v}
    falsas = 0
    for k, minimo in afirmaciones:
        ok = False
        for a in (de_resp or todos):
            for b in todos:
                if a == b:
                    continue
                # Probamos a/b y b/a; k < 1 son fracciones (un tercio), k > 1 múltiplos
                for r in (a / b, b / a):
                    if minimo and k <= r <= k * 1.6:
                        ok = True
                    elif abs(r - k) <= 0.12 * k:
                        ok = True
            if ok:
                break
        if not ok:
            falsas += 1
    return falsas


# ---------------------------------------------------------------------------
# Matices, alcance y causas
# ---------------------------------------------------------------------------

# Matices que restringen o suavizan una afirmación. Si la fuente los tiene y la
# respuesta no, es típico de «sobregeneralización» o de «endurecer un verbo».
MATICES = {
    "restrictivo": [r"\bsolo\b", r"\bunicamente\b", r"\bexclusivamente\b", r"\bsalvo\b",
                    r"\bexcepto\b", r"\bsiempre que\b", r"\ba condicion\b", r"\bse limita",
                    r"\bsolamente\b"],
    "condicional": [r"\bsi se\b", r"\bsi lo\b", r"\bsi la\b", r"\bsi el\b", r"\bsi los\b",
                    r"\bsi las\b", r"\bpreve\b", r"\bprevista?\b", r"\bprevisto\b",
                    r"\bestudia\b", r"\bplantea\b", r"\bpropone\b", r"\bpropuesta\b",
                    r"\bpodria", r"\bposible", r"\bpendiente", r"\bprovisional",
                    r"\bestima\b", r"\bestimo\b", r"\bespera\b", r"\bpretende\b",
                    r"\bproyecto\b", r"\bplanea\b", r"\bdepend", r"\bcondicionad",
                    r"\bsujet[oa]s? a\b", r"\ba expensas\b", r"\bse confirm"],
    "aproximado": [r"\bmas de\b", r"\bmenos de\b", r"\bal menos\b", r"\bcasi\b",
                   r"\bcerca de\b", r"\bunos\b", r"\bunas\b", r"\balrededor de\b",
                   r"\baproximadamente\b", r"\bhasta\b", r"\bcomo maximo\b",
                   r"\bcomo minimo\b", r"\bmaximo\b"],
    "parcial": [r"\balgunos\b", r"\balgunas\b", r"\bvarios\b", r"\bvarias\b",
                r"\bparte de\b", r"\bmayoria\b", r"\bde los \d", r"\bde las \d",
                r"\bentre \d", r"\bde ell[oa]s\b"],
}

# Palabras de alcance universal: si la respuesta las añade, puede estar ampliando.
# (Versión 2: quitamos «general», «toda la», «todo el», que salían en muchas
# respuestas fieles y solo metían ruido.)
UNIVERSALES = [r"\btodos\b", r"\btodas\b", r"\bcualquier", r"\bsiempre\b", r"\bnunca\b",
               r"\bningun", r"\bnadie\b", r"\btotalidad\b", r"\bintegramente\b",
               r"\bcompletamente\b", r"\bdel pais\b", r"\bde espana\b", r"\bsin excepcion"]

# Conectores de causa / consecuencia: si la respuesta los añade, puede estar
# inventándose un porqué. (Versión 2: fuera «para», que salía casi siempre.)
CAUSALES = [r"\bporque\b", r"\bya que\b", r"\bpuesto que\b", r"\bdebido a\b",
            r"\bgracias a\b", r"\bpor su\b", r"\bpor sus\b", r"\blo que permit",
            r"\bpermitio\b", r"\bpermitira\b", r"\bpor lo que\b", r"\bde ahi\b",
            r"\ba raiz de\b", r"\bprovoc", r"\bcausad", r"\bcomo consecuencia\b",
            r"\bcomo resultado\b", r"\bfruto de\b", r"\bpor ello\b", r"\bpor eso\b",
            r"\basi que\b", r"\bcon lo que\b", r"\bpara que\b", r"\bpara (ampliar|reducir|mejorar|evitar|financiar|cubrir)\b"]


def contar(patrones, texto_norm):
    return sum(1 for p in patrones if re.search(p, texto_norm))


def perdidos(patrones, fuente_norm, resp_norm):
    """Matices que están en la fuente pero no en la respuesta."""
    return sum(1 for p in patrones if re.search(p, fuente_norm) and not re.search(p, resp_norm))


def anadidos(patrones, fuentes_norm, resp_norm):
    """Expresiones que están en la respuesta pero en ninguna fuente."""
    return sum(1 for p in patrones if re.search(p, resp_norm) and not re.search(p, fuentes_norm))


def tramo_cubierto(fuente, raices_resp):
    """Trozo de la fuente que la respuesta 'usa': desde la primera hasta la
    última palabra de la fuente que también sale en la respuesta. Devuelve
    (texto_del_tramo, nº de palabras con contenido del tramo que la respuesta
    se ha dejado)."""
    tokens = re.findall(r"[a-zñ0-9.,%]+", normalizar(fuente))
    pos = [i for i, tok in enumerate(tokens)
           if (tok.isalpha() and tok not in VACIAS and raiz(tok) in raices_resp)]
    if len(pos) < 2:
        return "", 0
    tramo = tokens[min(pos):max(pos) + 1]
    huecos = sum(1 for tok in tramo
                 if tok.isalpha() and tok not in VACIAS and len(tok) > 2
                 and raiz(tok) not in raices_resp)
    return " ".join(tramo), huecos


# ---------------------------------------------------------------------------
# Características de un caso
# ---------------------------------------------------------------------------

def caracteristicas(caso):
    """Calcula todas las características (dict nombre -> número) de un caso."""
    fuentes = caso.get("fuentes") or []
    respuesta = caso.get("respuesta") or ""
    pregunta = caso.get("pregunta") or ""
    senales = caso.get("senales") or {}
    oraciones = senales.get("oraciones") or []
    fragmentos = senales.get("fragmentos") or []
    n_fuentes = len(fuentes)
    f = {}

    # --- 1. Señales del juez LLM (la PEOR oración manda) ---
    veredictos = [((o.get("llm") or {}).get("veredicto")) for o in oraciones]
    confianzas = [((o.get("llm") or {}).get("confianza") or 0.0) for o in oraciones]
    f["llm_no_respaldada"] = float(any(v == "no_respaldada" for v in veredictos))
    f["llm_contradicha"] = float(any(v == "contradicha" for v in veredictos))
    f["llm_sin_veredicto"] = float(any(v is None for v in veredictos))
    conf_resp = [c for v, c in zip(veredictos, confianzas) if v == "respaldada"]
    f["llm_conf_min_respaldada"] = min(conf_resp) if conf_resp else 0.0

    # --- 2. Señales del NLI ---
    def mins(lista, campo):
        vals = [x.get(campo, 0.0) for x in lista]
        return min(vals) if vals else 0.0

    def maxs(lista, campo):
        vals = [x.get(campo, 0.0) for x in lista]
        return max(vals) if vals else 0.0

    f["nli_resp_todas_min_orac"] = mins(oraciones, "respaldo_todas")
    f["nli_resp_todas_min_frag"] = mins(fragmentos, "respaldo_todas") if fragmentos else f["nli_resp_todas_min_orac"]
    f["nli_contr_todas_max_orac"] = maxs(oraciones, "contradiccion_todas")
    f["nli_contr_todas_max_frag"] = maxs(fragmentos, "contradiccion_todas")
    # La mejor fuente individual para la peor oración (lo que hacía el bot de ejemplo)
    f["nli_mejor_fuente_min_orac"] = min(
        [max(o.get("respaldo_por_fuente") or [0.0]) for o in oraciones] or [0.0])
    f["nli_contr_fuente_max_frag"] = max(
        [max(x.get("contradiccion_por_fuente") or [0.0]) for x in fragmentos] or [0.0])

    # Respaldo NLI de la fuente que cita el LLM: si el LLM dice «respaldada por
    # la fuente 3» y el NLI dice que la fuente 3 no la respalda, desconfiamos.
    citadas = set()
    resp_citada = []
    for o in oraciones:
        llm = o.get("llm") or {}
        i = llm.get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            citadas.add(i)
            rpf = o.get("respaldo_por_fuente") or []
            if i < len(rpf):
                resp_citada.append(rpf[i])
        sim = o.get("similitud_por_fuente") or []
        if sim:  # la fuente más parecida a cada oración también es «relevante»
            citadas.add(max(range(len(sim)), key=lambda k: sim[k]))
    f["nli_resp_fuente_citada_min"] = min(resp_citada) if resp_citada else 0.0

    # --- 2b. Reparto del respaldo entre fragmentos (v4) ---
    # Hasta ahora solo mirábamos el PEOR fragmento. Pero importa también
    # cuántos fragmentos flojean y cómo está el segundo peor: una respuesta con
    # un único fragmento flojo suele ser una paráfrasis; con varios, una invención.
    # Para cada fragmento tomamos el mejor respaldo, sea de una fuente sola o de
    # todas juntas (así no penalizamos las respuestas que combinan fuentes).
    resp_frag = sorted(max([x.get("respaldo_todas", 0.0)] + list(x.get("respaldo_por_fuente") or []))
                       for x in fragmentos) or [0.0]
    f["frag_respaldo_min"] = resp_frag[0]
    f["frag_respaldo_2o"] = resp_frag[1] if len(resp_frag) > 1 else resp_frag[0]
    # Fracciones (no recuentos), para que las respuestas largas del giro no suban solas
    f["frag_frac_bajo_05"] = sum(1 for v in resp_frag if v < 0.5) / len(resp_frag)
    f["frag_frac_bajo_02"] = sum(1 for v in resp_frag if v < 0.2) / len(resp_frag)
    f["frag_similitud_min"] = min([max(x.get("similitud_por_fuente") or [0.0]) for x in fragmentos] or [0.0])

    # --- 3. Cifras ---
    texto_fuentes = " ".join(fuentes) + " " + pregunta
    nums_fuente = extraer_numeros(texto_fuentes)
    valores_fuente = {v for n in nums_fuente for v in n["valores"]}
    nums_resp = extraer_numeros(respuesta)
    nums_por_fuente = [extraer_numeros(x) for x in fuentes]

    def esta_en(n, lista_nums):
        return any(_cerca(v, w) for v in n["valores"] for m in lista_nums for w in m["valores"])

    # Cifras de la respuesta que están literalmente en las fuentes
    directos = [n for n in nums_resp if esta_en(n, nums_fuente)]

    # Operandos «locales»: las cifras directas de la respuesta y las de las
    # fuentes en las que se apoya (las que contienen esas cifras y las citadas
    # o más parecidas). Las cuentas fieles casi siempre salen de ahí. Si solo
    # cuadran combinando cifras sueltas de todo el texto, es casualidad.
    # No usamos años (2026 - 2025 = 1 explicaría cualquier «1»).
    fuentes_base = set(citadas) | {i for i, ns in enumerate(nums_por_fuente)
                                   if any(esta_en(d, ns) for d in directos)}

    def operandos_de(lista_nums):
        ops = set()
        for n in lista_nums:
            if n["anio"]:
                continue
            ops.update(n["valores"])
            if n["pct"]:  # «el 70 %... el resto» -> también vale el 30 %
                ops.update(100.0 - v for v in n["valores"] if v < 100)
        return ops

    ops_locales = operandos_de(directos) | {100.0}
    for i in fuentes_base:
        ops_locales |= operandos_de(nums_por_fuente[i])
    ops_globales = operandos_de(nums_fuente)
    # Cifras de las fuentes que nombran a alguien que también nombra la respuesta
    nombres_fuentes = [nombres_propios(x) for x in fuentes]
    # nombres_propios() ignora la primera palabra de cada frase (siempre va en
    # mayúscula). Pero «Lusel retiene...» empieza por un nombre: lo recuperamos
    # si esa palabra aparece como nombre propio en las fuentes.
    nombres_conocidos = set().union(*nombres_fuentes) if nombres_fuentes else set()
    nombres_resp = nombres_propios(respuesta) | {
        normalizar(m) for m in re.findall(r"(?:^|[.;:!?]\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)", respuesta)
        if normalizar(m) in nombres_conocidos}
    fuentes_nombradas = set(fuentes_base) | {i for i, nf in enumerate(nombres_fuentes) if nf & nombres_resp}
    ops_nombradas = set(ops_locales)
    for i in fuentes_nombradas:
        ops_nombradas |= operandos_de(nums_por_fuente[i])
    der_locales = valores_derivados(ops_locales) if nums_resp else set()
    der_globales = valores_derivados(ops_globales) if nums_resp else set()
    resp_norm = normalizar(respuesta)

    # v4: primero el verificador de cuentas dentro de la frase. Si reconoce la
    # cuenta, su veredicto manda sobre las comprobaciones genéricas de abajo.
    verificadas = verificar_cuentas_frase(resp_norm, nums_resp, fuentes, nums_por_fuente)
    # v4: «el 30 % de 18.400» o «el resto (30 %) de 18.400»: porcentaje de una
    # cantidad, aunque estén en fuentes distintas
    pct_fuentes = [v for n in nums_fuente if n["pct"] for v in n["valores"] if v < 100]
    totales = [v for n in nums_fuente if not n["pct"] and not n["anio"] for v in n["valores"]]
    partes_de_total = {p * t / 100.0 for p in pct_fuentes + [100.0 - q for q in pct_fuentes] for t in totales}

    clases = []
    for k, n in enumerate(nums_resp):
        # Otras cifras de la respuesta («anclas»): una cuenta fiel suele usarlas
        anclas = operandos_de([d for d in directos if d is not n])
        tipo = tipo_de_cuenta(resp_norm, n)
        if k in verificadas:
            clases.append("cuenta_correcta" if verificadas[k][0] == "ok" else "cuenta_falsa")
        elif tipo and not n["anio"] and any(esta_en(n, nums_por_fuente[i]) for i in fuentes_base):
            # «pagarán un 9 % más»: no es una cuenta, el 9 % está tal cual en la
            # fuente en la que se apoya la respuesta. (v2: esto se mira ANTES de
            # buscar cuentas; si no, alguna resta casual lo «explicaba».)
            clases.append("directo")
        elif tipo and not n["anio"]:
            # La respuesta anuncia una cuenta: comprobamos ESA operación
            if (cuenta_cuadra(tipo, n, anclas, ops_locales) or
                    # «11 pozos menos que la otra ONG»: la otra cifra puede estar en
                    # otra fuente, pero (v2) solo en una que nombre a alguien de la
                    # respuesta, no en cualquiera
                    (tipo == "diferencia" and not anclas and cuenta_cuadra(tipo, n, set(), ops_nombradas))):
                clases.append("cuenta_correcta")
            else:
                clases.append("cuenta_falsa")
        elif esta_en(n, nums_fuente):
            clases.append("directo")
        elif anclas and cuenta_cuadra("calculo", n, anclas, ops_locales):
            clases.append("derivado_anclado")  # cuenta con cifras de la propia respuesta
        elif clasificar_numero(n, set(), der_locales) == "derivado":
            clases.append("derivado_local")
        elif not n["pct"] and any(_cerca(v, x, 0.005) for v in n["valores"] for x in partes_de_total):
            clases.append("derivado_local")  # v4: porcentaje de un total («5.520 pinos»)
        elif clasificar_numero(n, set(), der_globales) == "derivado":
            clases.append("derivado_casual")  # solo cuadra por casualidad: sospechoso
        else:
            clases.append("inexplicado")
    sospechosos = [n for n, c in zip(nums_resp, clases)
                   if c in ("inexplicado", "derivado_casual", "cuenta_falsa")]
    f["num_inexplicados"] = float(sum(1 for c in clases if c == "inexplicado"))
    f["num_derivado_casual"] = float(sum(1 for c in clases if c == "derivado_casual"))
    f["num_sospechosos_grandes"] = float(sum(1 for n in sospechosos if max(n["valores"]) > 10))
    f["num_derivado_local"] = float(sum(1 for c in clases if c == "derivado_local"))
    f["num_derivado_anclado"] = float(sum(1 for c in clases if c == "derivado_anclado"))
    f["num_cuenta_falsa"] = float(sum(1 for c in clases if c == "cuenta_falsa"))
    f["num_cuenta_correcta"] = float(sum(1 for c in clases if c == "cuenta_correcta"))
    f["num_hay_cifras"] = float(len(nums_resp) > 0)
    # v4: veredictos del verificador por separado, para que el modelo pueda
    # fiarse más de ellos que de las comprobaciones genéricas
    f["verif_cuenta_mal"] = float(min(2, sum(1 for e, _ in verificadas.values() if e == "mal")))
    f["verif_cuenta_ok"] = float(min(2, sum(1 for e, _ in verificadas.values() if e == "ok")))
    f["num_proporcion_falsa"] = float(proporciones_falsas(respuesta, nums_resp, valores_fuente))

    # --- 4. ¿De quién es la cifra? (atribución cruzada) ---
    # v2: distinguimos la entidad PRINCIPAL (la de la pregunta o la que sale en
    # muchas fuentes: «Lusel», «Valdemarina») de las SECUNDARIAS (las que solo
    # aparecen en alguna fuente: «Banca Ordial», «Algaida Baja», «Fuenvieja»).
    # Si una cifra solo está en fuentes que hablan de una entidad secundaria y
    # la respuesta no nombra a esa entidad, la cifra se está atribuyendo a otro.
    # Ej.: «Valdemarina tiene un 7,9 % de extranjeros», cuando el 7,9 % es de Algaida Baja.
    nombres_pregunta = {normalizar(p) for p in re.findall(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+", pregunta)}
    frecuencia = collections.Counter(n for nf in nombres_fuentes for n in nf)
    principales = nombres_pregunta | {n for n, k in frecuencia.items() if k >= 3}
    secundarios = [nf - principales for nf in nombres_fuentes]
    # Solo cuentan las fuentes que hablan de OTRA entidad comparable: lo dicen
    # («la comarca vecina», «competidora», «por su parte») o repiten el tipo de
    # entidad de la pregunta con otro nombre («Banco de Alimentos de Cuevalta»).
    # Una «Universidad de Lambrera» que firma un estudio no es otra entidad.
    otra_entidad = [bool(secundarios[i]) and (bool(RE_OTRA_ENTIDAD.search(normalizar(x)))
                                              or bool(nombres_fuentes[i] & nombres_pregunta))
                    for i, x in enumerate(fuentes)]
    ajenas = ajenas_fuerte = 0
    for n, c in zip(nums_resp, clases):
        if c != "directo" or n["anio"]:
            continue
        donde = [i for i, ns in enumerate(nums_por_fuente) if esta_en(n, ns)]
        if donde and all(otra_entidad[i] and not (secundarios[i] & nombres_resp) for i in donde):
            ajenas += 1
            # Más grave si además la respuesta nombra explícitamente a la principal
            if nombres_resp & principales:
                ajenas_fuerte += 1
    f["entidad_ajena"] = float(min(ajenas, 2))
    f["entidad_ajena_fuerte"] = float(min(ajenas_fuerte, 2))
    # Nombres propios de la respuesta que no salen en ninguna fuente ni en la pregunta
    todos_nombres = set().union(*nombres_fuentes) if nombres_fuentes else set()
    conocidos = todos_nombres | {normalizar(p) for p in re.findall(r"[A-Za-zÁÉÍÓÚÑÜáéíóúñü]+", pregunta)}
    f["nombre_inventado"] = float(len(nombres_resp - conocidos))

    # --- 4b. Fechas, plazos y horas (v2) ---
    # «nueve meses después», «tres años antes»: el plazo tiene que salir de
    # restar dos fechas de las fuentes. «a las 10:05»: la hora tiene que estar.
    falsos, correctos = comprobar_plazos(respuesta, " ".join(fuentes) + " " + pregunta)
    f["fecha_plazo_falso"] = float(min(falsos, 2))
    f["fecha_plazo_correcto"] = float(min(correctos, 2))
    horas_fuentes = set(RE_HORA.findall(" ".join(fuentes)))
    f["hora_inventada"] = float(any(h not in horas_fuentes for h in RE_HORA.findall(respuesta)))

    # --- 5. Matices perdidos, alcance ampliado y causas añadidas ---
    fuentes_norm = normalizar(" ".join(fuentes))
    # Texto de las fuentes en las que se apoya la respuesta: un «todos» o un
    # «porque» cuenta como añadido si no está en ESAS fuentes
    base_norm = normalizar(" ".join(fuentes[i] for i in sorted(fuentes_base)))
    raices_resp = raices_contenido(respuesta) | {
        tok for tok in re.findall(r"\d[\d.,]*", resp_norm)}
    perdidos_tramo = {k: 0 for k in MATICES}
    perdidos_fuente = {k: 0 for k in MATICES}
    huecos = []
    for i in citadas:
        fn = normalizar(fuentes[i])
        tramo, h = tramo_cubierto(fuentes[i], raices_resp)
        huecos.append(h)
        for k, pats in MATICES.items():
            perdidos_fuente[k] += perdidos(pats, fn, resp_norm)
            if tramo:
                perdidos_tramo[k] += perdidos(pats, tramo, resp_norm)
    # v2: los recuentos se limitan a 2. Con respuestas largas (el giro) hay más
    # oraciones y más fuentes citadas, y la suma crecería solo por la longitud.
    for k in MATICES:
        f["matiz_perdido_tramo_" + k] = float(min(perdidos_tramo[k], 2))
        f["matiz_perdido_fuente_" + k] = float(min(perdidos_fuente[k], 2))
    f["huecos_tramo_max"] = float(max(huecos) if huecos else 0)

    # v4: cobertura. (a) Qué parte de la fuente citada por el LLM se queda
    # fuera de la respuesta: si la respuesta usa una fuente larga y se deja la
    # mitad, puede estar quitando un matiz. (b) Qué parte de la respuesta no
    # sale de las fuentes en las que se apoya: si dice cosas que solo están en
    # otras fuentes, puede estar mezclando datos de entidades distintas.
    usadas = set(citadas)
    for x in fragmentos:
        rpf = x.get("respaldo_por_fuente") or []
        if rpf:
            usadas.add(max(range(len(rpf)), key=lambda k: rpf[k]))
    usadas = {i for i in usadas if 0 <= i < n_fuentes}
    raices_r = raices_contenido(respuesta)
    fuera = []
    for o in oraciones:
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            rf = raices_contenido(fuentes[i])
            fuera.append(len(rf - raices_r) / max(1, len(rf)))
    f["cobertura_citada_fuera"] = max(fuera) if fuera else 0.5
    raices_usadas = raices_contenido(" ".join(fuentes[i] for i in usadas) + " " + pregunta)
    f["resp_fuera_de_usadas"] = len(raices_r - raices_usadas) / max(1, len(raices_r))

    # v4: matiz perdido «por oración»: la oración no tiene NINGÚN matiz y la
    # fuente en la que se apoya sí. Más tolerante con las paráfrasis que contar
    # matiz a matiz: si la oración conserva alguna cautela, no salta.
    todos_matices = [p for pats in MATICES.values() for p in pats]
    pierde = 0.0
    for o in oraciones:
        on = normalizar(o.get("texto") or "")
        if any(re.search(p, on) for p in todos_matices):
            continue
        fuentes_o = set()
        i = (o.get("llm") or {}).get("fuente")
        if isinstance(i, int) and 0 <= i < n_fuentes:
            fuentes_o.add(i)
        sim = o.get("similitud_por_fuente") or []
        if sim:
            fuentes_o.add(max(range(len(sim)), key=lambda k: sim[k]))
        if any(any(re.search(p, normalizar(fuentes[j])) for p in todos_matices) for j in fuentes_o if j < n_fuentes):
            pierde = 1.0
    f["matiz_oracion_perdido"] = pierde
    f["anadido_universal"] = float(anadidos(UNIVERSALES, base_norm, resp_norm))
    f["anadido_causal"] = float(anadidos(CAUSALES, base_norm, resp_norm))
    # Los mismos, pero comparando con TODAS las fuentes (más estricto: si
    # ninguna fuente dice «porque», el porqué es inventado seguro)
    f["anadido_causal_todas"] = float(anadidos(CAUSALES, fuentes_norm, resp_norm))

    # --- 6. Palabras nuevas: con contenido y que no salen en fuentes ni pregunta ---
    raices_fuentes = raices_contenido(" ".join(fuentes) + " " + pregunta)
    nuevas = raices_contenido(respuesta) - raices_fuentes
    f["palabras_nuevas"] = float(len(nuevas))
    f["palabras_nuevas_frac"] = len(nuevas) / max(1, len(raices_contenido(respuesta)))

    # --- 7. Cruces útiles: el LLM da el visto bueno PERO hay otra sospecha ---
    llm_ok = 1.0 - max(f["llm_no_respaldada"], f["llm_contradicha"])
    f["llm_ok_y_num_sospechoso"] = llm_ok * min(1.0, f["num_sospechosos_grandes"])
    f["llm_ok_y_nli_bajo"] = llm_ok * float(f["nli_resp_fuente_citada_min"] < 0.3)
    # El LLM dice «contradicha» pero no hay nada más raro: todas las cifras
    # cuadran y el NLI no ve contradicción. Aquí el LLM suele equivocarse.
    cifras_ok = float(f["num_sospechosos_grandes"] == 0 and f["entidad_ajena"] == 0
                      and f["fecha_plazo_falso"] == 0)
    f["llm_contra_sin_apoyo"] = f["llm_contradicha"] * cifras_ok * (1.0 - f["nli_contr_todas_max_frag"])
    f["n_oraciones"] = float(len(oraciones))

    # --- 8. Las comprobaciones propias, en el régimen «el LLM dice que está bien» ---
    # Un modelo lineal da a cada sospecha el mismo peso siempre. Pero si el LLM
    # ya ha dicho «contradicha», una cifra rara añade poco; si ha dicho
    # «respaldada», esa cifra rara es justo lo que el LLM no ha visto. Duplicando
    # estas características multiplicadas por llm_ok, el modelo aprende un peso
    # distinto para cada situación.
    for nombre in list(f):
        if nombre.startswith(("num_", "verif_", "matiz_", "anadido_", "huecos_", "nombre_", "entidad_",
                              "fecha_", "hora_")):
            f["llmok_" + nombre] = llm_ok * f[nombre]
    return f


# ---------------------------------------------------------------------------
# Modelo: gradient boosting (árboles generados por entrenar.py)
# ---------------------------------------------------------------------------
# v3: en vez de una regresión logística (una suma de pesos), usamos «gradient
# boosting»: muchos árboles de decisión pequeños (3 niveles de preguntas tipo
# «¿llm_contradicha > 0,5?», «¿num_cuenta_correcta > 0,5?») que se suman, y
# cada uno corrige los errores de los anteriores. Así el modelo capta
# combinaciones («el LLM dice OK, PERO se ha perdido un "si se aprueba"») que
# una suma de pesos no puede representar.
#
# Los árboles se entrenan con scikit-learn en entrenar.py y se escriben aquí
# como listas de números, así que este fichero NO necesita scikit-learn:
# recorrerlos es Python puro.
#
# Cada árbol es (variable, umbral, hijo_izq, hijo_der, valor) por nodo; en las
# hojas hijo_izq = -1 y «valor» es lo que el árbol suma.

# >>> MODELO (generado automáticamente por entrenar.py; no editar a mano)
NOMBRES = ["llm_no_respaldada", "llm_contradicha", "llm_conf_min_respaldada", "nli_resp_todas_min_frag", "nli_contr_todas_max_frag", "nli_mejor_fuente_min_orac", "nli_contr_fuente_max_frag", "nli_resp_fuente_citada_min", "frag_respaldo_min", "frag_respaldo_2o", "frag_frac_bajo_05", "frag_frac_bajo_02", "frag_similitud_min", "num_inexplicados", "num_derivado_casual", "num_sospechosos_grandes", "num_derivado_local", "num_derivado_anclado", "num_cuenta_falsa", "num_cuenta_correcta", "num_hay_cifras", "verif_cuenta_mal", "verif_cuenta_ok", "entidad_ajena", "entidad_ajena_fuerte", "nombre_inventado", "fecha_plazo_falso", "fecha_plazo_correcto", "hora_inventada", "matiz_perdido_tramo_restrictivo", "matiz_perdido_fuente_restrictivo", "matiz_perdido_tramo_condicional", "matiz_perdido_fuente_condicional", "matiz_perdido_tramo_aproximado", "matiz_perdido_fuente_aproximado", "matiz_perdido_tramo_parcial", "matiz_perdido_fuente_parcial", "huecos_tramo_max", "matiz_oracion_perdido", "anadido_universal", "anadido_causal", "anadido_causal_todas", "palabras_nuevas_frac", "llm_ok_y_num_sospechoso", "llm_ok_y_nli_bajo", "llm_contra_sin_apoyo", "n_oraciones", "llmok_num_derivado_casual", "llmok_num_sospechosos_grandes", "llmok_num_derivado_local", "llmok_num_derivado_anclado", "llmok_num_cuenta_falsa", "llmok_num_cuenta_correcta", "llmok_num_hay_cifras", "llmok_verif_cuenta_mal", "llmok_verif_cuenta_ok", "llmok_entidad_ajena", "llmok_entidad_ajena_fuerte", "llmok_nombre_inventado", "llmok_fecha_plazo_falso", "llmok_fecha_plazo_correcto", "llmok_hora_inventada", "llmok_matiz_perdido_tramo_restrictivo", "llmok_matiz_perdido_fuente_restrictivo", "llmok_matiz_perdido_tramo_condicional", "llmok_matiz_perdido_fuente_condicional", "llmok_matiz_perdido_fuente_aproximado", "llmok_matiz_perdido_tramo_parcial", "llmok_matiz_perdido_fuente_parcial", "llmok_huecos_tramo_max", "llmok_matiz_oracion_perdido", "llmok_anadido_universal", "llmok_anadido_causal", "llmok_anadido_causal_todas"]
INICIAL = 0.089770
TASA = 0.05
ARBOLES = [[[53,69,3,-2,-2,-2,2,52,-2,-2,32,-2,-2],[0.5,0.5,0.03350000083446503,-2.0,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000367,0.379724,0.419923,1.855372,1.148966,-0.955268,-0.221218,-0.03633,0.124433,-1.946567,-0.355413,-1.577341,0.283742]],[[53,69,3,-2,-2,-2,2,52,-2,-2,18,-2,-2],[0.5,0.5,0.032500000670552254,-2.0,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004087,0.368397,0.40388,1.808458,1.161745,-0.693992,-0.207854,-0.031255,0.148748,-1.715737,-0.344105,-1.504107,0.733431]],[[53,5,20,-2,-2,-2,65,2,-2,-2,42,-2,-2],[0.5,0.9629999995231628,0.5,-2.0,-2.0,-2.0,0.5,0.9249999821186066,-2.0,-2.0,0.18333333730697632,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.007167,0.354468,0.384787,0.873165,1.642104,-0.921224,-0.207036,-0.257952,0.442741,-1.189743,0.236365,1.610517,-0.045273]],[[53,69,3,-2,-2,-2,2,52,-2,-2,18,-2,-2],[0.5,0.5,0.032500000670552254,-2.0,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.016569,0.320827,0.356681,1.631548,1.06863,-0.528219,-0.21295,-0.045759,0.02523,-1.563478,-0.33281,-1.428573,0.338019]],[[53,5,8,-2,-2,2,-2,-2,54,70,-2,-2,-2],[0.5,0.8190000057220459,0.09349999949336052,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.014087,0.301477,0.349599,1.605426,1.213767,-0.190191,0.189077,-1.69957,-0.198357,-0.227163,-1.135335,0.267625,1.493312]],[[53,5,5,-2,-2,1,-2,-2,2,52,-2,-2,18,-2,-2],[0.5,0.5289999842643738,0.051500000059604645,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003793,0.295516,0.348056,1.582277,1.191066,-0.005811,-0.745373,1.130693,-0.184284,-0.039488,0.04663,-1.465936,-0.290317,-1.294622,0.861923]],[[53,69,3,-2,-2,-2,2,38,-2,-2,11,-2,-2],[0.5,0.5,0.3295000046491623,-2.0,-2.0,-2.0,0.9750000238418579,0.5,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.008835,0.282654,0.323144,1.473019,0.857354,-1.153036,-0.162487,-0.035805,-0.427795,0.950583,-0.290971,-1.319461,0.218337]],[[53,5,3,-2,-2,42,-2,-2,64,21,-2,-2,-2],[0.5,0.5289999842643738,0.007500000298023224,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.011731,0.271786,0.311057,1.495479,1.113939,-0.000764,0.676,-1.329044,-0.178218,-0.208025,-0.963344,1.571289,1.341131]],[[53,69,5,-2,-2,-2,38,15,-2,-2,42,-2,-2],[0.5,0.5,0.4529999941587448,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.1819852963089943,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004461,0.251616,0.28913,1.388568,0.668421,-0.856202,-0.150331,-0.202868,-1.011379,0.571504,0.127858,1.649912,-0.709566]],[[53,69,5,-2,-2,-2,48,70,-2,-2,9,-2,-2],[0.5,0.5,0.08900000154972076,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9895000159740448,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000788,0.240357,0.270428,1.391675,0.871648,-0.695277,-0.143612,-0.181058,-0.954453,0.451731,0.185935,0.335863,1.708476]],[[53,69,5,-2,-2,-2,64,54,-2,-2,-2],[0.5,0.5,0.5289999842643738,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.000776,0.228032,0.254014,1.291154,0.405158,-0.734707,-0.136751,-0.162744,-0.781547,1.375233,1.469281]],[[53,5,39,-2,-2,-2,2,5,-2,-2,70,-2,-2],[0.5,0.8190000057220459,0.5,-2.0,-2.0,-2.0,0.9350000023841858,0.02350000012665987,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002974,0.220767,0.247779,1.256023,0.505119,-0.375043,-0.134655,0.277991,1.932439,0.639607,-0.169946,-0.942893,0.37659]],[[53,69,20,-2,-2,-2,2,52,-2,-2,65,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.0019,0.199911,0.230031,0.433881,1.200756,-0.859543,-0.117173,0.011519,0.26418,-1.430863,-0.216257,-1.040442,0.322639]],[[53,5,3,-2,-2,-2,11,42,-2,-2,4,-2,-2],[0.5,0.8190000057220459,0.032500000670552254,-2.0,-2.0,-2.0,0.22500000149011612,0.33908046782016754,-2.0,-2.0,0.2589999884366989,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002535,0.191268,0.217614,1.249337,0.64067,-0.39253,-0.119149,-0.173841,-0.542465,-1.489917,0.103352,1.354275,-0.19101]],[[53,69,3,-2,-2,-2,2,2,-2,-2,18,-2,-2],[0.5,0.5,0.032500000670552254,-2.0,-2.0,-2.0,0.9600000083446503,0.9350000023841858,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000494,0.179096,0.203085,1.244297,0.582386,-0.511455,-0.107682,0.02718,1.066715,-0.132475,-0.208186,-1.015009,0.586057]],[[53,69,3,-2,-2,-2,70,15,-2,-2,42,-2,-2],[0.5,0.5,0.03350000083446503,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.00528,0.170518,0.19636,1.200978,0.669232,-0.665908,-0.112845,-0.161723,-0.87945,0.748772,0.143925,1.541068,-1.206234]],[[53,5,3,-2,-2,-2,11,43,-2,-2,4,-2,-2],[0.5,0.965499997138977,0.032500000670552254,-2.0,-2.0,-2.0,0.22500000149011612,0.5,-2.0,-2.0,0.060499999672174454,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002578,0.169656,0.18799,1.23354,0.476707,-0.996104,-0.104477,-0.154301,-0.796548,0.633615,0.088505,1.594076,-0.11259]],[[53,5,10,-2,-2,1,-2,-2,2,19,-2,-2,18,-2,-2],[0.5,0.4529999941587448,0.8333333432674408,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004478,0.170012,0.20122,1.24088,0.858366,0.017554,-0.548636,1.01871,-0.09459,0.021615,0.264081,-1.212947,-0.182133,-0.931925,0.519594]],[[53,5,39,-2,-2,1,-2,-2,2,19,-2,-2,43,-2,-2],[0.5,0.4529999941587448,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9750000238418579,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001595,0.143291,0.182465,1.172093,0.2023,-0.050268,-0.859194,0.856142,-0.085607,0.00696,0.228708,-1.233087,-0.182501,-0.947381,0.330658]],[[53,5,5,-2,-2,-2,31,2,-2,-2,-2],[0.5,0.9629999995231628,0.09149999916553497,-2.0,-2.0,-2.0,0.5,0.9249999821186066,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.004126,0.151855,0.167778,1.166135,0.566848,-0.683669,-0.082727,-0.107226,0.804283,-0.60598,1.161317]],[[53,5,20,-2,-2,-2,38,54,-2,-2,42,-2,-2],[0.5,0.965499997138977,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004541,0.141909,0.157925,0.322602,1.062422,-1.010346,-0.080002,-0.125832,-0.680295,1.46165,0.159524,1.389247,-1.0296]],[[53,5,3,-2,-2,-2,2,9,-2,-2,43,-2,-2],[0.5,0.8765000104904175,0.0035000001080334187,-2.0,-2.0,-2.0,0.9350000023841858,0.9494999945163727,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001494,0.127561,0.146411,1.227786,0.624802,-0.625171,-0.07863,0.231649,-0.029968,1.826452,-0.105888,-0.626171,0.774238]],[[8,4,53,-2,-2,53,-2,-2,2,4,-2,-2,32,-2,-2],[0.09149999916553497,0.8700000047683716,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9350000023841858,0.17650000005960464,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001248,0.156176,0.210012,1.172675,1.303871,0.094329,1.121181,-0.359695,-0.07015,0.103208,1.072098,0.157621,-0.115417,-0.659504,0.86198]],[[2,69,3,-2,-2,-2,8,4,-2,-2,38,-2,-2],[0.9350000023841858,0.5,0.06550000235438347,-2.0,-2.0,-2.0,0.09149999916553497,0.8720000088214874,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002721,0.150301,0.128321,1.036061,0.272914,1.320847,-0.0549,0.11264,1.151246,-0.174758,-0.10925,-0.722289,0.444989]],[[53,69,1,-2,-2,-2,48,31,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9830000102519989,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.005423,0.114904,0.132912,0.615283,1.081409,-0.577229,-0.076624,-0.104882,-0.602453,1.078136,0.161744,0.014927,1.636478]],[[53,69,3,-2,-2,-2,31,43,-2,-2,-2],[0.5,0.5,0.9280000030994415,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.004485,0.122008,0.137232,1.002893,-0.186338,-0.544087,-0.063487,-0.083797,-0.532021,0.648102,1.144234]],[[11,1,42,-2,-2,8,-2,-2,17,65,-2,-2,-2],[0.20714285969734192,0.5,0.12916667014360428,-2.0,-2.0,0.9365000128746033,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.004258,-0.063751,-0.097317,0.082197,-0.848519,0.137465,0.719117,1.280372,0.114865,0.131823,0.700116,1.454136,-1.303975]],[[8,4,32,-2,-2,1,-2,-2,2,19,-2,-2,42,-2,-2],[0.1314999982714653,0.9259999990463257,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,0.5,-2.0,-2.0,0.11437908560037613,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.004207,0.115773,0.160339,0.860815,1.581646,0.049478,-0.330464,1.168296,-0.050524,0.038847,0.365436,-1.340062,-0.130932,-0.072961,-1.007077]],[[8,4,44,-2,-2,44,-2,-2,2,9,-2,-2,38,-2,-2],[0.09699999913573265,0.6700000166893005,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9350000023841858,0.945000022649765,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00608,0.129897,0.191149,1.124748,1.417811,0.071143,1.046317,-0.352152,-0.048671,0.104471,-0.041617,0.996864,-0.092095,-0.644763,0.48137]],[[8,11,-2,4,-2,-2,2,42,-2,-2,38,-2,-2],[0.09149999916553497,0.20714285969734192,-2.0,0.8700000047683716,-2.0,-2.0,0.9350000023841858,0.24038461595773697,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,-1,4,-1,-1,7,8,-1,-1,11,-1,-1],[6,3,-1,5,-1,-1,10,9,-1,-1,12,-1,-1],[-0.007882,0.114618,-0.114048,0.131739,1.152162,0.498,-0.064948,0.085303,0.693644,-0.254547,-0.10837,-0.719186,0.245607]],[[19,2,69,-2,-2,8,-2,-2,37,5,-2,-2,-2],[0.5,0.9350000023841858,0.5,-2.0,-2.0,0.0885000005364418,-2.0,-2.0,1.5,0.40150000154972076,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002746,0.027022,0.147099,0.937469,1.283737,-0.020906,0.735818,-0.348698,-0.257929,-0.230848,-1.348888,-1.039217,-1.757327]],[[19,8,4,-2,-2,42,-2,-2,8,-2,8,-2,-2],[0.5,0.09149999916553497,0.25949999690055847,-2.0,-2.0,0.12916667014360428,-2.0,-2.0,0.2420000061392784,-2.0,0.625,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003695,0.017077,0.122466,1.259393,0.566582,-0.032166,0.326233,-0.497651,-0.231988,-1.571188,-0.207882,-0.665298,-1.415698]],[[8,5,11,-2,-2,9,-2,-2,2,42,-2,-2,42,-2,-2],[0.09149999916553497,0.051500000059604645,0.4166666716337204,-2.0,-2.0,0.36650000512599945,-2.0,-2.0,0.9350000023841858,0.24038461595773697,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004847,0.103252,0.122837,0.503612,1.140177,0.029379,-1.128127,0.613132,-0.052753,0.077639,0.694634,-0.223225,-0.087157,0.034849,-0.747542]],[[19,2,16,-2,-2,8,-2,-2,4,42,-2,-2,-2],[0.5,0.9750000238418579,0.5,-2.0,-2.0,0.0949999988079071,-2.0,-2.0,0.6675000190734863,0.20714285969734192,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.01491,0.035463,0.093572,0.650708,-1.261061,-0.037993,0.823455,-0.515651,-0.225209,-0.187058,-1.316521,-0.781916,-1.480182]],[[11,38,1,-2,-2,42,-2,-2,65,5,-2,-2,-2],[0.20714285969734192,0.5,0.5,-2.0,-2.0,0.2361111119389534,-2.0,-2.0,0.5,0.02850000001490116,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003008,-0.0473,-0.078755,-0.600482,0.954561,0.118012,1.319837,-0.793094,0.098534,0.080268,1.066343,0.06983,1.451391]],[[2,69,7,-2,-2,-2,8,4,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.007500000298023224,-2.0,-2.0,-2.0,0.09149999916553497,0.9280000030994415,-2.0,-2.0,0.12916667014360428,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.001871,0.107992,0.085989,0.975329,0.294612,1.14644,-0.036378,0.067628,0.8244,-0.180462,-0.072747,0.174214,-0.7363]],[[5,19,69,-2,-2,-2,15,38,-2,-2,9,-2,-2],[0.0755000002682209,0.5,2.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.9819999933242798,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.000714,0.072467,0.085051,0.507832,1.135087,-1.058491,-0.055387,-0.077085,-0.604468,0.361533,0.143392,-0.140226,1.448349]],[[19,50,16,-2,-2,-2,44,-2,7,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,0.5,-2.0,0.0035000001080334187,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.001168,0.018849,0.030116,0.26855,-1.076027,-1.184431,-0.210544,-1.823356,-0.175701,-1.313494,-0.847196]],[[19,8,39,-2,-2,2,-2,-2,6,-2,3,-2,-2],[0.5,0.08900000154972076,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,0.7094999849796295,-2.0,0.016499999444931746,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.00444,0.022652,0.102681,0.907598,-0.288558,-0.014699,0.253235,-0.507878,-0.207126,-0.944857,-0.240021,-1.118292,-1.7041]],[[19,2,16,-2,-2,8,-2,-2,7,-2,2,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.04649999924004078,-2.0,-2.0,0.010999999940395355,-2.0,0.9750000238418579,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.007255,0.025615,0.086499,0.626607,-1.281286,-0.038713,0.995824,-0.469252,-0.184932,-0.882749,-0.222082,-1.593607,-1.215132]],[[2,2,7,-2,-2,10,-2,-2,42,70,-2,-2,8,-2,-2],[0.9350000023841858,0.8350000083446503,0.007500000298023224,-2.0,-2.0,0.2916666716337204,-2.0,-2.0,0.11324786394834518,0.5,-2.0,-2.0,0.05350000038743019,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00543,0.090679,0.061767,0.901747,0.095795,0.25517,1.36104,1.042511,-0.026405,0.061358,0.0091,1.597536,-0.060621,0.544936,-0.563737]],[[1,38,5,-2,-2,42,-2,-2,46,9,-2,-2,8,-2,-2],[0.5,0.5,0.022499999962747097,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,1.5,0.820499986410141,-2.0,-2.0,0.05700000002980232,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002117,-0.031494,-0.054873,0.352588,-0.491474,0.095234,1.161423,-0.724625,0.107967,0.090286,0.532372,1.109263,0.161957,1.111365,1.328637]],[[2,69,7,-2,-2,-2,21,32,-2,-2,-2],[0.9350000023841858,0.5,0.001500000071246177,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.008588,0.085455,0.067157,1.132281,0.342511,1.057723,-0.045071,-0.054795,-0.426521,0.519441,1.152793]],[[16,2,8,-2,-2,54,-2,-2,12,-2,-2],[0.5,0.9350000023841858,0.9359999895095825,-2.0,-2.0,0.5,-2.0,-2.0,0.5590000152587891,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002923,0.009509,0.104211,0.772089,1.197075,-0.025029,-0.217734,1.341274,-0.219574,-0.606002,-1.724408]],[[50,19,10,-2,-2,7,-2,-2,-2],[0.5,0.5,0.2678571492433548,-2.0,-2.0,0.010999999940395355,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.006482,0.016853,0.033802,-0.03484,0.619202,-0.161593,-0.677181,-1.42012,-1.339166]],[[19,16,2,-2,-2,9,-2,-2,6,-2,55,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.5600000023841858,-2.0,-2.0,0.7094999849796295,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.007654,0.007525,0.021611,0.84962,-0.050241,-0.217655,-1.805219,-0.839968,-0.181991,-0.888385,-0.214769,-1.7872,-1.182693]],[[8,5,6,-2,-2,-2,42,63,-2,-2,45,-2,-2],[0.09149999916553497,0.37699998915195465,0.5525000095367432,-2.0,-2.0,-2.0,0.31860901415348053,0.5,-2.0,-2.0,0.0024999999441206455,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.006546,0.073063,0.081145,1.080231,0.488771,-0.459929,-0.023812,0.007684,-0.074283,1.355285,-0.131124,-1.046451,1.207155]],[[19,5,50,-2,-2,2,-2,-2,22,6,-2,-2,12,-2,-2],[0.5,0.9715000092983246,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.5,0.7094999849796295,-2.0,-2.0,0.48099999129772186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002834,0.012123,0.030673,0.274971,-1.09631,-0.113387,-0.020214,-1.077451,-0.162988,-0.194413,-1.188874,-1.608051,-0.120651,-0.534277,-1.269035]],[[43,1,4,-2,-2,2,-2,-2,42,5,-2,-2,-2],[0.5,0.5,0.16350000351667404,-2.0,-2.0,0.4749999940395355,-2.0,-2.0,0.3541666716337204,0.20899999886751175,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.006536,-0.005108,-0.03059,0.128286,-0.586723,0.086819,0.810572,1.199596,0.192682,0.279936,0.905571,1.609935,-0.118719]],[[53,20,2,-2,-2,46,-2,-2,42,70,-2,-2,2,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,1.5,-2.0,-2.0,0.1715686321258545,0.5,-2.0,-2.0,0.9249999821186066,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005774,0.053664,-0.053829,0.309853,-1.408296,0.071888,0.570946,1.158986,-0.039711,0.019748,-0.134536,1.140697,-0.098088,0.590483,-0.675899]],[[54,5,17,-2,-2,31,-2,-2,-2],[0.5,0.045500000938773155,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.002883,-0.004902,0.05358,0.532697,-1.43881,-0.041177,-0.345338,1.31748,1.332314]],[[31,2,4,-2,-2,21,-2,-2,7,-2,-2],[0.5,0.9350000023841858,0.031499999575316906,-2.0,-2.0,0.5,-2.0,-2.0,0.039499999955296516,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.005989,-0.016978,0.060805,1.01546,0.380035,-0.047773,-0.379539,1.287552,0.188496,0.733477,1.481955]],[[19,16,2,-2,-2,12,-2,-2,4,6,-2,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.574999988079071,-2.0,-2.0,0.9634999930858612,0.8925000131130219,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000816,0.012914,0.022824,0.420645,-0.202327,-0.151804,-0.113442,-1.782098,-0.132798,-0.148694,-0.908124,-1.36178,-0.659593]],[[19,17,2,-2,-2,2,-2,-2,12,8,-2,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.6399999856948853,0.42249999940395355,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.002297,0.016085,0.02598,0.447749,-0.157075,-0.180888,-1.473859,-0.584644,-0.132068,-0.111184,-1.271861,-0.762654,-1.557009]],[[11,42,38,-2,-2,1,-2,-2,4,44,-2,-2,5,-2,-2],[0.20714285969734192,0.31643131375312805,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.20999999344348907,0.5,-2.0,-2.0,0.07699999958276749,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.006608,-0.024542,0.008265,-0.15424,0.869589,-0.123876,-1.037363,0.983682,0.067508,0.112708,0.959507,1.140115,0.036519,0.69092,-0.358804]],[[16,19,50,-2,-2,6,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.7094999849796295,-2.0,-2.0,0.9424999952316284,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000924,0.009392,0.022503,0.216296,-1.039679,-0.132862,-0.810076,-1.270834,-0.169278,-1.653843,0.177082]],[[42,69,38,-2,-2,4,-2,-2,1,8,-2,-2,37,-2,-2],[0.13076923787593842,1.5,0.5,-2.0,-2.0,0.029999999329447746,-2.0,-2.0,0.5,0.0625,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001367,0.048785,0.003291,-0.213356,1.373311,0.199541,1.950718,0.531457,-0.031187,-0.054788,0.276787,-0.553699,0.084232,0.340061,1.177431]],[[11,30,18,-2,-2,8,-2,-2,17,65,-2,-2,-2],[0.19090909510850906,0.5,0.5,-2.0,-2.0,0.8660000264644623,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000364,-0.030556,-0.048218,-0.392906,1.096891,0.161113,-0.074248,1.531015,0.053877,0.065065,0.507219,1.120482,-1.22287]],[[2,16,46,-2,-2,-2,1,0,-2,-2,-2],[0.9600000083446503,0.5,1.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.001109,0.041661,0.057543,0.509153,-0.755197,-1.162173,-0.040212,-0.056063,-0.532886,0.919945,1.20974]],[[19,1,38,-2,-2,3,-2,-2,5,8,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.030499999411404133,-2.0,-2.0,0.5864999890327454,0.35750000178813934,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001057,0.012505,-0.007313,-0.165954,0.513269,0.082666,0.923074,1.190603,-0.127034,-0.099655,-1.226568,-0.624267,-1.593842]],[[2,69,3,-2,-2,-2,38,15,-2,-2,42,-2,-2],[0.9350000023841858,0.5,0.25300000607967377,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.002274,0.061207,0.049536,0.837994,-0.101264,0.874296,-0.020833,-0.040537,-0.377176,0.555396,0.070342,1.051673,-0.703404]],[[48,1,4,-2,-2,5,-2,-2,9,-2,-2],[0.5,0.5,0.8039999902248383,-2.0,-2.0,0.1274999976158142,-2.0,-2.0,0.9524999856948853,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002335,-0.006851,-0.027672,-0.030201,-0.796869,0.067054,0.683697,1.143781,0.135715,-0.028856,1.339629]],[[54,64,53,-2,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[-0.000304,-0.007946,-0.016007,0.383341,-0.307461,1.127812,1.452992]],[[54,19,2,-2,-2,22,-2,-2,-2],[0.5,0.5,0.9750000238418579,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005036,-0.002337,0.007947,0.279182,-0.264537,-0.118896,-1.415663,-0.710228,1.427307]],[[19,5,17,-2,-2,2,-2,-2,12,6,-2,-2,-2],[0.5,0.8260000050067902,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,0.6399999856948853,0.8305000066757202,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.002772,0.00835,0.028886,0.290716,-0.910148,-0.05652,0.378872,-0.742967,-0.114729,-0.097511,-0.54909,-1.116442,-1.543747]],[[42,2,7,-2,-2,5,-2,-2,8,4,-2,-2,42,-2,-2],[0.02777777798473835,0.9950000047683716,0.8774999976158142,-2.0,-2.0,0.7744999825954437,-2.0,-2.0,0.07600000128149986,0.8865000009536743,-2.0,-2.0,0.33908046782016754,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005235,0.053227,0.092778,0.417995,1.784381,-0.088682,0.764531,-1.23921,-0.025009,0.034906,0.74281,-0.202153,-0.051785,-0.181234,-0.941924]],[[19,42,69,-2,-2,69,-2,-2,44,-2,5,-2,-2],[0.5,0.13076923787593842,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,0.09849999845027924,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002645,0.013916,0.06536,0.108821,1.188863,-0.015528,0.198673,-0.424314,-0.12518,-1.493797,-0.108098,-1.190147,-1.117265]],[[71,10,15,-2,-2,17,-2,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.004405,0.001546,-0.033516,-0.30711,0.818067,0.045414,0.544317,-1.605585,-1.3798]],[[16,42,69,-2,-2,8,-2,-2,9,-2,-2],[0.5,0.02777777798473835,1.5,-2.0,-2.0,0.09149999916553497,-2.0,-2.0,0.8634999990463257,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001698,0.010223,0.075527,0.289848,1.421992,-0.012631,0.494443,-0.279144,-0.158275,-1.735577,-0.315773]],[[63,10,2,-2,-2,64,-2,-2,8,-2,-2],[0.5,0.2678571492433548,0.9350000023841858,-2.0,-2.0,0.5,-2.0,-2.0,0.8989999890327454,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000393,-0.008356,-0.043288,0.387485,-0.436357,0.031825,0.214707,1.360558,0.143025,-0.261844,1.590398]],[[53,46,39,-2,-2,8,-2,-2,46,70,-2,-2,2,-2,-2],[0.5,1.5,0.5,-2.0,-2.0,0.06750000081956387,-2.0,-2.0,2.5,0.5,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.012419,0.032148,0.005349,0.273618,-2.047481,0.086962,1.07533,1.133528,-0.039664,-0.024182,-0.320706,0.709983,-0.202298,-1.628246,-0.839466]],[[42,48,17,-2,-2,8,-2,-2,5,-2,2,-2,-2],[0.4580419659614563,0.5,0.5,-2.0,-2.0,0.4255000054836273,-2.0,-2.0,0.01850000023841858,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.002453,0.013536,0.004969,0.104669,-1.100252,0.17389,0.138516,1.578945,-0.122119,0.101439,-0.170787,-1.559915,-0.977151]],[[48,5,70,-2,-2,42,-2,-2,9,3,-2,-2,-2],[0.5,0.04749999940395355,0.5,-2.0,-2.0,0.22361110895872116,-2.0,-2.0,0.9830000102519989,0.04050000011920929,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001328,-0.010495,0.031381,0.191208,0.875347,-0.037795,-0.077667,-0.711166,0.13356,0.034475,0.976037,-0.481545,1.414951]],[[42,69,38,-2,-2,4,-2,-2,5,12,-2,-2,1,-2,-2],[0.11324786394834518,1.5,0.5,-2.0,-2.0,0.0364999994635582,-2.0,-2.0,0.02949999924749136,0.5370000004768372,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001258,0.046754,0.008375,-0.167595,1.382695,0.197494,2.18151,0.285748,-0.01928,0.039918,0.732356,-0.224979,-0.052553,-0.481574,0.641163]],[[51,38,53,-2,-2,42,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.002282,-0.004648,-0.020428,0.401406,-0.367718,0.062751,0.999683,-0.707851,1.120103]],[[16,2,17,-2,-2,5,-2,-2,6,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.011500000022351742,-2.0,-2.0,0.9160000085830688,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002639,0.005723,0.040089,0.418355,-1.109372,-0.028095,0.727948,-0.392436,-0.147556,-1.459239,-0.274951]],[[64,48,53,-2,-2,12,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5375000238418579,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004586,-0.001897,-0.010186,0.334999,-0.247376,0.114842,1.099947,-0.257998,1.138494]],[[42,2,5,-2,-2,7,-2,-2,44,5,-2,-2,7,-2,-2],[0.029411764815449715,0.9850000143051147,0.21199999749660492,-2.0,-2.0,0.5690000057220459,-2.0,-2.0,0.5,0.5000000149011612,-2.0,-2.0,0.005499999970197678,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.008007,0.042332,0.09307,0.229231,1.194919,-0.069982,0.174952,-0.882797,-0.025932,0.007348,0.648351,-0.274671,-0.072039,-0.126385,-0.828772]],[[54,16,2,-2,-2,12,-2,-2,-2],[0.5,0.5,0.9600000083446503,-2.0,-2.0,0.5674999952316284,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.007584,0.001444,0.008359,0.325237,-0.208672,-0.126886,-0.197933,-1.717382,1.322607]],[[54,19,42,-2,-2,6,-2,-2,-2],[0.5,0.5,0.11324786394834518,-2.0,-2.0,0.7094999849796295,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005788,0.000143,0.009818,0.444924,-0.111856,-0.103883,-0.638823,-1.275158,1.238759]],[[19,42,69,-2,-2,69,-2,-2,3,-2,6,-2,-2],[0.5,0.13076923787593842,1.5,-2.0,-2.0,0.5,-2.0,-2.0,0.007000000216066837,-2.0,0.684499979019165,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.00036,0.008136,0.046681,0.077423,1.023276,-0.014281,0.202536,-0.423002,-0.084326,-0.389917,-0.103806,-0.612732,-1.403901]],[[16,19,69,-2,-2,8,-2,-2,3,-2,-2],[0.5,0.5,4.5,-2.0,-2.0,0.6675000190734863,-2.0,-2.0,0.0949999988079071,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001665,0.008703,0.017467,0.18527,-0.796584,-0.087109,-0.459319,-1.373613,-0.137135,-0.374202,-1.719073]],[[8,12,4,-2,-2,8,-2,-2,63,48,-2,-2,-2],[0.09149999916553497,0.47450000047683716,0.8745000064373016,-2.0,-2.0,0.022499999962747097,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.00035,0.04423,0.013602,0.699924,-0.607906,0.078654,0.581248,1.046323,-0.020582,-0.03152,-0.33705,0.791104,0.692736]],[[54,5,69,-2,-2,4,-2,-2,-2],[0.5,0.02850000001490116,2.5,-2.0,-2.0,0.2175000011920929,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.00458,-0.009827,0.033683,0.264732,0.744243,-0.030429,0.003552,-0.50712,1.091118]],[[71,38,16,-2,-2,42,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.22474747896194458,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001915,0.006653,-0.009603,-0.017031,-1.078812,0.080568,1.004839,-0.150731,-1.233821]],[[17,32,8,-2,-2,4,-2,-2,37,-2,-2],[0.5,0.5,0.0885000005364418,-2.0,-2.0,0.5185000002384186,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002248,0.004471,-0.004667,0.473074,-0.18118,0.094329,0.985026,-0.118561,-0.131607,-1.525205,-0.529978]],[[2,16,7,-2,-2,-2,1,0,-2,-2,-2],[0.9600000083446503,0.5,0.23349999636411667,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[0.001421,0.030803,0.042925,0.196122,0.862823,-1.056122,-0.029222,-0.04123,-0.466213,0.868931,1.155344]],[[54,11,42,-2,-2,17,-2,-2,-2],[0.5,0.20714285969734192,0.3060200661420822,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.006682,-0.011028,-0.030601,-0.062495,-0.905224,0.02815,0.440877,-1.108101,1.011139]],[[17,2,16,-2,-2,20,-2,-2,4,-2,-2],[0.5,0.9600000083446503,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5965000092983246,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003164,0.00388,0.029475,0.340555,-1.297112,-0.023221,-1.219232,-0.125658,-0.128697,-1.606832,-0.534472]],[[19,42,38,-2,-2,42,-2,-2,37,12,-2,-2,-2],[0.5,0.5857143104076385,0.5,-2.0,-2.0,0.7321428656578064,-2.0,-2.0,0.5,0.5225000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.001407,0.009742,0.01365,0.021016,0.525076,-0.183076,-1.35293,-1.809523,-0.09153,-0.055409,-0.261355,-1.160241,-1.430609]],[[39,42,19,-2,-2,42,-2,-2,5,-2,-2],[0.5,0.5657142996788025,0.5,-2.0,-2.0,0.7321428656578064,-2.0,-2.0,0.006500000134110451,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003966,0.002013,0.006131,0.11174,-1.038466,-0.177162,-1.350596,-1.913037,-0.117233,1.064144,-1.532521]],[[71,5,30,-2,-2,2,-2,-2,-2],[0.5,0.4519999921321869,0.5,-2.0,-2.0,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.000529,0.003995,0.023778,0.150291,1.312751,-0.037455,0.180155,-0.657477,-1.167224]],[[42,17,48,-2,-2,-2,8,3,-2,-2,53,-2,-2],[0.33908046782016754,0.5,0.5,-2.0,-2.0,-2.0,0.0625,0.007500000298023224,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.004711,0.01597,0.021621,0.124004,0.756237,-1.025282,-0.043824,0.037791,1.137322,-0.508293,-0.084298,0.300287,-1.01101]],[[42,69,38,-2,-2,4,-2,-2,8,4,-2,-2,1,-2,-2],[0.13076923787593842,1.5,0.5,-2.0,-2.0,0.06599999964237213,-2.0,-2.0,0.08750000223517418,0.8194999992847443,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.005907,0.033009,-2.1e-05,-0.19045,1.322178,0.141891,1.445266,0.176981,-0.029072,0.018718,0.719671,-0.383565,-0.055819,-0.518778,0.438765]],[[44,16,2,-2,-2,-2,6,-2,21,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,-2.0,0.02850000001490116,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.005282,0.02566,0.032248,0.69412,-0.083879,-1.380069,-0.025902,1.354542,-0.035613,-0.304633,0.886121]],[[42,9,5,-2,-2,63,-2,-2,5,11,-2,-2,1,-2,-2],[0.33974359929561615,0.945499986410141,0.8050000071525574,-2.0,-2.0,0.5,-2.0,-2.0,0.02350000012665987,0.0555555559694767,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005051,0.01797,-0.020463,-0.033329,-1.071594,0.045416,0.270125,1.267249,-0.046853,0.033075,-0.861589,0.926578,-0.094391,-1.110169,0.952239]],[[19,37,42,-2,-2,-2,3,-2,2,-2,-2],[0.5,4.5,0.5857143104076385,-2.0,-2.0,-2.0,0.006500000134110451,-2.0,0.9600000083446503,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.000707,0.008522,0.013464,0.148888,-1.579167,-0.82124,-0.082898,-0.327501,-0.10516,-1.340964,-1.223112]],[[17,11,42,-2,-2,10,-2,-2,9,-2,-2],[0.5,0.20714285969734192,0.26875001192092896,-2.0,-2.0,0.5833333432674408,-2.0,-2.0,0.9010000228881836,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004587,0.001777,-0.016008,0.06162,-0.62292,0.036637,0.76011,-0.232492,-0.131862,-1.559507,-0.612597]],[[19,42,69,-2,-2,7,-2,-2,37,42,-2,-2,69,-2,-2],[0.5,0.11437908560037613,1.5,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.5,0.3095238208770752,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000352,0.008311,0.0449,0.032595,1.146056,-0.008519,0.320648,-0.301032,-0.081634,-0.05354,-1.134208,-0.446213,-0.104024,-1.424836,-1.113421]],[[5,44,16,-2,-2,4,-2,-2,12,8,-2,-2,2,-2,-2],[0.9794999957084656,0.5,0.5,-2.0,-2.0,0.05999999865889549,-2.0,-2.0,0.7409999966621399,0.9799999892711639,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000376,0.007956,0.031277,0.482773,-1.3925,-0.020928,0.36951,-0.430536,-0.063842,-0.107073,-1.351299,-0.643443,0.020141,0.52484,-1.13204]],[[42,19,9,-2,-2,6,-2,-2,3,2,-2,-2,8,-2,-2],[0.4580419659614563,0.5,0.945499986410141,-2.0,-2.0,0.7094999849796295,-2.0,-2.0,0.013999999966472387,0.875,-2.0,-2.0,0.33650000393390656,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002049,0.004798,0.011165,-0.133842,0.235274,-0.075861,-0.263857,-1.309113,-0.077721,0.003159,0.86974,-0.277855,-0.125931,-1.646092,-1.063719]],[[71,10,63,-2,-2,17,-2,-2,-2],[0.5,0.2678571492433548,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002245,0.002178,-0.024095,-0.265247,0.633388,0.035354,0.47751,-0.934116,-1.436219]],[[8,3,39,-2,-2,6,-2,-2,42,63,-2,-2,12,-2,-2],[0.0925000011920929,0.02049999963492155,0.5,-2.0,-2.0,0.948500007390976,-2.0,-2.0,0.22649572789669037,0.5,-2.0,-2.0,0.6024999916553497,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.001461,0.035427,0.02172,0.472825,-1.194045,0.115537,0.532875,1.482171,-0.014171,0.0056,-0.089006,0.997628,-0.049938,-0.127197,-1.264192]],[[44,2,4,-2,-2,5,-2,-2,4,4,-2,-2,9,-2,-2],[0.5,0.9950000047683716,0.4389999955892563,-2.0,-2.0,0.5449999868869781,-2.0,-2.0,0.007500000298023224,0.0035000001080334187,-2.0,-2.0,0.11149999871850014,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000512,0.022526,0.042053,0.764808,-0.005517,-0.024231,1.074176,-0.923273,-0.031314,0.088009,0.175002,1.187344,-0.050258,-1.392651,-0.281418]],[[10,1,2,-2,-2,12,-2,-2,9,12,-2,-2,5,-2,-2],[0.2678571492433548,0.5,0.9350000023841858,-2.0,-2.0,0.6585000157356262,-2.0,-2.0,0.38199999928474426,0.7354999780654907,-2.0,-2.0,0.36249999701976776,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004534,-0.02627,-0.037574,0.369007,-0.373713,0.064071,1.072332,-0.001349,0.022896,-0.029538,-0.917835,0.775002,0.050292,0.792791,-0.151802]],[[17,42,70,-2,-2,70,-2,-2,37,-2,-2],[0.5,0.2752525359392166,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004457,0.009913,0.026156,0.108873,0.76701,-0.023201,-0.105612,-1.085335,-0.100233,-1.42237,-0.310963]],[[42,2,69,-2,-2,5,-2,-2,2,53,-2,-2,1,-2,-2],[0.02777777798473835,0.9850000143051147,0.5,-2.0,-2.0,0.9855000078678131,-2.0,-2.0,0.9350000023841858,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.003249,0.035121,0.076566,0.419534,1.007563,-0.058711,-0.285301,-1.180234,-0.015815,0.034616,0.209577,0.749613,-0.03143,-0.340685,1.12038]],[[71,48,10,-2,-2,8,-2,-2,-2],[0.5,0.5,0.2361111119389534,-2.0,-2.0,0.45000000298023224,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001706,0.005706,-0.000298,-0.207632,0.367171,0.102949,-0.092024,1.379463,-1.032466]],[[19,2,7,-2,-2,53,-2,-2,37,6,-2,-2,69,-2,-2],[0.5,0.9750000238418579,0.8925000131130219,-2.0,-2.0,0.5,-2.0,-2.0,0.5,0.926499992609024,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005006,0.012926,0.035078,0.187641,1.417763,-0.015008,0.594891,-0.336822,-0.083104,-0.060863,-0.762546,-1.140203,-0.105756,-1.389303,-1.116737]],[[69,69,12,-2,-2,8,-2,-2,-2],[4.5,2.5,0.2575000077486038,-2.0,-2.0,0.886000007390976,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.003605,0.000352,-0.00731,0.849654,-0.139862,0.069712,-0.202337,1.260828,-0.777762]],[[44,16,2,-2,-2,-2,5,69,-2,-2,9,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,-2.0,0.6425000131130219,2.5,-2.0,-2.0,0.9905000030994415,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.004962,0.013109,0.01855,0.578204,-0.281195,-1.494251,-0.032097,-0.017884,-0.279827,0.553323,-0.151184,-1.548404,0.111311]],[[71,16,42,-2,-2,9,-2,-2,-2],[0.5,0.5,0.5857143104076385,-2.0,-2.0,0.9784999787807465,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001814,0.001616,0.006611,0.095505,-1.533472,-0.088533,-1.175927,0.415543,-1.010765]],[[5,44,6,-2,-2,4,-2,-2,12,2,-2,-2,2,-2,-2],[0.9794999957084656,0.5,0.9784999787807465,-2.0,-2.0,0.030499999411404133,-2.0,-2.0,0.5264999866485596,0.9950000047683716,-2.0,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00691,0.000901,0.027061,0.052876,1.035718,-0.031189,0.314127,-0.403928,-0.074723,-0.122089,-1.309595,-1.117998,-0.029862,0.528371,-0.830237]],[[56,16,42,-2,-2,9,-2,-2,6,-2,-2],[0.5,0.5,0.22361110895872116,-2.0,-2.0,0.8634999990463257,-2.0,-2.0,0.9695000052452087,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.004623,-4.1e-05,0.005305,0.215968,-0.209298,-0.088183,-1.260978,-0.016159,-0.096301,0.062318,-1.111092]],[[16,54,4,-2,-2,-2,4,-2,-2],[0.5,0.5,0.05849999934434891,-2.0,-2.0,-2.0,0.8650000095367432,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,-1],[6,5,4,-1,-1,-1,8,-1,-1],[1e-06,0.005259,0.001264,0.266374,-0.174457,0.930684,-0.100561,-1.444342,-0.126212]],[[42,69,6,-2,-2,4,-2,-2,20,12,-2,-2,53,-2,-2],[0.13076923787593842,1.5,0.3095000088214874,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.5199999809265137,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003345,0.03181,-0.000104,-0.83105,0.273619,0.130653,1.398729,0.11161,-0.012582,-0.126103,-0.032217,-1.640095,-0.005725,0.487589,-0.210441]],[[64,54,9,-2,-2,-2,-2],[0.5,0.5,0.945499986410141,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.008935,0.003877,-0.000185,-0.306038,0.18277,1.035328,1.185475]],[[42,69,38,-2,-2,4,-2,-2,66,5,-2,-2,-2],[0.11324786394834518,1.5,0.5,-2.0,-2.0,0.07649999856948853,-2.0,-2.0,0.5,0.02850000001490116,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000789,0.032436,0.006492,-0.135452,1.289387,0.133557,1.681509,0.239518,-0.013701,-0.007993,0.332525,-0.245441,-1.175416]],[[42,15,4,-2,-2,8,-2,-2,5,2,-2,-2,8,-2,-2],[0.4580419659614563,0.5,0.061000000685453415,-2.0,-2.0,0.45250000059604645,-2.0,-2.0,0.019499999471008778,0.9249999821186066,-2.0,-2.0,0.20600000023841858,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.000327,0.005267,-0.001567,0.218138,-0.190635,0.070265,0.190847,1.145384,-0.06127,0.019756,1.356228,-0.168388,-0.097163,-1.369383,-0.820899]],[[10,63,44,-2,-2,-2,9,4,-2,-2,69,-2,-2],[0.2678571492433548,0.5,0.5,-2.0,-2.0,-2.0,0.9255000054836273,0.1525000035762787,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002309,-0.020909,-0.03091,-0.027381,-0.514772,0.713566,0.02224,-0.013603,0.647196,-0.488851,0.071109,0.624495,1.388675]],[[69,70,6,-2,-2,42,-2,-2,-2],[4.5,0.5,0.9535000026226044,-2.0,-2.0,0.25357143580913544,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002291,0.000924,-0.005834,-0.258139,0.276606,0.062205,0.860586,-0.863156,-0.702797]],[[43,11,63,-2,-2,3,-2,-2,9,6,-2,-2,-2],[0.5,0.22500000149011612,0.5,-2.0,-2.0,0.02049999963492155,-2.0,-2.0,0.9830000102519989,0.875,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00622,0.000116,-0.017532,-0.246964,0.711874,0.034262,0.300143,0.816545,0.099002,0.017896,-0.54299,0.632947,1.408161]],[[3,44,6,-2,-2,6,-2,-2,7,-2,-2],[0.99549999833107,0.5,0.9784999787807465,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.9560000002384186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.002729,0.000436,0.016994,-0.024604,0.917387,-0.022931,1.239666,-0.243609,-0.163719,-1.443821,-1.284509]],[[67,5,6,-2,-2,4,-2,-2,-2],[0.5,0.6755000054836273,0.02850000001490116,-2.0,-2.0,0.8935000002384186,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001888,-0.001805,0.009315,1.259258,0.022043,-0.030701,-0.112623,-1.428951,0.948406]],[[38,44,6,-2,-2,6,-2,-2,42,9,-2,-2,12,-2,-2],[0.5,0.5,0.9794999957084656,-2.0,-2.0,0.02850000001490116,-2.0,-2.0,0.2361111119389534,0.9914999902248383,-2.0,-2.0,0.4919999986886978,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004508,-0.015087,0.003177,-0.232806,0.871069,-0.04181,0.991141,-0.362855,0.043562,0.096257,0.640912,1.352856,-0.041899,0.119811,-1.22037]],[[71,5,17,-2,-2,4,-2,-2,-2],[0.5,0.5785000026226044,0.5,-2.0,-2.0,0.882999986410141,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.005171,-0.001684,0.01398,0.202563,-1.142766,-0.040678,-0.208125,-1.246823,-1.210864]],[[2,2,39,-2,-2,9,-2,-2,37,44,-2,-2,6,-2,-2],[0.9350000023841858,0.8350000083446503,0.5,-2.0,-2.0,0.9784999787807465,-2.0,-2.0,2.5,0.5,-2.0,-2.0,0.976500004529953,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000625,0.032808,0.012128,0.369209,-1.816306,0.134855,0.140367,1.373992,-0.011566,-0.022995,0.027928,-0.380191,0.037238,0.629154,-0.495823]],[[42,2,7,-2,-2,7,-2,-2,42,7,-2,-2,42,-2,-2],[0.02777777798473835,0.9950000047683716,0.9165000021457672,-2.0,-2.0,0.6820000112056732,-2.0,-2.0,0.10435571894049644,0.005499999970197678,-2.0,-2.0,0.612500011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.000404,0.032604,0.058876,0.273751,1.576501,-0.067504,0.298658,-1.146625,-0.010843,-0.118468,-0.420396,-1.086967,-0.005231,-0.020208,-1.594567]],[[54,19,3,-2,-2,6,-2,-2,-2],[0.5,0.5,0.9964999854564667,-2.0,-2.0,0.6404999792575836,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.005104,0.001876,0.007952,0.100206,-1.388088,-0.061018,-0.325524,-1.124856,0.902603]],[[38,11,51,-2,-2,10,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.22500000149011612,0.5,-2.0,-2.0,0.5833333432674408,-2.0,-2.0,0.22474747896194458,0.9379999935626984,-2.0,-2.0,0.4919999986886978,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003604,-0.005149,-0.018703,-0.226558,0.819391,0.023866,0.586323,-0.234818,0.047734,0.092864,1.278348,-0.032061,-0.029551,0.426065,-1.225171]],[[5,44,16,-2,-2,6,-2,-2,42,-2,6,-2,-2],[0.992499977350235,0.5,0.5,-2.0,-2.0,0.009999999776482582,-2.0,-2.0,0.1339285746216774,-2.0,0.26900000870227814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.004145,0.009026,0.026776,0.450579,-1.243273,-0.013406,1.597984,-0.155249,-0.068864,-0.080392,-0.103174,-0.868011,-1.227651]],[[69,16,2,-2,-2,9,-2,-2,-2],[4.5,0.5,0.9350000023841858,-2.0,-2.0,0.9595000147819519,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001377,0.004615,0.008972,0.715529,-0.034024,-0.065417,-1.00585,0.726044,-0.915466]],[[32,48,1,-2,-2,3,-2,-2,6,6,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.02500000037252903,-2.0,-2.0,0.9694999754428864,0.265500009059906,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000621,-0.006383,-0.012936,-0.250328,0.771948,0.080194,1.054698,0.014045,0.076328,0.105518,0.873313,1.121888,-0.123185]],[[17,42,51,-2,-2,5,-2,-2,37,-2,-2],[0.5,0.42217741906642914,0.5,-2.0,-2.0,0.01899999938905239,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001798,0.006369,0.01366,0.087543,0.882447,-0.047616,0.630545,-0.979611,-0.097228,-1.479924,-0.315058]],[[43,56,5,-2,-2,4,-2,-2,7,-2,-2],[0.5,0.5,0.9895000159740448,-2.0,-2.0,0.632999986410141,-2.0,-2.0,0.024500000290572643,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.003558,-0.001038,0.003054,0.094266,-0.527415,-0.09039,-0.071575,-1.307708,0.07892,0.959782,-0.108284]],[[44,3,8,-2,-2,2,-2,-2,6,-2,6,-2,-2],[0.5,0.9905000030994415,0.9794999957084656,-2.0,-2.0,0.9850000143051147,-2.0,-2.0,0.02850000001490116,-2.0,0.24050000309944153,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003231,0.011257,0.017294,0.08503,0.699661,-0.117577,-1.423343,-1.125168,-0.025153,1.033064,-0.033637,-1.05697,-0.152004]],[[7,3,4,-2,-2,8,-2,-2,4,6,-2,-2,1,-2,-2],[0.006500000134110451,0.7254999876022339,0.05849999934434891,-2.0,-2.0,0.9794999957084656,-2.0,-2.0,0.21050000190734863,0.9945000112056732,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.004427,0.021421,0.030799,1.031714,0.196741,-0.043027,-0.918743,0.449256,-0.021056,0.00338,-0.071952,1.327148,-0.0551,-0.656731,0.138489]],[[38,3,9,-2,-2,6,-2,-2,42,6,-2,-2,12,-2,-2],[0.5,0.001500000071246177,0.9914999902248383,-2.0,-2.0,0.9594999849796295,-2.0,-2.0,0.2361111119389534,0.9839999973773956,-2.0,-2.0,0.4714999943971634,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001047,-0.010144,0.028118,0.085385,1.64547,-0.019485,-0.32134,0.095761,0.041926,0.080131,1.059902,-0.102829,-0.020524,0.558035,-1.126201]],[[10,4,37,-2,-2,4,-2,-2,9,4,-2,-2,3,-2,-2],[0.2678571492433548,0.004500000039115548,0.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.9835000038146973,0.17549999803304672,-2.0,-2.0,0.02049999963492155,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.00452,-0.021673,-0.08561,-1.177445,-0.48458,-0.011871,1.059977,-0.169508,0.01641,-0.002805,0.603658,-0.370825,0.072053,0.396844,1.349937]],[[48,10,63,-2,-2,17,-2,-2,6,-2,8,-2,-2],[0.5,0.2361111119389534,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.7529999911785126,-2.0,0.4479999989271164,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.001812,-0.003818,-0.022597,-0.299617,0.60952,0.019019,0.352626,-1.0312,0.080685,-0.427817,0.14902,0.484421,1.229525]],[[17,42,2,-2,-2,66,-2,-2,37,-2,-2],[0.5,0.02777777798473835,0.9950000047683716,-2.0,-2.0,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.003675,0.000484,0.034316,0.586098,-0.530884,-0.011199,-0.06128,-1.114358,-0.08386,-1.343142,-0.209381]],[[51,16,44,-2,-2,6,-2,-2,8,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.8865000009536743,-2.0,-2.0,0.35750000178813934,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.002146,-0.002026,0.003243,0.253172,-0.167032,-0.10055,-1.537647,-0.440014,0.087947,0.145589,1.024237]],[[9,9,53,-2,-2,48,-2,-2,2,12,-2,-2,10,-2,-2],[0.945499986410141,0.05849999934434891,0.5,-2.0,-2.0,0.5,-2.0,-2.0,0.9600000083446503,0.6565000116825104,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.00034,-0.01996,0.048246,1.037235,1.241398,-0.033678,-0.274149,-0.951022,0.016249,0.050646,0.19194,0.929546,-0.021203,-0.384144,0.675175]],[[19,2,7,-2,-2,12,-2,-2,4,6,-2,-2,-2],[0.5,0.9750000238418579,0.18699999898672104,-2.0,-2.0,0.6089999973773956,-2.0,-2.0,0.9664999842643738,0.684499979019165,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.0041,0.001229,0.019714,0.036187,0.621617,-0.021636,0.008503,-0.764328,-0.061198,-0.081677,-1.07045,-1.292307,-0.097554]],[[64,1,6,-2,-2,3,-2,-2,-2],[0.5,0.5,0.9964999854564667,-2.0,-2.0,0.25300000607967377,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.00327,-0.006292,-0.01514,-0.075887,-0.749539,0.026709,1.055981,-0.436336,0.623595]],[[17,42,51,-2,-2,5,-2,-2,42,-2,-2],[0.5,0.33908046782016754,0.5,-2.0,-2.0,0.004500000039115548,-2.0,-2.0,0.2678571492433548,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000189,0.003834,0.014493,0.090114,0.886499,-0.038518,1.258296,-0.837162,-0.078742,-1.426381,0.475897]],[[10,63,44,-2,-2,-2,9,4,-2,-2,8,-2,-2],[0.2678571492433548,0.5,0.5,-2.0,-2.0,-2.0,0.9835000038146973,0.16099999845027924,-2.0,-2.0,0.06149999983608723,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.002769,-0.019638,-0.027595,-0.044706,-0.487122,0.551631,0.018092,-0.000901,0.708183,-0.402854,0.077835,0.310807,1.336238]],[[2,69,17,-2,-2,6,-2,-2,1,4,-2,-2,-2],[0.9600000083446503,2.5,0.5,-2.0,-2.0,0.9290000200271606,-2.0,-2.0,0.5,0.07649999856948853,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000503,0.020444,0.007787,0.176303,-1.354717,0.118029,1.292548,-0.356167,-0.020528,-0.028377,0.027503,-0.629502,1.098205]],[[42,17,2,-2,-2,2,-2,-2,5,-2,8,-2,-2],[0.4752941131591797,0.5,0.9750000238418579,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.021499999798834324,-2.0,0.2409999966621399,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.000662,0.005223,0.00964,0.247943,-0.117932,-0.072534,-1.35024,0.064713,-0.056586,0.304203,-0.089817,-1.521073,-0.986973]],[[39,9,69,-2,-2,43,-2,-2,46,-2,-2],[0.5,0.945499986410141,1.5,-2.0,-2.0,0.5,-2.0,-2.0,1.5,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.000353,0.004067,-0.015959,0.032749,-0.851401,0.018541,0.091871,1.119583,-0.089192,-1.758655,-0.018612]],[[11,8,5,-2,-2,63,-2,-2,42,3,-2,-2,-2],[0.20714285969734192,0.2865000069141388,0.022499999962747097,-2.0,-2.0,0.5,-2.0,-2.0,0.5584415793418884,0.02049999963492155,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.000484,-0.012798,-0.070088,0.034331,-1.205142,-0.004101,-0.105112,0.602426,0.024276,0.027565,0.304375,0.705604,-1.222602]],[[5,9,19,-2,-2,5,-2,-2,42,-2,6,-2,-2],[0.9895000159740448,0.9975000023841858,0.5,-2.0,-2.0,0.8959999978542328,-2.0,-2.0,0.04545454680919647,-2.0,0.26900000870227814,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[0.003626,0.009237,0.003612,0.101322,-0.899339,0.093838,0.490043,1.809204,-0.054983,0.328885,-0.086318,-0.182039,-1.271377]],[[35,42,20,-2,-2,-2,-2],[0.5,0.7321428656578064,0.5,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.00258,-0.000543,0.000749,-0.483114,0.05189,-1.673033,0.897215]],[[19,20,2,-2,-2,6,-2,-2,42,2,-2,-2,-2],[0.5,0.5,0.9850000143051147,-2.0,-2.0,0.24400000274181366,-2.0,-2.0,0.3166666775941849,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.003613,0.008873,-0.061812,-0.020739,-1.138919,0.013874,-0.292485,0.245865,-0.056024,-0.071795,-1.24701,-1.059191,-0.632721]],[[12,12,12,-2,-2,-2,12,44,-2,-2,35,-2,-2],[0.2580000013113022,0.24149999767541885,0.2174999937415123,-2.0,-2.0,-2.0,0.3110000044107437,0.5,-2.0,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000144,0.052707,0.030171,0.828195,-0.382796,1.385931,-0.006291,-0.078341,-0.139037,-1.136305,-0.000586,-0.040753,0.996103]],[[3,3,7,-2,-2,-2,3,-2,3,-2,-2],[0.9504999816417694,0.9415000081062317,0.005499999970197678,-2.0,-2.0,-2.0,0.9665000140666962,-2.0,0.9805000126361847,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.003779,0.010136,0.008053,0.387204,-0.075688,1.374264,-0.042743,-1.429381,-0.018892,0.474362,-0.431775]],[[32,5,69,-2,-2,42,-2,-2,12,-2,3,-2,-2],[0.5,0.02949999924749136,2.5,-2.0,-2.0,0.3550420254468918,-2.0,-2.0,0.4440000057220459,-2.0,0.01699999999254942,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004572,-0.009809,0.013519,-0.032628,1.202883,-0.020693,-0.10633,-0.832735,0.051312,1.048953,0.016384,-0.728756,0.604045]],[[5,3,3,-2,-2,5,-2,-2,4,2,-2,-2,-2],[0.6755000054836273,0.68299999833107,0.5590000152587891,-2.0,-2.0,0.00849999999627471,-2.0,-2.0,0.882999986410141,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.005963,0.004406,0.011982,0.098778,1.376891,-0.042804,-1.269805,-0.183705,-0.034313,-0.018652,0.316778,-0.571013,-1.431405]],[[16,42,44,-2,-2,2,-2,-2,6,-2,-2],[0.5,0.612500011920929,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.8865000009536743,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.00753,0.011176,0.013272,0.370832,-0.064479,-0.124582,-1.721763,-1.190629,-0.063555,-1.228625,0.257251]],[[51,19,42,-2,-2,42,-2,-2,-2],[0.5,0.5,0.5857143104076385,-2.0,-2.0,0.3166666775941849,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00568,0.002294,0.00723,0.099602,-1.359002,-0.05385,-1.182681,-0.62544,0.600572]],[[35,56,19,-2,-2,69,-2,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001307,-0.00157,0.002121,0.069193,-0.86527,-0.076983,-0.117656,-0.959375,0.979777]],[[11,44,37,-2,-2,3,-2,-2,7,8,-2,-2,5,-2,-2],[0.4166666716337204,0.5,0.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.005499999970197678,0.022499999962747097,-2.0,-2.0,0.05949999950826168,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-5e-05,-0.009787,0.010026,-0.31853,0.360636,-0.037565,-1.041392,-0.074179,0.0281,0.061892,0.499352,1.141215,-0.003819,0.6178,-0.404058]],[[17,4,63,-2,-2,7,-2,-2,12,-2,-2],[0.5,0.35899999737739563,0.5,-2.0,-2.0,0.007500000298023224,-2.0,-2.0,0.4960000216960907,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.001073,0.002963,0.016941,0.099619,0.901533,-0.016558,0.345461,-0.655095,-0.088521,-0.194138,-1.444166]],[[71,70,6,-2,-2,6,-2,-2,-2],[0.5,0.5,0.9535000026226044,-2.0,-2.0,0.8205000162124634,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.000775,0.001891,-0.003782,-0.245471,0.320627,0.050141,1.07747,-0.29228,-1.01267]],[[64,5,4,-2,-2,6,-2,-2,-2],[0.5,0.9835000038146973,0.18400000035762787,-2.0,-2.0,0.22599999606609344,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.002834,-0.006139,0.000397,0.211467,-0.189429,-0.064149,0.0241,-0.962878,0.920434]],[[4,69,4,-2,-2,-2,5,4,-2,-2,12,-2,-2],[0.0345000009983778,2.5,0.02850000001490116,-2.0,-2.0,-2.0,0.02850000001490116,0.22450000047683716,-2.0,-2.0,0.2580000013113022,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000183,0.025882,0.008685,-0.00351,1.048855,0.962956,-0.011498,0.021247,-0.442289,0.618244,-0.028283,0.614863,-0.362907]],[[5,7,6,-2,-2,2,-2,-2,3,-2,2,-2,-2],[0.9895000159740448,0.9335000216960907,0.014500000048428774,-2.0,-2.0,0.9950000047683716,-2.0,-2.0,0.4675000160932541,-2.0,0.9850000143051147,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.003994,0.001378,-0.004942,1.060165,-0.101202,0.05867,1.188758,-0.407727,-0.069641,-1.153474,-0.042109,0.094202,-1.103077]],[[54,12,8,-2,-2,12,-2,-2,-2],[0.5,0.2580000013113022,0.38449999690055847,-2.0,-2.0,0.3279999941587448,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001545,-0.001514,0.049645,0.359603,1.175577,-0.007673,-0.798343,-0.013702,1.069171]],[[44,16,6,-2,-2,-2,9,12,-2,-2,10,-2,-2],[0.5,0.5,0.3265000134706497,-2.0,-2.0,-2.0,0.9255000054836273,0.8194999992847443,-2.0,-2.0,0.4166666716337204,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[0.000619,0.015038,0.019462,-0.282374,0.503946,-1.286431,-0.020408,-0.057527,-0.624921,0.567894,0.002647,-0.206288,1.08025]],[[2,69,39,-2,-2,-2,31,1,-2,-2,-2],[0.9350000023841858,0.5,0.5,-2.0,-2.0,-2.0,0.5,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,-1],[-0.002033,0.026961,0.01555,0.42426,-1.548524,0.825977,-0.012709,-0.017902,-0.232736,1.13789,0.730874]],[[51,16,69,-2,-2,6,-2,-2,-2],[0.5,0.5,4.5,-2.0,-2.0,0.9535000026226044,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.00363,-0.000255,0.003729,0.075928,-0.704541,-0.073225,-1.563579,0.27662,0.682896]],[[44,2,8,-2,-2,46,-2,-2,6,-2,7,-2,-2],[0.5,0.9950000047683716,0.9574999809265137,-2.0,-2.0,1.5,-2.0,-2.0,0.009499999694526196,-2.0,0.014500000048428774,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004173,0.009466,0.024204,0.166156,0.627017,-0.025088,-1.175544,0.702054,-0.02493,1.363414,-0.030038,-0.058808,-0.673333]],[[4,4,6,-2,-2,8,-2,-2,5,8,-2,-2,4,-2,-2],[0.351500004529953,0.24499999731779099,0.9945000112056732,-2.0,-2.0,0.597000002861023,-2.0,-2.0,0.6545000076293945,0.45000000298023224,-2.0,-2.0,0.8935000002384186,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.003429,0.016774,0.008794,0.004941,0.944518,0.099939,1.587033,0.371168,-0.014453,-0.001597,-0.278305,0.558486,-0.076807,-0.326709,-1.36656]],[[43,65,2,-2,-2,12,-2,-2,6,-2,-2],[0.5,0.5,0.9350000023841858,-2.0,-2.0,0.42800000309944153,-2.0,-2.0,0.7529999911785126,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004388,0.000674,-0.003574,0.364298,-0.146485,0.060353,1.043714,0.207842,0.062885,-0.625309,0.935813]],[[64,5,9,-2,-2,8,-2,-2,-2],[0.5,0.9835000038146973,0.9494999945163727,-2.0,-2.0,0.9884999990463257,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.000872,-0.003964,0.001511,-0.22694,0.193028,-0.049738,-1.262775,-0.145775,0.789636]],[[51,42,2,-2,-2,12,-2,-2,-2],[0.5,0.02777777798473835,0.9850000143051147,-2.0,-2.0,0.2604999989271164,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.001865,-0.001537,0.024415,0.572779,-0.394677,-0.01055,0.684347,-0.207932,0.626523]],[[17,19,16,-2,-2,12,-2,-2,9,-2,-2],[0.5,0.5,0.5,-2.0,-2.0,0.5895000100135803,-2.0,-2.0,0.9050000011920929,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001887,0.004869,0.009062,0.139324,-0.562143,-0.039802,-0.630294,-1.415525,-0.066154,-1.51761,-0.106447]],[[42,9,4,-2,-2,63,-2,-2,4,53,-2,-2,-2],[0.4580419659614563,0.945499986410141,0.02949999924749136,-2.0,-2.0,0.5,-2.0,-2.0,0.8709999918937683,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-5.9e-05,0.004374,-0.014999,0.752655,-0.382292,0.019136,0.104628,1.036342,-0.050833,-0.026838,0.73932,-0.967957,-1.464677]],[[5,5,3,-2,-2,48,-2,-2,7,69,-2,-2,2,-2,-2],[0.6845000088214874,0.23349999636411667,0.34700000286102295,-2.0,-2.0,0.5,-2.0,-2.0,0.8244999945163727,0.5,-2.0,-2.0,0.9950000047683716,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.001738,0.007399,-0.003702,0.109584,-0.514953,0.04718,0.385471,1.099522,-0.026452,-0.092955,-0.132575,-1.429522,-0.00478,0.419582,-1.051377]],[[63,10,7,-2,-2,9,-2,-2,-2],[0.5,0.2678571492433548,0.029000000096857548,-2.0,-2.0,0.3865000009536743,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.004508,0.001253,-0.01428,-0.43205,0.095702,0.020124,-0.199359,0.542108,0.505041]],[[5,69,6,-2,-2,42,-2,-2,4,2,-2,-2,-2],[0.6775000095367432,2.5,0.009499999694526196,-2.0,-2.0,0.2182539701461792,-2.0,-2.0,0.8935000002384186,0.9750000238418579,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.00232,0.012901,0.005688,1.415765,0.012026,0.076421,1.061892,-0.000989,-0.026406,-0.015053,0.302235,-0.521802,-1.302861]],[[69,53,46,-2,-2,5,-2,-2,5,-2,42,-2,-2],[2.5,0.5,3.5,-2.0,-2.0,0.09950000047683716,-2.0,-2.0,0.027999999932944775,-2.0,0.1569230780005455,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.004232,-0.010117,0.00971,0.016651,0.955824,-0.024835,-0.450315,-0.101558,0.035328,1.100293,0.000255,0.552261,-0.565661]],[[9,53,20,-2,-2,4,-2,-2,63,8,-2,-2,-2],[0.9494999945163727,0.5,0.5,-2.0,-2.0,0.05849999934434891,-2.0,-2.0,0.5,0.4399999976158142,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[-0.001034,-0.01762,0.011981,-0.992218,0.670801,-0.044262,0.072884,-0.624836,0.011838,0.00435,0.680697,-0.135961,0.968482]],[[71,42,4,-2,-2,2,-2,-2,-2],[0.5,0.5857143104076385,0.28550000488758087,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.001124,0.001689,0.003791,0.187229,-0.148176,-0.103528,-1.619057,-1.225327,-0.983404]],[[38,6,6,-2,-2,4,-2,-2,8,6,-2,-2,-2],[0.5,0.7080000042915344,0.6550000011920929,-2.0,-2.0,0.08950000256299973,-2.0,-2.0,0.984499990940094,0.9869999885559082,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.004311,-0.002654,-0.022878,-0.163827,-1.289159,0.009711,0.55056,-0.099793,0.036348,0.021382,0.504619,-0.664974,1.105618]],[[42,69,12,-2,-2,-2,42,-2,66,-2,-2],[0.02777777798473835,1.5,0.35749998688697815,-2.0,-2.0,-2.0,0.08712121471762657,-2.0,0.5,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.000934,0.022919,0.002467,-1.105876,0.183859,0.903527,-0.006563,-0.781672,-0.003363,-0.00026,-0.794636]],[[20,12,7,-2,-2,-2,30,8,-2,-2,8,-2,-2],[0.5,0.6569999754428864,0.0605000015348196,-2.0,-2.0,-2.0,0.5,0.6304999887943268,-2.0,-2.0,0.7159999907016754,-2.0,-2.0],[1,2,3,-1,-1,-1,7,8,-1,-1,11,-1,-1],[6,5,4,-1,-1,-1,10,9,-1,-1,12,-1,-1],[-0.000344,-0.068944,-0.034823,-0.889503,0.003131,-1.156863,0.003606,-0.000521,0.258339,-0.256957,0.058597,-0.511418,1.078397]],[[17,42,6,-2,-2,35,-2,-2,4,-2,-2],[0.5,0.06458333507180214,0.37800000607967377,-2.0,-2.0,0.5,-2.0,-2.0,0.6245000064373016,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.004859,0.008495,0.036226,-0.367943,0.661378,-0.001415,-0.05532,1.104055,-0.069266,-1.521678,0.298129]],[[4,69,5,-2,-2,42,-2,-2,44,6,-2,-2,8,-2,-2],[0.0345000009983778,1.5,0.3969999998807907,-2.0,-2.0,0.11678004637360573,-2.0,-2.0,0.5,0.9784999787807465,-2.0,-2.0,0.9639999866485596,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.002097,0.022419,0.006561,0.378131,-0.295577,0.080694,1.134036,0.119224,-0.006483,0.008676,-0.182517,0.814522,-0.026036,-0.118298,-1.082672]],[[4,7,8,-2,-2,6,-2,-2,3,69,-2,-2,-2],[0.9715000092983246,0.23499999940395355,0.9560000002384186,-2.0,-2.0,0.9790000021457672,-2.0,-2.0,0.010499999858438969,0.5,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,-1],[0.000401,-0.005781,-0.022046,-0.121572,-0.561225,0.018983,0.031506,1.008398,0.031473,0.013038,0.679844,-0.452604,1.952675]],[[5,48,37,-2,-2,12,-2,-2,-2],[0.9964999854564667,0.5,0.5,-2.0,-2.0,0.5394999980926514,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.003489,0.005818,0.002163,-0.200048,0.159815,0.061167,0.916698,-0.134902,-0.933228]],[[21,19,5,-2,-2,6,-2,-2,-2],[0.5,0.5,0.9895000159740448,-2.0,-2.0,0.8925000131130219,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[0.002335,5.9e-05,0.004224,0.098089,-0.460498,-0.039978,-0.412348,-1.040653,1.000628]],[[65,56,5,-2,-2,12,-2,-2,8,-2,-2],[0.5,0.5,0.6545000076293945,-2.0,-2.0,0.48100000619888306,-2.0,-2.0,0.11649999767541885,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[-0.00143,-0.006085,-0.001918,0.095611,-0.252605,-0.09481,-1.158312,-0.647457,0.067894,0.357131,0.760514]],[[16,48,69,-2,-2,6,-2,-2,6,-2,-2],[0.5,0.5,4.5,-2.0,-2.0,0.7529999911785126,-2.0,-2.0,0.9535000026226044,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,-1],[0.001469,0.00536,0.001018,0.048566,-0.659507,0.064303,-0.393013,0.880854,-0.072485,-1.285347,-0.04175]],[[44,4,2,-2,-2,3,-2,-2,5,37,-2,-2,42,-2,-2],[0.5,0.004500000039115548,0.9000000059604645,-2.0,-2.0,0.8765000104904175,-2.0,-2.0,0.6229999959468842,2.5,-2.0,-2.0,0.18333333730697632,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[-0.002166,0.011322,-0.040416,0.676528,-1.035794,0.019135,0.175291,0.739888,-0.021946,-0.010818,-0.254109,0.497319,-0.133769,-1.576521,-0.569677]],[[44,2,7,-2,-2,6,-2,-2,6,-2,2,-2,-2],[0.5,0.9850000143051147,0.948500007390976,-2.0,-2.0,0.99549999833107,-2.0,-2.0,0.009499999694526196,-2.0,0.9350000023841858,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,-1,11,-1,-1],[8,5,4,-1,-1,7,-1,-1,10,-1,12,-1,-1],[-0.000534,0.011742,0.027753,0.244672,0.81862,-0.009736,-0.436987,1.273585,-0.01914,1.53088,-0.025555,0.447833,-0.311313]],[[54,4,12,-2,-2,12,-2,-2,-2],[0.5,0.351500004529953,0.4675000011920929,-2.0,-2.0,0.8384999930858612,-2.0,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,-1],[8,5,4,-1,-1,7,-1,-1,-1],[-0.00117,-0.003364,0.010661,0.464615,-0.06437,-0.020621,-0.391743,1.165858,0.771859]],[[37,73,42,-2,-2,-2,-2],[4.5,0.5,0.5857143104076385,-2.0,-2.0,-2.0,-2.0],[1,2,3,-1,-1,-1,-1],[6,5,4,-1,-1,-1,-1],[0.003861,0.007026,0.004677,0.079815,-1.362696,0.782558,-0.724088]],[[42,51,4,-2,-2,-2,8,-2,8,-2,-2],[0.4752941131591797,0.5,0.19449999928474426,-2.0,-2.0,-2.0,0.03099999949336052,-2.0,0.20600000023841858,-2.0,-2.0],[1,2,3,-1,-1,-1,7,-1,9,-1,-1],[6,5,4,-1,-1,-1,8,-1,10,-1,-1],[0.00278,0.006311,0.003284,0.190091,-0.139688,0.646904,-0.042037,0.677157,-0.063279,-1.497913,-0.43608]],[[42,3,69,-2,-2,2,-2,-2,8,53,-2,-2,6,-2,-2],[0.33908046782016754,0.009499999694526196,0.5,-2.0,-2.0,0.9600000083446503,-2.0,-2.0,0.0625,0.5,-2.0,-2.0,0.1524999961256981,-2.0,-2.0],[1,2,3,-1,-1,6,-1,-1,9,10,-1,-1,13,-1,-1],[8,5,4,-1,-1,7,-1,-1,12,11,-1,-1,14,-1,-1],[0.005302,0.011704,-0.012878,0.292746,-0.554339,0.02666,0.454111,-2.8e-05,-0.020626,0.033156,0.220302,1.181521,-0.051559,-1.355,-0.506591]]]
UMBRAL = 0.20
# <<< MODELO


def _recorrer(arbol, x):
    """Baja por el árbol respondiendo a sus preguntas hasta llegar a una hoja."""
    variable, umbral, izq, der, valor = arbol
    nodo = 0
    while izq[nodo] != -1:
        nodo = izq[nodo] if x[variable[nodo]] <= umbral[nodo] else der[nodo]
    return valor[nodo]


def probabilidad(feats):
    """Probabilidad de alucinación: suma de todos los árboles -> sigmoide."""
    # scikit-learn compara en precisión simple (float32); hacemos lo mismo para
    # que el bot dé exactamente las mismas probabilidades que en el entrenamiento
    x = [struct.unpack("f", struct.pack("f", float(feats.get(n, 0.0))))[0] for n in NOMBRES]
    z = INICIAL + TASA * sum(_recorrer(a, x) for a in ARBOLES)
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def reescalar(p, umbral):
    """Lleva el corte elegido (umbral) a 0,5 manteniendo el orden de los casos:
    [0, umbral] -> [0, 0,5] y [umbral, 1] -> [0,5, 1]."""
    if p < umbral:
        return 0.5 * p / umbral
    return 0.5 + 0.5 * (p - umbral) / (1.0 - umbral)


class Guardian:
    def score(self, caso):
        try:
            p = probabilidad(caracteristicas(caso))
        except Exception:
            # Ante un caso raro, mejor dar la alarma (un escape cuesta 4 veces más)
            return 1.0
        return reescalar(p, UMBRAL)

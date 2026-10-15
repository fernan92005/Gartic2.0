"""Características B1–B4 deterministas; sin etiquetas, IDs ni dependencias externas.

No aplican descuentos ni vetos sobre el riesgo. Sus valores se aprenden con GB.
Las equivalencias son conservadoras y no constituyen una prueba de fidelidad.
"""
import re
import unicodedata

B1 = ['matiz_clausula_' + k for k in ('restrictivo','condicional','aproximado','parcial')]
B2 = ['matiz_parafrasis_preservado','matiz_parafrasis_perdido','matiz_equivalencia_incierta']
B3 = ['cobertura_mejor_fuente','cobertura_union','ganancia_cobertura_union','numero_fuentes_complementarias']
B4 = ['sujeto_generalizado','sujeto_generalizado_confianza']
STOP = set('a al algo ante antes asi aun cada como con cual cuando de del desde donde durante e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba estan estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mientras muy ni no nos o otra otras otro otros para pero por que quien se sea segun ser si sido sin sobre son su sus tambien tan tanto te tiene tienen todo tras tu un una uno unos unas y ya yo'.split())
NAME_STOP = set('el la los las un una unos unas en de del por para con segun este esta estos estas esto eso ademas enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split())
PREDICATE = re.compile(r'\b(?:es|son|sera|seran|fue|fueron|tiene|tienen|hay|durara|duraran|cuenta|cuentan|incluye|incluyen|preve|preven|previsto|prevista|planea|planean|podria|podrian|puede|pueden|aprob\w*|aprueb\w*|realiz\w*|recib\w*|instal\w*|cost\w*|pag\w*|cerr\w*|abr\w*|impart\w*|afect\w*|estudia|estudian|limita|limitan|\w+(?:aron|ieron|aba|aban|ara|aran|eran))\b')
# Cada grupo fuerte mantiene el tipo y la fuerza del matiz. No se mezclan
# previsión, posibilidad, condición, cota superior, aproximación y mayoría.
EQUIVALENCES = {
 'restriction': ('restrictivo', r'\b(?:solo|solamente|unicamente|exclusivamente|se limita(?:n)? a)\b'),
 'exception': ('restrictivo', r'\b(?:salvo|excepto|a excepcion de)\b'),
 'approval': ('condicional', r'\b(?:si se aprueba\w*|en caso de aprobacion|sujet[oa]s? a (?:la )?aprobacion)\b'),
 'condition': ('condicional', r'\b(?:siempre que|a condicion de|si (?:el|la|los|las|lo|se))\b'),
 'forecast': ('condicional', r'\b(?:preve\w*|previst[oa]s?|planea\w*|tiene(?:n)? previsto)\b'),
 'possibility': ('condicional', r'\b(?:podria\w*|es posible|cabe la posibilidad)\b'),
 'study': ('condicional', r'\b(?:se estudia\w*|se contempla\w*)\b'),
 'proposal': ('condicional', r'\b(?:propuesta|propone\w*|proyecta\w*)\b'),
 'approximation': ('aproximado', r'\b(?:aproximadamente|alrededor de|cerca de|unos|unas)\b'),
 'upper': ('aproximado', r'\b(?:hasta|como maximo|a lo sumo)\b'),
 'lower': ('aproximado', r'\b(?:al menos|como minimo)\b'),
 'more': ('aproximado', r'\bmas de\b'),
 'less': ('aproximado', r'\bmenos de\b'),
 'almost': ('aproximado', r'\bcasi\b'),
 'some': ('parcial', r'\b(?:algun[oa]s|determinad[oa]s|parte de)\b'),
 'several': ('parcial', r'\bvari[oa]s\b'),
 'majority': ('parcial', r'\bla mayoria\b'),
 'subset': ('parcial', r'\b(?:de (?:los|las) \d|entre \d|de ell[oa]s)\b'),
}
# Relaciones débiles: nunca cuentan como preservación fuerte.
WEAK_PAIRS = {frozenset(x) for x in [('approval','proposal'),('approval','forecast'),('possibility','study'),('forecast','proposal'),('some','majority'),('some','several'),('upper','approximation')]}
COLLECTIVES = {'municipios':'municipio','empresas':'empresa','centros':'centro','organizaciones':'organizacion','hospitales':'hospital','ayuntamientos':'ayuntamiento','colegios':'colegio','locales':'local','nadadores':'nadador','ciudades':'ciudad','provincias':'provincia'}


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFD',str(text or '').lower()) if unicodedata.category(c)!='Mn')


def roots(text):
    words=re.findall(r'[a-z]+',norm(text));out=set()
    for word in words:
        if word in STOP or len(word)<3:continue
        if word.startswith(('aprob','aprueb')):out.add('aprob')
        else:out.add(word[:5])
    return out


def entities(text):
    return {norm(w) for w in re.findall(r'\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,}\b',text or '') if norm(w) not in NAME_STOP}


def numbers(text):
    return set(re.findall(r'\b\d+(?:[.,]\d+)*\b',norm(text)))


def clauses(text):
    """Conserva decimales, listas y complementos; separa proposiciones claras."""
    chunks=re.split(r';|(?<!\d)[.!?](?!\d)|(?<=\d)[.!?](?!\d)|\bpero\b',str(text or ''),flags=re.I)
    out=[]
    for chunk in chunks:
        pieces=re.split(r'(,|\s+y\s+)',chunk)
        current=pieces[0]
        for i in range(1,len(pieces),2):
            sep=pieces[i];right=pieces[i+1]
            qualifier=re.match(r'\s*(?:solo|si\b|aunque|salvo|excepto|en caso|sujeto|sujeta)',norm(right))
            if PREDICATE.search(norm(current)) and PREDICATE.search(norm(right)) and not qualifier:
                if current.strip():out.append(current.strip())
                current=right
            else:current+=sep+right
        if current.strip():out.append(current.strip())
    return out


def relevant(a,b):
    """>=2 raíces compartidas y cobertura sustancial, sin sujeto incompatible."""
    ra,rb=roots(a),roots(b);common=ra&rb
    if len(common)<2:return False
    ea,eb=entities(a),entities(b)
    if ea and eb and not ea&eb:return False
    overlap=len(common)/max(1,min(len(ra),len(rb)))
    anchor=bool((ea&eb) or (numbers(a)&numbers(b)))
    return overlap>=.5 or (anchor and overlap>=.3)


def signals_rank(caso):
    sources=caso.get('fuentes') or [];response=caso.get('respuesta') or '';rr=roots(response);rank=[]
    signals=caso.get('senales') or {};items=(signals.get('oraciones') or [])+(signals.get('fragmentos') or [])
    for i,source in enumerate(sources):
        lexical=len(rr&roots(source))/max(1,len(rr));e=sim=0.;cited=False
        for item in items:
            for key in ('respaldo_por_fuente','similitud_por_fuente'):
                values=item.get(key) or []
                value=float(values[i] or 0.) if i<len(values) else 0.
                if key=='respaldo_por_fuente':e=max(e,value)
                else:sim=max(sim,value)
            cited=cited or (item.get('llm') or {}).get('fuente')==i
        rank.append((.4*e+.3*sim+.25*lexical+.05*cited,i))
    return [i for _,i in sorted(rank,key=lambda pair:(-pair[0],pair[1]))[:3]]


def nuance_groups(text):
    t=norm(text)
    # «Hasta el 31 de julio» expresa fin temporal, no cota aproximada de cifra.
    t=re.sub(r'\bhasta\s+(?:el\s+)?\d{1,2}\s+de\s+[a-z]+','limite temporal',t)
    found={group for group,(_,pattern) in EQUIVALENCES.items() if re.search(pattern,t)}
    if 'approval' in found:
        found.discard('condition')
        if re.search(r'\bsolo si se aprueba',t):
            without=re.sub(r'\bsolo si se aprueba\w*','',t)
            if not re.search(EQUIVALENCES['restriction'][1],without):found.discard('restriction')
    return found


def aligned_response(source_clause,response_clauses):
    """Preservación en toda la respuesta, ligada a la misma entidad/afirmación."""
    aligned=[];subject=entities(source_clause)
    for i,claim in enumerate(response_clauses):
        if relevant(source_clause,claim):aligned.append(claim)
        # Anáfora conservadora: la condición en la siguiente oración debe
        # referir a la oración alineada y no introducir otra entidad.
        elif i and re.match(r'\s*(?:esto|esta medida|dicha medida|este plan|esa actuacion)\b',norm(claim)):
            previous=response_clauses[i-1]
            if relevant(source_clause,previous) and not (entities(claim)-subject):aligned.append(claim)
        elif subject and entities(claim)&subject and not (entities(claim)-subject):
            # Una condición separada sobre el mismo sujeto solo se acepta si
            # comparte también contenido de la acción (no basta el nombre).
            nonentity=roots(claim)-{x[:5] for x in subject}
            if nonentity&(roots(source_clause)-{x[:5] for x in subject}):aligned.append(claim)
    return aligned


def matiz_features(caso):
    response=caso.get('respuesta') or '';sources=caso.get('fuentes') or [];rc=clauses(response)
    lost={k:0 for k in ('restrictivo','condicional','aproximado','parcial')};preserved=missing=uncertain=0;comparisons=0;seen=set()
    for i in signals_rank(caso):
        for source in clauses(sources[i]):
            if not any(relevant(source,claim) for claim in rc):continue
            aligned=aligned_response(source,rc);answer=' '.join(aligned)
            sg=nuance_groups(source);ag=nuance_groups(answer)
            # B1 conserva los patrones literales; B2 añade equivalencias fuertes.
            for category in lost:
                expressions={m.group(0) for group,(cat,pattern) in EQUIVALENCES.items() if cat==category for m in re.finditer(pattern,norm(source))}
                for expression in expressions:
                    key=(norm(source),category,expression)
                    if key in seen:continue
                    seen.add(key);comparisons+=1
                    if not re.search(r'\b'+re.escape(expression)+r'\b',norm(answer)):lost[category]+=1
            for group in sg:
                if group in ag:preserved+=1
                else:
                    missing+=1
                    if any(frozenset((group,other)) in WEAK_PAIRS for other in ag):uncertain+=1
    out={'matiz_clausula_'+k:float(min(v,2)) for k,v in lost.items()}
    out.update(matiz_parafrasis_preservado=float(min(preserved,2)),matiz_parafrasis_perdido=float(min(missing,2)),matiz_equivalencia_incierta=float(min(uncertain,2)))
    out.update({'matiz_clausula_ratio_'+k:lost[k]/max(1,comparisons) for k in lost})
    return out


def coverage_features(caso):
    rr=roots(caso.get('respuesta') or '');sources=caso.get('fuentes') or [];sets=[roots(sources[i])&rr for i in signals_rank(caso)]
    individual=[len(s)/max(1,len(rr)) for s in sets];union=set().union(*sets) if sets else set()
    # Cuenta fuentes cuya retirada reduce cobertura; no convierte duplicados
    # en fuentes complementarias. Es 0 si una sola basta para toda la unión.
    complementary=sum(bool(s-set().union(*(v for j,v in enumerate(sets) if j!=i))) for i,s in enumerate(sets)) if len(sets)>1 else 0
    gain=len(union)/max(1,len(rr))-max(individual,default=0.)
    return dict(cobertura_mejor_fuente=max(individual,default=0.),cobertura_union=len(union)/max(1,len(rr)),ganancia_cobertura_union=max(0.,gain),numero_fuentes_complementarias=float(complementary if gain>0 else 0))


def subject_features(caso):
    sources=caso.get('fuentes') or [];response=caso.get('respuesta') or '';suspicion=0.;confidence=0.
    source_clauses=[c for i in signals_rank(caso) for c in clauses(sources[i])]
    for claim in clauses(response):
        tokens=set(re.findall(r'[a-z]+',norm(claim)));collective=[p for p in COLLECTIVES if p in tokens]
        if not collective:continue
        # Una restricción explícita evita identificar el grupo como universal.
        if re.search(r'\b(?:algunos|algunas|estos|estas|esos|esas|determinados|determinadas|parte de|solo|unicamente|de los \d|de las \d)\b',norm(claim)):continue
        cp=PREDICATE.search(norm(claim))
        if cp is None:continue
        actions={m.group(0)[:5] for m in PREDICATE.finditer(norm(claim))}
        for plural in collective:
            noun=COLLECTIVES[plural];nr=noun[:5]
            action_roots=roots(claim)-{nr}
            # Si alguna evidencia menciona el colectivo o al menos dos sujetos
            # de esa clase con la misma acción, nos abstenemos.
            candidates=[s for s in source_clauses if len(action_roots&roots(s))>=1 and any(a in {m.group(0)[:5] for m in PREDICATE.finditer(norm(s))} for a in actions)]
            if any(plural in re.findall(r'[a-z]+',norm(s)) for s in candidates):continue
            relevant_sources=[s for s in candidates if re.search(r'\b'+re.escape(noun)+r'\b',norm(s)) or entities(s)]
            names=set().union(*(entities(s) for s in relevant_sources)) if relevant_sources else set()
            claim_names=entities(claim)
            if len(names)>1 or len(claim_names)>1:continue
            for source in relevant_sources:
                content=roots(source)&action_roots
                if len(content)<2 or not PREDICATE.search(norm(source)):continue
                # Colectivo y predicado deben pertenecer a la misma cláusula.
                if norm(source).find(plural)>=0:continue
                explicit_singular=bool(re.search(r'\b'+re.escape(noun)+r'\b',norm(source)))
                if not explicit_singular and len(entities(source))!=1:continue
                suspicion=1.;confidence=max(confidence,.85 if explicit_singular else .6)
    return dict(sujeto_generalizado=suspicion,sujeto_generalizado_confianza=confidence)


def caracteristicas_b(caso):
    out=matiz_features(caso);out.update(coverage_features(caso));out.update(subject_features(caso));return out

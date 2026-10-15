"""Tablas diagnósticas derivadas de OOF; no modifica ningún clasificador."""
import collections
import itertools
import json
from pathlib import Path
import re
import numpy as np
from sklearn.metrics import roc_auc_score
from auditar_version import ROOT,OUT,load,clean

TYPES=['distorsion','contradiccion','no_respaldado','sobregeneralizacion']
LABELS={'distorsion':'Distorsión','contradiccion':'Contradicción','no_respaldado':'No respaldado','sobregeneralizacion':'Sobregeneralización'}


def save(name,lines):(OUT/name).write_text('\n'.join(lines)+'\n')

def fmt(a):
    if not len(a):return '—'
    q=np.quantile(a,[.25,.5,.75]);return f'{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]'


def table_metric(name,m):
    return f"| {name} | {m['n']} | {m['score']:.4f} | {m['auroc']:.4f} | {m['fp']} | {m['fn']} | {m['cost']} | [[{m['tn']},{m['fp']}],[{m['fn']},{m['tp']}]] |"


def generate(allcases,dev,hold,records,res,folds,metric,metrics,write_csv):
    cases=[allcases[i] for i in dev];y=np.array([r['etiqueta'] for r in records]);h=np.array([r['id'].startswith('H') for r in records]);p=np.array([r['joseca_raw'] for r in records]);t=res['provenance']['threshold'];pred=p>=t
    fp=(y==0)&pred;fn=(y==1)&~pred;totalcost=int(fp.sum()+4*fn.sum());fields=list(records[0]['signals']);s={k:np.array([r['signals'][k] for r in records]) for k in fields};features=[r['features'] for r in records]
    e4=load(ROOT/'entregas/candela_v1/mi_bot.py','e4_for_report');joseca=load(ROOT/'entregas/joseca_v1/mi_bot.py','joseca_for_report')
    variants=[('joseca_oof_corte_actual',p,t),('joseca_oof_corte_anidado_ponderado',p,np.array([r['nested_threshold'] for r in records])),('joseca_oof_corte_anidado_H',p,np.array([r['nested_h_threshold'] for r in records])),('candela_hgb_oof_referencia',np.array([r['hgb_raw'] for r in records]),records[0]['e4_threshold']),('candela_e4_oof_referencia',np.array([r['e4_raw'] for r in records]),records[0]['e4_threshold'])]
    variants.insert(1,('joseca_oof_corte_anidado_oficial',p,np.array([r['nested_official_threshold'] for r in records])))
    type_rows=[];by_type={}
    for name,scores,threshold in variants:
        for domain,scope in [('todos',np.ones(len(y),dtype=bool)),('H',h),('no-H',~h)]:
            pp=scores>=threshold;cost=int(np.sum(scope&(y==0)&pp)+4*np.sum(scope&(y==1)&~pp))
            for category in TYPES:
                mask=scope&np.array([r['tipo']==category for r in records]);n=int(mask.sum());miss=int(np.sum(mask&~pp));tp=n-miss
                row=dict(arquitectura=name,protocolo='OOF agrupado desarrollo',dominio=domain,tipo=category,casos=n,tp=tp,fn=miss,recall=tp/n if n else '',tn=0,fp=0,coste_fn=4*miss,coste_fp=0,coste=4*miss,porcentaje_coste_total=100*4*miss/cost if cost else 0.)
                type_rows.append(row)
                if name=='joseca_oof_corte_actual' and domain=='todos':by_type[category]=row
            mask=scope&(y==0);nfp=int(np.sum(mask&pp));type_rows.append(dict(arquitectura=name,protocolo='OOF agrupado desarrollo',dominio=domain,tipo='fiel_sin_categoria_alucinacion',casos=int(mask.sum()),tp=0,fn=0,recall='',tn=int(np.sum(mask&~pp)),fp=nfp,coste_fn=0,coste_fp=nfp,coste=nfp,porcentaje_coste_total=100*nfp/cost if cost else 0.))
    write_csv(OUT/'metricas_por_tipo.csv',type_rows)
    ranked=sorted(TYPES,key=lambda c:(-by_type[c]['coste_fn'],-by_type[c]['casos'],c))
    header=['| Arquitectura / dominio | N | Score | AUROC | FP | FN | Coste | Matriz [[TN,FP],[FN,TP]] |','|---|---:|---:|---:|---:|---:|---:|---|']
    lines=['# Diagnóstico por tipo — Joseca v1','',f"Objetivo principal: `entregas/joseca_v1/mi_bot.py`, SHA-256 `{res['provenance']['sha256']}`. Logística de 60 características, peso H=5, L2=10, corte crudo **0,15**. No es el HGB ni el E4 de Candela.",'',
           '## Métrica oficial verificada','',
           'El DOCX (sección 7.1), kit/README.md y evaluar_local.py coinciden: score = 1 − (4 FN + FP)/N_fieles; decisión score ≥0,5. No existen pesos diferentes por tipo. Cada FN cuesta cuatro y cada FP uno. El peso H=5 es un peso de entrenamiento, no una penalización del evaluador. La ponderación 0,7 H + 0,3 no-H es un criterio interno de desarrollo, no una regla del leaderboard. El caso especial del evaluador cuando no hay fieles devuelve recall; no se confunde con la fórmula normal. No se encontró discrepancia con el DOCX.','',
           '## Evaluación OOF principal','',*header]
    for domain in ['H','no-H','todos']:lines.append(table_metric('Joseca OOF / '+domain,res['joseca_oof'][domain]))
    lines+=['','## Ranking por coste de FN (desarrollo completo al corte actual)','',
            '| Orden | Tipo | Casos positivos | TP | FN | Recall | Coste FN | % del coste total, incluyendo FP | IC95% recall por grupos |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for order,category in enumerate(ranked,1):
        r=by_type[category];ci=res['bootstrap_current']['recall_'+category]
        lines.append(f"| {order} | {LABELS[category]} | {r['casos']} | {r['tp']} | {r['fn']} | {r['recall']:.3f} | {r['coste_fn']} | {r['porcentaje_coste_total']:.1f}% | [{ci[0]:.3f},{ci[1]:.3f}] |")
    lines+=['','Hay empate de coste entre distorsión/sobregeneralización y entre no respaldado/contradicción. El orden dentro de cada empate usa frecuencia y no demuestra una diferencia de coste esperada.','',
            '## Estabilidad de los falsos negativos','', '| Tipo | FN fold 0 | FN fold 1 | FN fold 2 | FN fold 3 | Grupos con FN |','|---|---:|---:|---:|---:|---:|']
    stability=[]
    for category in ranked:
        bad=fn&np.array([r['tipo']==category for r in records]);counts=[int(sum(bad[i] and r['fold']==k for i,r in enumerate(records))) for k in range(4)]
        ng=len({r['group'] for i,r in enumerate(records) if bad[i]})
        stability.append(dict(tipo=category,**{f'fn_fold_{k}':n for k,n in enumerate(counts)},grupos_con_fn=ng))
        lines.append('| '+LABELS[category]+' | '+' | '.join(str(n) for n in counts+[ng])+' |')
    write_csv(OUT/'estabilidad_por_tipo.csv',stability)
    lines+=['','La concentración de errores por fold y los IC de recall impiden interpretar pocos FN como una jerarquía estable. El bootstrap remuestrea 2.000 veces grupos completos con semilla 20261015, condicionado a estos modelos OOF: no incluye la variación de volver a seleccionar la arquitectura ni nuevas realizaciones de entrenamiento.']
    lines+=['',f"Falsas alarmas sobre fieles: **{int(fp.sum())}**, coste **{int(fp.sum())}**, {100*fp.sum()/max(1,totalcost):.1f}% del coste total **{totalcost}**. No se les asigna un tipo de alucinación.", '',
            'Todos los 1.707 ejemplos tienen un único campo tipo: 817 fieles con null y 890 positivos con una de las cuatro etiquetas. No hay listas ni categorías múltiples. Los costes por tipo más FP suman exactamente el coste oficial; las cohortes de mecanismo posteriores sí se solapan y no deben sumarse.','',
            '## Categorías por arquitectura y dominio','']
    for name in [v[0] for v in variants]:
        lines+=['### '+name,'','| Dominio | Tipo | Casos | TP | FN | Recall | Coste FN |','|---|---|---:|---:|---:|---:|---:|']
        for r in type_rows:
            if r['arquitectura']==name and r['tipo'] in TYPES:lines.append(f"| {r['dominio']} | {LABELS[r['tipo']]} | {r['casos']} | {r['tp']} | {r['fn']} | {r['recall']:.3f} | {r['coste_fn']} |")
        lines.append('')
    # Mechanism cohorts are explicitly proxies, never new dataset labels.
    cohort={
      'Distorsión con año explícito':np.array([r['tipo']=='distorsion' and bool(re.search(r'\b(?:19|20)\d{2}\b',c['respuesta'])) for r,c in zip(records,cases)]),
      'Distorsión con cuenta sospechosa':np.array([r['tipo']=='distorsion' and (r['features'].get('num_cuenta_falsa',0)>0 or r['features'].get('num_proporcion_falsa',0)>0) for r in records]),
      'Distorsión con posible atribución cruzada':np.array([r['tipo']=='distorsion' and r['features'].get('num_entidad_ajena',0)>0 for r in records]),
      'Contradicción sin negación explícita':np.array([r['tipo']=='contradiccion' and not re.search(r'\b(no|nunca|jamas|tampoco)\b',c['respuesta'].lower()) for r,c in zip(records,cases)]),
      'Contradicción con verbo direccional':np.array([r['tipo']=='contradiccion' and bool(re.search(r'\b(aument\w*|crec\w*|subi\w*|redu\w*|dismin\w*|baj\w*|cay\w*)',c['respuesta'].lower())) for r,c in zip(records,cases)]),
      'No respaldado con causa añadida':np.array([r['tipo']=='no_respaldado' and r['features'].get('anadido_causal',0)>0 for r in records]),
      'No respaldado con MiniMax favorable':np.array([r['tipo']=='no_respaldado' for r in records])&(s['llm_all_supported']>0),
      'Sobregeneralización con absoluto añadido':np.array([r['tipo']=='sobregeneralizacion' and r['features'].get('anadido_universal',0)>0 for r in records]),
      'Sobregeneralización con condición perdida':np.array([r['tipo']=='sobregeneralizacion' and (r['features'].get('matiz_perdido_fuente_condicional',0)>0 or r['features'].get('matiz_perdido_fuente_restrictivo',0)>0) for r in records]),
      'Respuesta >=60 palabras':np.array([len(c['respuesta'].split())>=60 for c in cases]),
    }
    lines+=['## Mecanismos observables (proxies, no nuevos subtipos anotados)','', '| Cohorte | N | FP | FN | Coste |','|---|---:|---:|---:|---:|']
    for name,mask in cohort.items():lines.append(f'| {name} | {int(mask.sum())} | {int((mask&fp).sum())} | {int((mask&fn).sum())} | {int((mask&fp).sum()+4*(mask&fn).sum())} |')
    lines+=['','Los proxies no permiten distinguir exhaustivamente fecha incorrecta frente a fecha añadida, oposición semántica, atribución o pérdida de excepción. Se conservan los tipos oficiales y se evita reclasificar manualmente.','',
            '## Ejemplos FN de desarrollo, sin proponer reglas por caso','']
    for category in ranked:
        idx=sorted([i for i,r in enumerate(records) if fn[i] and r['tipo']==category],key=lambda i:p[i])[:4]
        lines+=['### '+LABELS[category],'']
        for i in idx:lines.append(f"- {records[i]['id']} (fold {records[i]['fold']}, p={p[i]:.3f}): {cases[i]['respuesta']}")
        if not idx:lines.append('Sin FN en este protocolo y corte; no implica ausencia de fallos futuros.')
        lines.append('')
    # Robustness by domain/structure, explicitly diagnostic rather than predictors.
    masks=[('dominio','H',h),('dominio','no-H',~h)]
    for field,values,bins in [
        ('fuentes',[len(c['fuentes']) for c in cases],[(0,1),(2,5),(6,8),(9,1000)]),
        ('oraciones',[len(c['senales']['oraciones']) for c in cases],[(0,1),(2,3),(4,1000)]),
        ('fragmentos',[len(c['senales']['fragmentos']) for c in cases],[(0,1),(2,4),(5,1000)]),
        ('palabras',[len(c['respuesta'].split()) for c in cases],[(0,24),(25,59),(60,10000)])]:
        a=np.array(values)
        for lo,hi in bins:masks.append((field,f'{lo}–{hi}' if hi<1000 else f'>={lo}',(a>=lo)&(a<=hi)))
    masks.extend([('entidades','>=2 nombres (proxy)',np.array([len(joseca.nombres_propios(c['respuesta']))>=2 for c in cases])),('cifras','presentes',np.array([r['features']['num_hay_cifras']>0 for r in records])),('fechas','año explícito',np.array([bool(re.search(r'\b(?:19|20)\d{2}\b',c['respuesta'])) for c in cases])),('multifuente','respaldo repartido (proxy)',(s['s_e_any_min']<.5)&((s['s_e_all_min']>.5)|(s['f_e_any_min']>.5))),('tecnico_legal','palabras documentales (proxy)',np.array([bool(re.search(r'\b(apartado|articulo|poliza|contrato|plan|norma|reglamento|postmortem)\b',e4.normalize(' '.join(c['fuentes'])))) for c in cases]))])
    robustness=[]
    for field,label,mask in masks:
        if mask.any():robustness.append(dict(variable=field,cohorte=label,**metric(y[mask],p[mask],t)))
    write_csv(OUT/'robustez.csv',robustness)
    scenario=[]
    for group in sorted(set(r['group'] for r in records)):
        mask=np.array([r['group']==group for r in records]);scenario.append(dict(grupo=group,escenarios='|'.join(sorted(set(records[i]['id'].split('::')[0] for i in np.flatnonzero(mask)))),**metric(y[mask],p[mask],t)))
    write_csv(OUT/'metricas_por_escenario.csv',scenario)
    lines+=['## Robustez y variación por escenario','',
            'Métricas por cada grupo en `metricas_por_escenario.csv`; cortes por fold en `folds_detalle.json`. Las variables estructurales siguientes son solo diagnósticas. La cohorte técnico/legal es una búsqueda léxica y no identifica exhaustivamente el dominio. No se ha puntuado ni etiquetado el giro.','',
            '| Variable/cohorte | N | Score | AUROC | FP | FN |','|---|---:|---:|---:|---:|---:|']
    for row in robustness:lines.append(f"| {row['variable']}: {row['cohorte']} | {row['n']} | {row['score']:.3f} | {row['auroc']:.3f} | {row['fp']} | {row['fn']} |")
    save('diagnostico_tipos.md',lines)
    # Full signal distributions by category, correctness and domain.
    distributions=[]
    for domain,scope in [('todos',np.ones(len(y),dtype=bool)),('H',h),('no-H',~h)]:
        for category in ['fiel']+TYPES:
            cat=(y==0) if category=='fiel' else np.array([r['tipo']==category for r in records])
            for outcome,mask in [('todos',scope&cat),('error',scope&cat&(fp|fn)),('acierto',scope&cat&~(fp|fn))]:
                if not mask.any():continue
                for field in fields:
                    a=s[field][mask];q=np.quantile(a,[.25,.5,.75]);distributions.append(dict(dominio=domain,tipo=category,resultado=outcome,senal=field,n=int(mask.sum()),media=float(np.mean(a)),p25=float(q[0]),mediana=float(q[1]),p75=float(q[2])))
    write_csv(OUT/'distribuciones_senales.csv',distributions)
    nl=['# Diagnóstico NLI y MiniMax — OOF Joseca','',
        'Las señales se analizan con las etiquetas reales de desarrollo; no se recalculan NLI, similitud ni MiniMax. Riesgo NLI descriptivo = 1 − max(respaldo conjunto, mejor respaldo individual). Sus mínimos/máximos/medias se calculan por oración y fragmento, separados del score logístico de Joseca.','',
        '## Cuando MiniMax acepta alucinaciones','',
        '| Tipo | Positivos | MiniMax respalda todas | FN entre ellos | NLI conjunto/relevante >0,5 y MiniMax favorable | Ambos favorables NLI riesgo<=0,2 | FN entre ambos favorables |','|---|---:|---:|---:|---:|---:|---:|']
    judge_rows=[]
    for category in TYPES:
        mask=np.array([r['tipo']==category for r in records]);good=mask&(s['llm_all_supported']>0);ignored=good&((s['s_c_all_max']>.5)|(s['s_c_relevant_max']>.5));both=good&(s['s_risk_max']<=.2)
        row=dict(tipo=category,positivos=int(mask.sum()),minimax_acepta=int(good.sum()),fn_minimax_acepta=int((good&fn).sum()),nli_contradiccion_minimax_ignora=int(ignored.sum()),ambos_favorables=int(both.sum()),fn_ambos_favorables=int((both&fn).sum()));judge_rows.append(row)
        nl.append('| '+LABELS[category]+' | '+' | '.join(str(row[k]) for k in list(row)[1:])+' |')
    nl+=['','El veredicto favorable no garantiza fidelidad; el desacuerdo NLI puede aportar señal o proceder de una fuente distractora. Se distingue contradicción conjunta/alineada de máxima en cualquier fuente.','',
         '## Distribuciones principales: mediana [p25,p75]','',
         '| Tipo / resultado | N | Respaldo individual | Respaldo conjunto | Contradicción fuente | Contradicción conjunta | Similitud | Confianza MiniMax | Riesgo fragmento |','|---|---:|---|---|---|---|---|---|---|']
    for category in ['fiel']+TYPES:
        cat=(y==0) if category=='fiel' else np.array([r['tipo']==category for r in records])
        for outcome,mask in [('error',cat&(fp|fn)),('acierto',cat&~(fp|fn))]:
            if mask.any():nl.append(f'| {category}/{outcome} | {int(mask.sum())} | '+' | '.join(fmt(s[field][mask]) for field in ['s_e_any_min','s_e_all_min','s_c_any_max','s_c_all_max','s_similarity_max','llm_conf_mean','f_risk_max'])+' |')
    nl+=['','El CSV `distribuciones_senales.csv` amplía estas tablas por H/no-H e incluye respaldo y contradicción individual/conjunto, similitud, veredicto, confianza, desacuerdo, riesgo mínimo/máximo/medio por oración/fragmento y su diferencia.','',
         '## Proximidad a señales de respuestas fieles','',
         'AUROC de cada señal por categoría frente a fieles; % positivos dentro del intervalo intercuartílico de fieles. Es discriminación de señales individuales, no el AUROC del bot.','',
         '| Tipo | AUROC riesgo NLI | AUROC riesgo MiniMax | % NLI en IQR fiel | % MiniMax en IQR fiel |','|---|---:|---:|---:|---:|']
    for category in TYPES:
        cat=np.array([r['tipo']==category for r in records]);mask=cat|(y==0);vals=[];over=[]
        for field in ['s_risk_max','llm_risk_max']:
            vals.append(roc_auc_score(cat[mask].astype(int),s[field][mask]));lo,hi=np.quantile(s[field][y==0],[.25,.75]);over.append(100*np.mean((s[field][cat]>=lo)&(s[field][cat]<=hi)))
        nl.append(f'| {LABELS[category]} | {vals[0]:.3f} | {vals[1]:.3f} | {over[0]:.1f}% | {over[1]:.1f}% |')
    sp=s['s_risk_max']>=.5;fr=s['f_risk_max']>=.5;changed=sp!=fr
    changes=[dict(id=r['id'],tipo=r['tipo'] or 'fiel',etiqueta=r['etiqueta'],risk_sentence=float(s['s_risk_max'][i]),risk_fragment=float(s['f_risk_max'][i]),sentence_pred=int(sp[i]),fragment_pred=int(fr[i])) for i,r in enumerate(records) if changed[i]];write_csv(OUT/'cambios_agregacion.csv',changes)
    nl+=['','## Qué cambia al usar fragmentos','',
         f"Control descriptivo de máximo riesgo NLI a corte fijo 0,5: cambian **{len(changes)}** casos. Pasan de fiel a alerta {int((~sp&fr).sum())}: {int((~sp&fr&(y==1)).sum())} positivos y {int((~sp&fr&(y==0)).sum())} fieles. Pasan de alerta a fiel {int((sp&~fr).sum())}: {int((sp&~fr&(y==1)).sum())} positivos y {int((sp&~fr&(y==0)).sum())} fieles.", '',
         'Esto no es una sustitución ni una calibración del bot: aísla el efecto de granularidad sobre las señales. Los casos concretos están en cambios_agregacion.csv. Un fragmento puede aportar evidencia local o perder sujeto/contexto. No se atribuye causalidad a una señal individual.']
    save('diagnostico_nli_minimax.md',nl)
    # E4 contribution, only as a separate architecture.
    eb=np.array([r['hgb_raw'] for r in records]);et=records[0]['e4_threshold'];ep=np.array([r['e4_raw'] for r in records]);bp=eb>=et;activation={k:np.array([r['relations'][k]>0 for r in records]) for k in ['pair_bad','source_math_bad','binding_bad']};e4_rows=[]
    for domain,scope in [('todos',np.ones(len(y),dtype=bool)),('H',h),('no-H',~h)]:
        base_metric=metric(y[scope],eb[scope],et)
        for key,active in list(activation.items())+[('union_max',np.logical_or.reduce(list(activation.values())))]:
            raw=np.round(np.maximum(eb,np.where(active,.99,0.)),12);m=metric(y[scope],raw[scope],et);introduced=scope&(y==0)&~bp&active;corrected=scope&(y==1)&~bp&active
            e4_rows.append(dict(dominio=domain,verificador=key,activaciones=int((scope&active).sum()),alucinadas_activadas=int((scope&active&(y==1)).sum()),fieles_activados=int((scope&active&(y==0)).sum()),fp_introducidos=int(introduced.sum()),fn_corregidos=int(corrected.sum()),delta_coste=m['cost']-base_metric['cost'],delta_score=m['score']-base_metric['score'],**{cat:int((active&scope&np.array([r['tipo']==cat for r in records])).sum()) for cat in TYPES}))
    write_csv(OUT/'contribucion_e4.csv',e4_rows)
    er=['# Diagnóstico E4: referencia separada del bot Joseca','',
        '**Joseca no contiene pair_bad, source_math_bad, binding_bad ni max(HGB,relacional).** Esos componentes pertenecen a `entregas/candela_v1/mi_bot.py`, cuyo SHA-256 coincide exactamente con el candidato E4 corregido de fase 3. No se le atribuyen sus mejoras al bot solicitado. En Joseca las comprobaciones numéricas/matices son características aprendidas de una logística, no un veto max.','',
        'Se reentrena el HGB de Candela dentro de cada fold de desarrollo, con sus parámetros originales (120 árboles, 7 hojas, mínimo 20, learning rate 0,06, L2=5, peso no-H=0,35). Después se aplica el verificador fijo a validación. El HGB reproduce score H OOF 0,3092369478, comprobación adicional de la reconstrucción de particiones.','',
        '| Dominio | Verificador | Activa | Alucinadas | Fieles | FP nuevos | FN corregidos | ΔCoste | ΔScore |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in e4_rows:er.append(f"| {r['dominio']} | {r['verificador']} | {r['activaciones']} | {r['alucinadas_activadas']} | {r['fieles_activados']} | {r['fp_introducidos']} | {r['fn_corregidos']} | {r['delta_coste']:+d} | {r['delta_score']:+.4f} |")
    er+=['','Las filas individuales se solapan potencialmente: el efecto conjunto se calcula por la unión de casos, nunca sumando alarmas repetidas. Las categorías detectadas por cada verificador están en contribucion_e4.csv; las que tienen cero activaciones son cobertura no demostrada, no tipos que el HGB deje necesariamente de detectar.','',
         '## Solapamientos','', '| Par de verificadores | Intersección | Unión | Jaccard |','|---|---:|---:|---:|']
    for a,b in itertools.combinations(activation,2):
        inter=int((activation[a]&activation[b]).sum());union=int((activation[a]|activation[b]).sum());er.append(f'| {a} / {b} | {inter} | {union} | {inter/max(1,union):.3f} |')
    er+=['','## Cobertura de comprobaciones y unidades','', '| Verificador | Casos examinados | Casos con fallo |','|---|---:|---:|']
    for key in ['pair','source_math','binding']:er.append(f"| {key} | {sum(r['relations'][key+'_checked']>0 for r in records)} | {sum(r['relations'][key+'_bad']>0 for r in records)} |")
    unit_counter=collections.Counter(q['unit'] for c in cases for q in e4.sci_quantities(c['respuesta']));scaled=sum(any(q['scale']!=1 for q in e4.sci_quantities(c['respuesta'])) for c in cases)
    union=np.logical_or.reduce(list(activation.values()));joseca_overlap=int((union&fn).sum())
    er+=['',f'Unidades extraídas en respuestas (conteos de cantidades, no de casos): {dict(unit_counter)}. Casos con algún factor de normalización distinto de uno: {scaled}.', '',
         f"De los FN reales del Joseca OOF, **{joseca_overlap}** coinciden con alguna alerta de E4. Es solo cobertura potencial: no se implementó ni evaluó una nueva combinación Joseca+E4 y no se asume que el balance global de FP sería favorable.", '',
         'La normalización convierte céntimos, escalas, distancias/masas y separa meses/tasas; se abstiene ante números escritos no interpretados. No se declara verificada una conversión solamente porque su unidad se haya normalizado. No se desactiva ni modifica la normalización del bot para presentar una ganancia nueva. Los falsos positivos atribuibles al max son exactamente los FP nuevos de la tabla. La ausencia de activaciones en binding implica que su utilidad empírica en este lote es nula.']
    save('diagnostico_e4.md',er)
    # FP patterns and component contribution.
    patterns={
      'Síntesis correcta multifuente (proxy)':(s['s_e_any_min']<.5)&((s['s_e_all_min']>.5)|(s['f_e_any_min']>.5)),
      'Paráfrasis / vocabulario nuevo (proxy)':np.array([f.get('palabras_nuevas_frac',0)>=.15 for f in features]),
      'Aritmética válida según comprobador (proxy)':np.array([f.get('num_cuenta_correcta',0)>0 for f in features]),
      'Posible conversión de unidades':np.array([bool(re.search(r'\b(km|kilometros?|centimos?|toneladas?)\b',e4.normalize(c['respuesta']))) and bool(re.search(r'\b(metros?|euros?|kg|kilogramos?)\b',e4.normalize(' '.join(c['fuentes'])))) for c in cases]),
      'Contradicción NLI espuria (cualquier fuente)':s['s_c_any_max']>.5,
      'MiniMax pesimista en fiel':s['llm_bad']>0,
      'Bajo respaldo en fuente citada':np.array([f.get('nli_resp_fuente_citada_min',1)<.3 for f in features]),
      'Fragmento bajo, oración alta':(s['f_risk_max']>.7)&(s['s_risk_max']<.2),
      'Sospecha numérica del Joseca':np.array([f.get('num_cuenta_falsa',0)>0 or f.get('num_sospechosos_grandes',0)>0 for f in features]),
      'Alerta E4 (no componente Joseca)':union,
    }
    fp_rows=[]
    for name,mask in patterns.items():fp_rows.append(dict(patron=name,fieles_cohorte=int((mask&(y==0)).sum()),fp=int((mask&fp).sum()),coste_fp=int((mask&fp).sum()),fraccion_fp=float((mask&fp).sum()/max(1,fp.sum()))))
    write_csv(OUT/'patrones_fp.csv',fp_rows)
    fr=['# Falsos positivos del Joseca','',f"Hay **{int(fp.sum())}** respuestas fieles clasificadas como alucinadas; coste **{int(fp.sum())}**, {100*fp.sum()/max(1,totalcost):.1f}% del coste total. No son una quinta categoría de alucinación.",'',
        'Los patrones se calculan solo dentro de fieles y se solapan. No se suman sus costes para obtener el total. Paráfrasis, conversión y síntesis son proxies observables; no nuevas anotaciones manuales.','',
        '| Patrón | Fieles en cohorte | FP | Coste FP | % de todos los FP |','|---|---:|---:|---:|---:|']
    for r in fp_rows:fr.append(f"| {r['patron']} | {r['fieles_cohorte']} | {r['fp']} | {r['coste_fp']} | {100*r['fraccion_fp']:.1f}% |")
    fr+=['','## Contribuciones del modelo lineal','',
         'Se agrupan las contribuciones al logit centradas por la media del entrenamiento de cada fold. La familia de mayor contribución no prueba causalidad: hay correlaciones y cruces entre características. Puede ser negativa si todas lo son; el intercepto y el corte también determinan la decisión.','',
         '| Familia dominante entre FP | Casos |','|---|---:|']
    dominant=collections.Counter(max(records[i]['logit_contributions'],key=records[i]['logit_contributions'].get) for i in np.flatnonzero(fp))
    for family,n in dominant.most_common():fr.append(f'| {family} | {n} |')
    fr+=['','## Ablación diagnóstica de señales a cero, sin reentrenar','',
         'Se ponen a cero las características de una familia y sus cruces manteniendo coeficientes y corte. No se modifica el bot ni se propone desplegar esta ablación. Es una intervención descriptiva sobre variables correlacionadas y no una estimación de un modelo reentrenado.','',
         '| Familia a cero | FP | FN | ΔFP | ΔFN | ΔCoste |','|---|---:|---:|---:|---:|---:|']
    bm=res['joseca_oof']['todos']
    for family,values in res['feature_zero_ablation'].items():
        m=values['todos'];fr.append(f"| {family} | {m['fp']} | {m['fn']} | {m['fp']-bm['fp']:+d} | {m['fn']-bm['fn']:+d} | {m['cost']-bm['cost']:+d} |")
    fr+=['','## Ejemplos FP más confiados (solo desarrollo)','']
    for i in sorted(np.flatnonzero(fp),key=lambda i:-p[i])[:12]:fr.append(f"- {records[i]['id']} (fold {records[i]['fold']}, p={p[i]:.3f}): {cases[i]['respuesta']}")
    fr+=['','Las falsas alarmas E4 se analizan aparte; Joseca no ejecuta esos verificadores. Ni una cifra sospechosa ni un matiz presente en otra fuente demuestran que esa característica haya causado el FP: el score surge de la suma de todas las señales.']
    save('falsos_positivos.md',fr)
    # Calibration: fixed cutoff and strictly internal threshold diagnostics.
    ca=['# Calibración y umbral — diagnóstico sin cambios','',
        'Corte del bot Joseca: p ≥0,15, reescalado monótonamente a score ≥0,5. Se conserva sin cambios. Los coeficientes y la normalización se reajustan dentro de cada fold; no se aplican los pesos aprendidos con todo train para construir OOF.','',
        'El código de entrenamiento original eligió configuración y un único corte sobre OOF H repetido con semillas 0,1,2. Ese proceso tiene optimismo de selección. Aquí se identifica la configuración ya existente mediante igualdad de parámetros, sin elegir una nueva. Para evaluar otro corte de forma diagnóstica se usa calibración anidada: cuatro folds externos y tres internos, con el mismo agrupamiento.','',*header]
    for key,label in [('joseca_oof','Corte actual 0,15'),('joseca_nested_official','Corte interno, score oficial global'),('joseca_nested_weighted','Corte interno, objetivo 0,7/0,3'),('joseca_nested_H','Corte interno, criterio H original')]:
        for domain in ['H','no-H','todos']:ca.append(table_metric(label+' / '+domain,res[key][domain]))
    ca+=['','## Cortes por fold y rendimiento externo','', '| Fold | Corte actual | Corte interno 0,7/0,3 | Corte interno H | Objetivo al corte actual | Objetivo al corte interno |','|---|---:|---:|---:|---:|---:|']
    for f in folds:ca.append(f"| {f['fold']} | {f['raw_threshold_current']:.3f} | {f['raw_threshold_nested_weighted']:.3f} | {f['raw_threshold_nested_H']:.3f} | {f['current']['objective']:.4f} | {f['nested']['objective']:.4f} |")
    ca+=['','| Fold | Corte elegido solo con score oficial interno | Score global externo |','|---|---:|---:|']
    for f in folds:ca.append(f"| {f['fold']} | {f['raw_threshold_nested_official']:.3f} | {f['nested_official']['todos']['score']:.4f} |")
    ca+=['','Se muestran tres criterios de selección interna: score oficial global, criterio histórico H y objetivo interno 0,7 H + 0,3 no-H. El último no pertenece a la métrica oficial. Los tres usan exactamente los mismos modelos y predicciones internas; se informa de todos, sin elegir retrospectivamente el que gane en los folds externos.']
    ca+=['','## Incertidumbre por escenarios','', '| Protocolo | IC95% score H | IC95% score no-H | IC95% objetivo |','|---|---|---|---|']
    for key,label in [('bootstrap_current','Corte actual'),('bootstrap_official','Corte interno oficial'),('bootstrap_nested','Corte interno ponderado')]:
        b=res[key];ca.append('| '+label+' | '+' | '.join(f'[{b[d][0]:.4f},{b[d][1]:.4f}]' for d in ['H','no-H','objective'])+' |')
    best=res['descriptive_global_best'];delta=res['joseca_nested_weighted']['objective']-res['joseca_oof']['objective'];ca+=['',
         f"La mejor rejilla retrospectiva en el OOF completo es {best['weighted_threshold']:.2f} para el objetivo ponderado y {best['h_threshold']:.2f} para H. Son valores descriptivos optimistas, no una recomendación ni cortes adoptados. La calibración anidada cambia el objetivo en {delta:+.4f} respecto al corte actual.", '',
         'Curva completa en `curva_coste_oof.csv`: contiene FP, FN y coste H/no-H/global para todos los cortes de la rejilla original 0,02–0,79. La figura `curva_coste.svg` muestra coste normalizado ponderado y el corte actual. No se ajusta con leaderboard, test o giro.','',
         'Interpretación: comparar AUROC con score permite separar ordenación y corte; el AUROC no cambia al recolocar un único corte global. Los cortes anidados variables por fold se comparan como decisiones externas, no se confunden con una transformación monótona global. La diferencia actual/anidado no aísla causalmente toda la selección previa de variables o hiperparámetros. El bot ya utilizó todo train al entrenarse; ninguna sección de este fichero constituye una nueva reserva ciega.']
    paired=res['bootstrap_paired_delta'];ca+=['','## Valoración del corte actual','',
         'No hay evidencia para sustituir 0,15 por un corte mayor: la calibración anidada reduce el coste global de 405 a 385, pero empeora H (183 a 200) y el objetivo interno ponderado de 0,3016 a 0,2844. La mejor rejilla retrospectiva queda cerca, en 0,16, y no es validación independiente. El resultado sugiere estudiar las falsas alarmas sistemáticas antes de atribuir el problema exclusivamente al umbral. No demuestra que 0,15 sea óptimo en el test público.','',
         '| Diferencia de score anidado − actual | IC95% bootstrap pareado por grupos |','|---|---|']
    for domain in ['H','no-H','todos','objective']:
        low,high=paired[domain];ca.append(f'| {domain} | [{low:+.4f},{high:+.4f}] |')
    mo=res['joseca_nested_official'];mc=res['joseca_oof'];ci=res['bootstrap_paired_official_delta']['todos']
    ca+=['',f"Con selección interna de la métrica oficial global: score {mc['todos']['score']:.4f} → {mo['todos']['score']:.4f}; coste {mc['todos']['cost']} → {mo['todos']['cost']}; FP {mc['todos']['fp']} → {mo['todos']['fp']}; FN {mc['todos']['fn']} → {mo['todos']['fn']}. Diferencia global {mo['todos']['score']-mc['todos']['score']:+.4f}, IC95% pareado [{ci[0]:+.4f},{ci[1]:+.4f}]. Score H {mc['H']['score']:.4f} → {mo['H']['score']:.4f}; no-H {mc['no-H']['score']:.4f} → {mo['no-H']['score']:.4f}.", '',
         'El balance depende de la mezcla de dominios. Estos son cortes variables elegidos dentro de cada entrenamiento y evaluados fuera; no identifican un único nuevo corte para el bot. El umbral actual es defendible para H frente a los criterios evaluados, pero no se ha demostrado óptimo. La calidad de ordenación en H (AUROC 0,860 frente a 0,966 no-H) y los patrones de falsas alarmas también limitan la puntuación.']
    ca+=['','2.000 remuestreos de grupos completos con las mismas selecciones para ambos métodos. Los intervalos condicionan las predicciones OOF existentes y no incorporan toda la incertidumbre de selección de arquitectura. No se afirma superioridad estadística por comparar solamente los promedios.']
    save('calibracion_y_umbral.md',ca)
    # Standalone SVG; stdlib only, no visualization dependencies in submitted bot.
    curve_rows=list(csv_reader(OUT/'curva_coste_oof.csv'));xx=[float(r['threshold']) for r in curve_rows];yy=[1-float(r['objective']) for r in curve_rows];lo=min(yy);hi=max(yy);width=900;height=340
    sx=lambda v:60+(v-.02)/.77*800;sy=lambda v:270-(v-lo)/max(.001,hi-lo)*220
    pts=' '.join(f'{sx(x):.2f},{sy(v):.2f}' for x,v in zip(xx,yy))
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="340" viewBox="0 0 900 340"><rect width="900" height="340" fill="white"/><text x="60" y="25" font-family="sans-serif" font-size="16">Coste ponderado OOF descriptivo (menor es mejor)</text><path d="M60 50 V270 H860" fill="none" stroke="#444"/><polyline points="{pts}" fill="none" stroke="#2463a6" stroke-width="2"/><path d="M{sx(t):.2f} 50 V270" stroke="#c33" stroke-dasharray="5 4"/><text x="{sx(t)+5:.2f}" y="70" font-family="sans-serif" font-size="13">corte actual 0,15</text><text x="60" y="292" font-family="sans-serif" font-size="12">0,02</text><text x="825" y="292" font-family="sans-serif" font-size="12">0,79</text><text x="60" y="315" font-family="sans-serif" font-size="12">Coste normalizado = 0,7 C_H/N0_H + 0,3 C_noH/N0_noH; no usar como validación independiente</text></svg>'
    (OUT/'curva_coste.svg').write_text(svg+'\n')
    # Recommendations: no implementation. At most two hypotheses per category.
    recommendations={
      'distorsion':[
        ('Interpretación aritmética y temporal con cobertura explícita','Separar suma, diferencia, porcentaje, fechas y cambio de unidad; contrastar resultados solo cuando la relación es inequívoca.','Extractor numérico; investigar signos y cruces aprendidos en el régimen MiniMax favorable.','Ablación por familia de operación y controles fieles de conversiones, tasas y redondeos.'),
        ('Atribución de cantidades al atributo y sujeto correctos','Restringir operandos por evidencia de atributo/entidad y unidad, con abstención si hay varias lecturas.','Comprobaciones numéricas y cruces llmok; no un veto de cifra ausente.','Comparar extracción local frente a global, retirando solo esta restricción; medir ΔFP y ΔFN por distorsión.')],
      'contradiccion':[
        ('Contradicción en evidencia relevante','Combinar contradicción con la fuente que comparte entidad/atributo, sin maximizar sobre distractores.','Características NLI/alineamiento y llm_contra_sin_apoyo.','Retirar solo la alineación y mantener el resto; comparar contradicción real frente a FP NLI.'),
        ('Oposición direccional y de relaciones','Comprobar pares verbales opuestos en cláusulas alineadas; abstenerse ante negación o sujeto ambiguo.','Verificador textual limitado, candidato futuro.','Separar casos con/sin negación y evaluar pares fieles con otros sujetos antes de integrar.')],
      'no_respaldado':[
        ('Detalle local aceptado por MiniMax','Conservar relación fragmento→oración y comprobar soporte del detalle en la evidencia local citada.','Señales de detalle y desacuerdo; no confiar únicamente en confianza MiniMax.','Ablación por granularidad y por respaldo conjunto para comprobar que no se castiga síntesis fiel.'),
        ('Causa o consecuencia sin evidencia','Vincular conector causal a la relación que expresa y sus fuentes, en vez de buscar cualquier conector en el documento.','anadido_causal y sus cruces.','Retirar componente causal; medir FN no respaldados y FP en causas que sí aparecen para el sujeto correcto.')],
      'sobregeneralizacion':[
        ('Alcance aceptado por MiniMax y señales próximas a fieles','Representar cuantificadores y modalidades sobre el mismo sujeto/intervalo, sin vetar palabras universales aisladas.','Características de alcance/modalidad.','Retirar modalidad o cuantificadores por separado; incluir fieles con expresiones absolutas justificadas.'),
        ('Condición o excepción eliminada','Alinear el atributo con su restricción local; distinguir omisión irrelevante de endurecimiento de una afirmación.','matiz_perdido_tramo/fuente y selección de evidencia.','Comparar tramo contra documento completo con mismo clasificador y cortes internos.')],
    }
    pr=['# Prioridades de mejora — propuestas, no implementadas','',
        'Orden por coste de FN del Joseca OOF de desarrollo al corte actual. La estabilidad y el riesgo de FP se usan para orientar las pruebas, no para declarar que una propuesta funcionará. Los pesos de categorías siguen siendo iguales.','',
        '| Prioridad | Tipo | Problema principal | FN actuales | Coste | Mejora propuesta | Riesgo de FP |','|---|---|---|---:|---:|---|---|']
    for order,category in enumerate(ranked,1):
        row=by_type[category];pr.append(f"| {order} | {LABELS[category]} | {recommendations[category][0][0]} | {row['fn']} | {row['coste_fn']} | {recommendations[category][0][1]} | Alto si falta alineamiento o se castigan paráfrasis |")
    pr+=['',f"Prioridad transversal: las **{int(fp.sum())} falsas alarmas** cuestan {int(fp.sum())}; antes de añadir nuevas alarmas debe comprobarse que cada FN corregido no produce más de cuatro FP, y que mejora el objetivo ponderado sin deteriorar H. Esta comparación de costes no justifica tratar FP como tipo de alucinación.", '',
         'Las propuestas se vinculan a mecanismos observados, no a mejoras ya demostradas. La referencia E4 solo coincide con '+str(joseca_overlap)+' FN del Joseca: su ganancia sobre otro HGB no puede trasladarse automáticamente.', '']
    pr+=['En los empates de coste no hay una prioridad estadística demostrada. Distorsión concentra sus siete FN en dos folds, sobregeneralización en tres y contradicción en uno; consultar estabilidad_por_tipo.csv. Las cohortes de atribución cruzada, causa añadida y condición perdida detectada tienen cero FN: las propuestas sobre esos mecanismos son exploratorias y tienen menor respaldo que las basadas en FN observados.','',
         'Primer experimento transversal propuesto, aún sin implementar: reentrenar ablaciones de palabras nuevas, alineamiento citado y matices, una familia cada vez, con corte interno y mismo protocolo agrupado. Evidencia: 151 FP coinciden con vocabulario nuevo, 184 con poco respaldo citado y 116 con síntesis multifuente (cohortes solapadas). La intervención a cero de matices quita 37 FP pero añade 13 FN y empeora coste en 15: no justifica suprimirla. Hipótesis: mejorar la alineación y el trato de paráfrasis conserva sensibilidad con menos falsas alarmas. Riesgo: perder FN que dependen de esas mismas señales. Esfuerzo: bajo para la ablación, medio para cambiar alineamiento; cualquier aceptación requiere mejora pareada estable y futura evidencia no inspeccionada.','']
    for category in ranked:
        mask=np.array([r['tipo']==category for r in records]);bad=mask&fn;accepted=int((bad&(s['llm_all_supported']>0)).sum());nlibad=int((bad&(s['s_risk_max']>.5)).sum())
        pr+=['## '+LABELS[category],'',f"Evidencia de diagnóstico: {int(bad.sum())} FN; MiniMax respalda todas las oraciones en {accepted}; riesgo NLI de oración >0,5 en {nlibad}. Distribuciones, cohortes y ejemplos están en diagnostico_tipos.md y diagnostico_nli_minimax.md.",'']
        evidence={
            'distorsion':'Tres FN contienen un año y dos pertenecen a la cohorte de cuenta sospechosa (pueden solaparse); la atribución cruzada detectada tiene cero FN. Las operaciones/fechas tienen mayor respaldo como primera investigación. Esfuerzo medio; atribución completa alto.',
            'sobregeneralizacion':'NLI por sí solo separa poco esta categoría de fieles (AUROC 0,608) y MiniMax respalda 54 de 97 positivos; los siete FN reciben su respaldo. La cohorte de condición perdida ya detectada tiene cero FN, así que ampliar condiciones es una hipótesis de cobertura, no una causa comprobada. Esfuerzo medio para alcance y alto para alineación de excepciones.',
            'no_respaldado':'Los dos FN son aceptados por MiniMax; no hay FN en la cohorte de causa añadida detectada. Priorizar soporte del detalle antes de nuevas reglas causales. Esfuerzo medio; evidencia limitada a dos errores.',
            'contradiccion':'Ambos FN contienen un verbo direccional y están en el mismo fold. Hay 111 FP con contradicción NLI en alguna fuente: maximizar contradicción indiscriminadamente tiene riesgo alto. Esfuerzo medio para verbos, alto para alineación robusta.'}
        pr += [evidence[category],'']
        for j,(hypothesis,mechanism,component,ablation) in enumerate(recommendations[category],1):
            pr += [f'{j}. **Hipótesis: {hypothesis}.** Mecanismo: {mechanism} Componente: {component} Ablación necesaria: {ablation} Validación: conservar grupos/semilla, ajustar el corte solo dentro de cada entrenamiento, comparar coste por fold e IC bootstrap pareado; exigir estabilidad y posteriormente contrastar en datos no inspeccionados. Riesgo: falsa atribución por paráfrasis, unidades o fuentes distractoras. No utilizar la reserva histórica ni el leaderboard para aceptar reglas.','']
    pr+=['## Qué no hacer todavía','',
         'No desplegar las intervenciones de características a cero; sus pesos están correlacionados y no equivalen a modelos reentrenados. No endurecer el corte por observar FN aislados. No añadir el E4 de Candela al Joseca sin un experimento propio. No usar longitud, prefijos o categorías como predictores. Las ideas de granularidad, alineamiento y complementariedad son inspiraciones funcionales de fases anteriores, no reproducciones de modelos neuronales.']
    save('prioridades_mejora.md',pr)
    # README distinguishes all evaluation settings and historical evidence.
    rd=['# Fase 4 — diagnóstico de entregas/joseca_v1','',
        '**No se modifica ningún bot, parámetro ni umbral. No se genera una entrega.**','',
        '## Versión y procedencia','',
        '- Objetivo: `entregas/joseca_v1/mi_bot.py`, SHA-256 `'+res['provenance']['sha256']+'`.',
        '- El archivo coincide byte por byte con `b6e5a97:kit/mi_bot.py`. El mensaje de ese commit registra entrega 1 con score público 0,353 y AUROC 0,905. Esto identifica la versión dentro del repositorio; no sustituye un recibo del servidor ni permite recalcular métricas públicas sin etiquetas.',
        '- Es una regresión logística de 60 características, no E4. Se recuperó de manera única peso H=5 y L2=10: entrenar con todos los casos y redondear sus parámetros a seis decimales reproduce exactamente MEDIAS/ESCALAS/PESOS/SESGO embebidos. La comparación de nueve configuraciones fue una auditoría de identidad, no selección de un modelo nuevo.',
        '- E4 de referencia: `entregas/candela_v1/mi_bot.py`, SHA-256 `'+res['provenance']['e4_sha256']+'`; coincide con el candidato corregido de fase 3. Se analiza por separado.',
        '- En el checkout actual no existen `kit/mi_bot.py`, los informes anteriores ni los manifiestos de particiones originales. No se crean ni se recuperan en rutas de entrega.','',
        '## Reconstrucción y límites de OOF','',
        'Se reconstruyó el algoritmo histórico: componentes conectados por ID base, pregunta normalizada, fuentes idénticas o solape >=80% del conjunto menor; semilla 20261015, StratifiedGroupKFold de cuatro particiones. Resultan 337 grupos, desarrollo de 1.280 casos (473 H) y reserva histórica de 427 (158 H). Desarrollo usa otros cuatro folds con semilla 20261016. No hay grupos compartidos. La coincidencia de tamaños y del score H OOF del HGB histórico sirve de comprobación, pero no sustituye una comparación bit a bit con el manifiesto original ausente.','',
        'Para Joseca se reentrenan estándarización, coeficientes e intercepto en cada entrenamiento. No se usan sus pesos congelados de todo train para producir OOF. Se fijan las mismas 60 características y regularización ya identificadas. La calibración anidada genera tres particiones internas por fold externo, semilla 20261017 + fold. No se seleccionan hiperparámetros nuevos.','',
        'Las características y la configuración actuales se eligieron originalmente con train: aunque los coeficientes OOF no ven su grupo, sigue existiendo sesgo de selección de arquitectura/variables. La evaluación no es una nueva prueba ciega.','',
        '## Cuatro contextos que no se deben mezclar','',*header]
    for label,key in [('Joseca OOF desarrollo','joseca_oof'),('Joseca pesos congelados / train (in-sample)','joseca_frozen_train'),('Joseca pesos congelados / reserva histórica (también entrenada)','joseca_frozen_historical')]:rd.append(table_metric(label,res[key]['todos']))
    rd+=['',
         'La reserva histórica ya forma parte del entrenamiento del Joseca publicado (ver reproducción exacta de sus parámetros con todo train), por lo que sus métricas congeladas NO son independientes. Para Candela, el HGB sí procedía de desarrollo, pero esa reserva fue inspeccionada en fases anteriores: aquí sus cifras congeladas son solo referencia histórica, no selección. El leaderboard 0,353/0,905 corresponde a otro conjunto sin etiquetas locales; no se compara como si fueran los mismos ejemplos.','',
         '## Reproducción adicional del protocolo original del autor','',
         'Se fija peso H=5/L2=10 y se repiten las semillas originales 0,1,2 sobre familias H, sin buscar configuración ni corte. Todo no-H entrena siempre, así que este protocolo produce OOF solamente para H y no se le atribuye AUROC OOF no-H.','',
         '| Semilla | N H | Score al corte 0,15 | AUROC | FP | FN |','|---|---:|---:|---:|---:|---:|']
    for r in res['author_repeated_h']:rd.append(f"| {r['seed']} | {r['n']} | {r['score']:.4f} | {r['auroc']:.4f} | {r['fp']} | {r['fn']} |")
    rd+=['','## Archivos','',
         '- `diagnostico_tipos.md` y `metricas_por_tipo.csv`: ranking, TP/FN/recall/coste por tipo y dominio; FP separados.',
         '- `diagnostico_nli_minimax.md` y `distribuciones_senales.csv`: distribuciones, desacuerdos y errores de jueces.',
         '- `diagnostico_e4.md` y `contribucion_e4.csv`: contribución del E4 de Candela, claramente separado de Joseca.',
         '- `falsos_positivos.md`: cohortes y contribuciones del Joseca.',
         '- `calibracion_y_umbral.md`, `curva_coste_oof.csv`, `curva_coste.svg`: umbral actual y calibración interna, sin modificaciones.',
         '- `prioridades_mejora.md`: propuestas futuras; ninguna implementada.',
         '- `predicciones_oof.json/.csv`, `folds_detalle.json`, `particiones_reconstruidas.json`: datos para auditar cada tabla.',
         '- `procedencia.json`, `hashes_antes.json`, `integridad.json`: identificación e integridad.','',
         '## Ejecutar','',
         'Desde la raíz del repositorio, con NumPy y scikit-learn instalados (versiones exactas en resumen_metricas.json → environment). En esta máquina se usó /private/tmp/guardian-venv/bin/python; puede sustituirse python por la ruta de ese entorno:','',
         '```bash','python kit/experimentos/fase4_diagnostico/ejecutar_diagnostico.py','```','',
         'verificar_diagnostico.py comprueba las métricas con el evaluador oficial, reconstruye cada score con sus coeficientes de fold y verifica separación de grupos externa/interna, cobertura una vez por caso y hashes. Los resultados se guardan en verificacion.json.','',
         'No se llama al main de entrenar.py, que podría sobrescribir pesos. Se importan sus funciones puras de entrenamiento. Los cachés numéricos del diagnóstico no son archivos necesarios para la inferencia del bot. Los datos oficiales y ambos bots se verifican por hash al finalizar. No se consulta ninguna API ni se usan modelos de IA adicionales.']
    save('README.md',rd)
    errors=[dict(id=r['id'],fold=r['fold'],tipo=r['tipo'] or 'fiel',error='FP' if fp[i] else 'FN',probabilidad=float(p[i]),pregunta=cases[i]['pregunta'],respuesta=cases[i]['respuesta']) for i,r in enumerate(records) if fp[i] or fn[i]]
    write_csv(OUT/'errores_oof.csv',errors)
    (OUT/'resumen_diagnostico.json').write_text(json.dumps(dict(ranking=[by_type[c] for c in ranked],fp_cost=int(fp.sum()),total_cost=totalcost,minimax=judge_rows,e4_rows=e4_rows,joseca_fn_cubiertos_por_e4=joseca_overlap,calibration_delta_objective=delta,calibration_delta_official_global=mo['todos']['score']-mc['todos']['score'],calibration_delta_official_global_ci=ci),ensure_ascii=False,indent=2)+'\n')


def csv_reader(path):
    import csv
    with path.open() as f:yield from csv.DictReader(f)

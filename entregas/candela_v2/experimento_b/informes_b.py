"""Informes reproducibles: comparaciones de casos, categorías y robustez."""
import collections
import csv
import json
from pathlib import Path
import re
import numpy as np
from experimentar import OUT,ROOT,PRIMARY,TYPES,load,clean,metric,metrics,compare,cohort_masks,write_csv,save
from caracteristicas_b import B1,B2,B3,B4,clauses,norm,roots


def md(name,lines):(OUT/name).write_text('\n'.join(lines)+'\n')


def generate():
    data=json.loads((OUT/'datos_diagnostico.json').read_text());runs=json.loads((OUT/'predicciones_repetidas.json').read_text());folds=json.loads((OUT/'folds.json').read_text());audit=json.loads((OUT/'auditoria_baseline.json').read_text());protocol=json.loads((OUT/'protocolo.json').read_text())
    allcases=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()];byid={c['id']:c for c in allcases};cases=[byid[i] for i in data['ids']];n=len(cases)
    y=np.array([c['etiqueta'] for c in cases]);h=np.array([c['id'].startswith('H') for c in cases]);multi=np.array([len(c['senales']['oraciones'])>1 for c in cases]);features=data['base_features'];extra=data['extra_features'];cohorts=cohort_masks(features)
    base=runs['0']['B0'];bp=np.array(base['p']);bt=np.array(base['threshold']);bold=bp>=bt;fp=h&(y==0)&bold;fn=h&(y==1)&~bold;types=np.array([c.get('tipo') for c in cases]);g=np.array(data['groups'])
    # Selección explícita y ajustada por comparar siete variantes principales.
    gate=[]
    for key in PRIMARY[1:]:
        r=runs['0'][key];r2=runs['1'][key];fold_deltas=[]
        for k in range(4):
            b=next(f for f in folds if f['repeat']==0 and f['variant']=='B0' and f['fold']==k)
            v=next(f for f in folds if f['repeat']==0 and f['variant']==key and f['fold']==k)
            fold_deltas.append(v['metrics']['H']['score']-b['metrics']['H']['score'])
        p=np.array(r['p']);t=np.array(r['threshold']);sob=h&(types=='sobregeneralizacion');new_sob=int(sum(sob&(p<t)))-int(sum(sob&~bold))
        checks=dict(positive_folds=sum(d>0 for d in fold_deltas)>=3,adjusted_ci_positive=r['bootstrap_paired']['H']['ci95_bonferroni7'][0]>0,repeat_positive=r2['metrics']['H']['score']>runs['1']['B0']['metrics']['H']['score'],auroc_non_decreasing=r['metrics']['H']['auroc']>=base['metrics']['H']['auroc'] and r2['metrics']['H']['auroc']>=runs['1']['B0']['metrics']['H']['auroc'],global_cost_non_increasing=r['metrics']['global']['cost']<=base['metrics']['global']['cost'],multi_cost_non_increasing=r['metrics']['multi']['cost']<=base['metrics']['multi']['cost'],sobregeneralizacion_fn_non_increasing=new_sob<=0)
        gate.append(dict(variant=key,passed=all(checks.values()),checks=checks,fold_deltas=fold_deltas,delta_h=r['metrics']['H']['score']-base['metrics']['H']['score'],delta_auroc_h=r['metrics']['H']['auroc']-base['metrics']['H']['auroc'],delta_sob_fn=new_sob))
    eligible=[r for r in gate if r['passed']];selected=max(eligible,key=lambda r:r['delta_h'])['variant'] if eligible else None
    save('seleccion.json',dict(selected=selected,gate=gate,caveat='Selección exploratoria sobre datos de desarrollo ya inspeccionados. No sustituye validación nueva ni permite prometer una mejora pública.'))
    # Métricas completas, a corte anidado y corte fijo original.
    metric_rows=[];type_rows=[];transitions=[];error_rows=[]
    write_csv('metricas_por_fold.csv',[dict(repeat=f['repeat'],variant=f['variant'],fold=f['fold'],threshold=f['threshold'],domain=domain,**m) for f in folds for domain,m in f['metrics'].items()])
    for repeat,outputs in runs.items():
        for key,r in outputs.items():
            p=np.array(r['p']);t=np.array(r['threshold']);pred=p>=t
            for domain,m in r['metrics'].items():metric_rows.append(dict(repeat=repeat,variant=key,domain=domain,cut='nested_H',**m))
            for domain,m in r['fixed_threshold_metrics'].items():metric_rows.append(dict(repeat=repeat,variant=key,domain=domain,cut='fixed_0.16',**m))
            for domain,mask in [('H',h),('global',np.ones(n,dtype=bool)),('multi',multi)]:
                for cat in TYPES:
                    m=mask&(types==cat);nn=int(sum(m));miss=int(sum(m&~pred));type_rows.append(dict(repeat=repeat,variant=key,domain=domain,tipo=cat,casos=nn,tp=nn-miss,fn=miss,recall=(nn-miss)/nn if nn else '',coste_fn=4*miss))
            if repeat=='0':
                for domain,m in [('H',h),('global',np.ones(n,dtype=bool))]:transitions.append(dict(variant=key,domain=domain,**r['transitions'][domain]))
                for i,c in enumerate(cases):
                    if pred[i]!=bold[i]:error_rows.append(dict(variant=key,id=c['id'],domain='H' if h[i] else 'no-H',tipo=c.get('tipo') or 'fiel',etiqueta=int(y[i]),b0_pred=int(bold[i]),new_pred=int(pred[i]),effect=('FP_nuevo' if pred[i] else 'FP_eliminado') if y[i]==0 else ('FN_eliminado' if pred[i] else 'FN_nuevo'),respuesta=c['respuesta']))
    write_csv('metricas.csv',metric_rows);write_csv('comparativa_por_tipo.csv',type_rows);write_csv('transiciones.csv',transitions);write_csv('cambios_por_caso.csv',error_rows)
    lines=['# Resultados de ablaciones B','',
      'Todas las variantes se reentrenan con GB de 300 árboles, profundidad 3, learning rate 0,03, subsample 0,8, mínimo 20 y peso H=5. Se mantiene la configuración de v3 para aislar las características. El corte se selecciona por score oficial H dentro de tres folds internos; cuatro folds externos evalúan. No se usa el leaderboard.','',
      '| Variante | Score H | Δ H | AUROC H | FP H | FN H | Score global | AUROC global | Score multi |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in runs['0'].items():
        mh=r['metrics']['H'];mg=r['metrics']['global'];mm=r['metrics']['multi'];lines.append(f"| {key} | {mh['score']:.4f} | {mh['score']-base['metrics']['H']['score']:+.4f} | {mh['auroc']:.4f} | {mh['fp']} | {mh['fn']} | {mg['score']:.4f} | {mg['auroc']:.4f} | {mm['score']:.4f} |")
    lines+=['','## Transiciones frente a B0 — H','', '| Variante | FP eliminados | FP nuevos | FN eliminados | FN nuevos | ΔCoste |','|---|---:|---:|---:|---:|---:|']
    for row in transitions:
        if row['domain']=='H':lines.append('| '+row['variant']+' | '+' | '.join(str(row[k]) for k in ['fp_eliminados','fp_nuevos','fn_eliminados','fn_nuevos','delta_coste'])+' |')
    lines+=['','Las transiciones globales están en transiciones.csv; cada caso que cambia está en cambios_por_caso.csv. El coste se comprueba como ΔFP + 4 ΔFN. Las etiquetas de tipo nunca entran en el extractor.','',
      '## Segunda partición agrupada de desarrollo','', '| Variante | Score H | Δ H frente a B0 | AUROC H | ΔAUROC H |','|---|---:|---:|---:|---:|']
    for key,r in runs['1'].items():
        a=r['metrics']['H'];b=runs['1']['B0']['metrics']['H'];lines.append(f"| {key} | {a['score']:.4f} | {a['score']-b['score']:+.4f} | {a['auroc']:.4f} | {a['auroc']-b['auroc']:+.4f} |")
    lines+=['','## Estabilidad, incertidumbre y decisión','', '| Variante | Δ H por fold principal | IC95% ΔH | IC95% simultáneo (7) | Supera controles |','|---|---|---|---|---|']
    for row in gate:
        r=runs['0'][row['variant']];ci=r['bootstrap_paired']['H'];lines.append('| '+row['variant']+' | '+', '.join(f'{v:+.3f}' for v in row['fold_deltas'])+' | '+str([round(v,4) for v in ci['ci95']])+' | '+str([round(v,4) for v in ci['ci95_bonferroni7']])+' | '+str(row['passed'])+' |')
    lines+=['','Criterios fijados en protocolo.json: '+protocol['gate'], '',
      'Bootstrap pareado de 3.000 muestras de grupos completos (semilla 20261015), condicionado a estas predicciones. Los intervalos Bonferroni son un control aproximado por las siete comparaciones principales; las cuatro ablaciones auxiliares son exploratorias y no compiten por exportación. La segunda partición reutiliza casos y no es una prueba independiente.','',
      '## Ablaciones auxiliares','',
      '- B1_norm sustituye únicamente los cuatro recuentos B1 por proporciones sobre las comprobaciones de matiz; las características v3 se conservan.',
      '- B0_sin_NLI retira las características NLI y sus cruces (incluido llm_contra_sin_apoyo); B3_sin_NLI añade cobertura a ese mismo control. Comparar su diferencia con B3−B0 mide complementariedad, sin atribuir causalidad a una sola variable.',
      '- B5_sustituye_matices retira explícitamente matiz_perdido_* y llmok_matiz_* originales y añade B1+B2. El resto de variantes conserva todas las características v3.', '',
      'Los resultados al corte fijo original 0,16 también figuran en metricas.csv. Permiten separar sensibilidad al umbral de cambios de ordenación, sin seleccionar con etiquetas externas.','',
      '## Recomendación','',('Se exporta '+selected+' porque supera todos los controles especificados. Su rendimiento futuro sigue sin estar garantizado.' if selected else '**Conservar B0/v3. Ninguna variante supera todos los controles; no se genera mi_bot_candidato_b.py.**')]
    lines+=['','## Revisión de cobertura B4 y control de columnas constantes','',
       'La primera ejecución se conserva en piloto_sujeto_singular/. B4 no se activaba en ningún caso de desarrollo: cubría singular→colectivo, pero no subconjunto numerado→grupo completo. Sus variaciones de score y las diferencias B6/B7 de aquella ejecución no demostraban valor de B4: añadir columnas constantes cambia la permutación de variables/desempates del algoritmo.','',
       'La revisión añade únicamente el mecanismo de subconjuntos cuantificados, ya solicitado en el encargo, y anclaje de confianza con la pregunta; seis controles sintéticos adicionales prueban fracción estricta, fracción completa, propiedad distinta, restricción preservada y evidencia completa. Se han repetido todas las variantes con el mismo protocolo y criterios. El cambio ocurrió después de inspeccionar el piloto: las conclusiones son exploratorias y los IC simultáneos no corrigen toda esta adaptación. No se ha buscado otra configuración o corte sobre folds externos.']
    # Aporte por familias: no inferir utilidad a partir de importancias o de un
    # solo incremento de score que también puede proceder de desempates.
    def dh(key):return runs['0'][key]['metrics']['H']['score']-base['metrics']['H']['score']
    lines+=['','## Qué aporta valor y qué añade ruido','',
       f"B1: ΔH {dh('B1'):+.4f}; normalizar sus recuentos produce ΔH {dh('B1_norm'):+.4f}. B2: ΔH {dh('B2'):+.4f}; B5 con ambas familias: {dh('B5'):+.4f}. No se declara utilidad estable solo por una mejora puntual.", '',
       f"B3 con NLI: ΔH {dh('B3'):+.4f}; sin NLI, el incremento B3_sin_NLI−B0_sin_NLI es {runs['0']['B3_sin_NLI']['metrics']['H']['score']-runs['0']['B0_sin_NLI']['metrics']['H']['score']:+.4f}. La cobertura no demuestra sustituir las señales NLI. Retirar los matices originales (B5_sustituye_matices) produce ΔH {dh('B5_sustituye_matices'):+.4f}: no se recomienda reemplazarlos sin evidencia adicional."]
    if (OUT/'uso_caracteristicas.json').exists():
        usage=json.loads((OUT/'uso_caracteristicas.json').read_text())['rows']
        lines+=['','Se reconstruyen los 32 modelos externos principales sin cambiar configuración y se comprueba igualdad de sus predicciones OOF. uso_caracteristicas.csv enumera activaciones, divisiones e importancia por variable. Esta importancia es descriptiva y no prueba valor causal.','',
                '| Variante | Familia nueva | Divisiones en cuatro modelos externos |','|---|---|---:|']
        for variant,family in [('B1',B1),('B2',B2),('B3',B3),('B4',B4)]:
            splits=sum(r['splits_across_four_outer_models'] for r in usage if r['variant']==variant and r['feature'] in family)
            lines.append(f'| {variant} | {", ".join(family)} | {splits} |')
        b4_use=[r for r in usage if r['variant']=='B4']
        if b4_use and sum(r['splits_across_four_outer_models'] for r in b4_use)==0:
            lines+=['','**B4 no se utiliza en ninguna división.** La revisión detecta los dos escapes como señal, pero la activación es demasiado escasa para min_samples_leaf=20. Añadirla no demuestra aportar información al clasificador: su variación de score procede de la dinámica de entrenamiento/desempates al cambiar el número de columnas. La comparación B7/B6 tampoco acredita una ganancia de la señal B4. No se reduce min_samples_leaf tras observar estos casos.']
        weak=[r for r in usage if r['variant']=='B2' and r['feature']=='matiz_equivalencia_incierta']
        if weak and weak[0]['splits_across_four_outer_models']==0:lines+=['','matiz_equivalencia_incierta tampoco se utiliza en las divisiones de B2: su cobertura es demasiado escasa para demostrar valor. B1 conserva la señal más prometedora para una investigación futura, pero su AUROC H cae en ambas particiones y sus IC incluyen cero. B2/B3 aportan cambios mixtos; la combinación B7 mejora el promedio de score H, con regresiones de FN y estabilidad insuficiente.']
    md('resultados_ablaciones.md',lines)
    # Auditoría de las cifras previas: no se inventa un manifiesto retrospectivo.
    prior=['# Diagnóstico de falsas alarmas','',
      'Las cifras 39 FP por matices, 70 FP sin sospecha concreta y 2 FN de sobregeneralización son afirmaciones del encargo. No hay un informe/manifiesto con esos IDs en el checkout: fase4_diagnostico corresponde a joseca_v1. No se atribuyen sus errores a v3.','',
      'Se reproduce el protocolo original de v3: cinco folds por ID base, semillas 0,1,2, parámetros congelados y corte publicado 0,16. Se guarda cada predicción en oof_protocolo_original.json. El corte original optimizado sobre esos mismos OOF es retrospectivo y no es el criterio principal de comparación.','',
      '| Semilla original | FP H | FP H con matiz | FP H sin sospecha y NLI citado<0,3 | FN H sobregeneralización |','|---|---:|---:|---:|---:|']
    for row in audit['historical_protocol']:prior.append(f"| {row['seed']} | {row['metrics']['H']['fp']} | {len(row['matices_ids'])} | {len(row['sin_sospecha_ids'])} | {len(row['sobregeneralizacion_fn_ids'])} |")
    prior+=['','Las cohortes son operacionales, no causas anotadas. Matiz = cualquier matiz_perdido_* positivo. Sin sospecha = todas las oraciones MiniMax respaldadas, ninguna sospecha numérica/entidad/nombre/fecha/hora/matiz/alcance/causa y respaldo NLI citado mínimo<0,3; la definición exacta está en cohort_masks(). Los patrones pueden solaparse con síntesis y no se suman como errores distintos. No se cambian filtros para forzar 39 o 70.','',
      '## Cohortes de B0 en el protocolo nuevo','',
      '| Variante | FP matiz B0 corregidos | FP sin sospecha B0 corregidos | FP multifuente B0 corregidos | FP H nuevos | FN H nuevos |','|---|---:|---:|---:|---:|---:|']
    synthesis=np.array([e['ganancia_cobertura_union']>.1 and e['cobertura_union']>=.7 for e in extra]);cohorts['sintesis_multifuente']=synthesis
    cohort_info={name:[cases[i]['id'] for i in np.flatnonzero(fp&mask)] for name,mask in cohorts.items()}
    cohort_rows=[]
    for key,r in runs['0'].items():
        pred=np.array(r['p'])>=np.array(r['threshold']);transition=r['transitions']['H'];counts={name:int(sum(fp&mask&~pred)) for name,mask in cohorts.items()}
        cohort_rows.append(dict(variant=key,**counts,fp_h_nuevos=transition['fp_nuevos'],fn_h_nuevos=transition['fn_nuevos']))
        prior.append(f"| {key} | {counts['matices']} / {len(cohort_info['matices'])} | {counts['sin_sospecha']} / {len(cohort_info['sin_sospecha'])} | {counts['sintesis_multifuente']} / {len(cohort_info['sintesis_multifuente'])} | {transition['fp_nuevos']} | {transition['fn_nuevos']} |")
    write_csv('cohortes_fp.csv',cohort_rows);save('cohortes_ids.json',cohort_info)
    prior+=['','## Seguimiento de la cohorte histórica reconstruida (semilla 0)','',
      'Cambiar de particiones cambia modelos y errores. Solo se puede hablar de corregir un FP si B0 vuelve a fallar en ese mismo caso en los folds nuevos. Se informa de la intersección, sin presentar los denominadores 39/70 como confirmados.','',
      '| Cohorte histórica | Tamaño original reconstruido | Presente en desarrollo | Sigue siendo FP B0 nuevo |','|---|---:|---:|---:|']
    legacy0=audit['historical_protocol'][0];historical_intersections={}
    for label,field in [('matices','matices_ids'),('sin_sospecha','sin_sospecha_ids')]:
        ids=set(legacy0[field]);inside=np.array([c['id'] in ids for c in cases]);confirmed=inside&fp;historical_intersections[label]=confirmed
        prior.append(f'| {label} | {len(ids)} | {int(sum(inside))} | {int(sum(confirmed))} |')
    prior+=['','| Variante | FP históricos matiz, confirmados por B0 nuevo, corregidos | FP históricos sin sospecha, confirmados, corregidos |','|---|---:|---:|']
    for key,r in runs['0'].items():
        pred=np.array(r['p'])>=np.array(r['threshold']);prior.append(f"| {key} | {int(sum(historical_intersections['matices']&~pred))} | {int(sum(historical_intersections['sin_sospecha']&~pred))} |")
    prior+=['','La cobertura léxica es un predictor, nunca una prueba de veracidad ni un descuento del score. Los controles sintéticos incluyen intercambio de entidades con cobertura 1. Las correcciones y regresiones completas se conservan por ID en cambios_por_caso.csv.']
    md('diagnostico_falsas_alarmas.md',prior)
    # Sobregeneralización: los supuestos dos casos pueden no ser recuperables.
    sob=['# Diagnóstico de sobregeneralización','',
      'Se comparan los FN reales de B0 en los folds actuales. Los dos FN citados en el encargo no traen IDs ni un manifiesto verificable; la reproducción original permite mostrar cuántos hay con cada semilla, sin asumir identidad.','',
      '| Variante | Casos H | FN H | Recall H | Coste FN H | FN global | FN H B0 corregidos | FN H nuevos |','|---|---:|---:|---:|---:|---:|---:|---:|']
    sobmask=types=='sobregeneralizacion';sob_rows=[]
    for key,r in runs['0'].items():
        pred=np.array(r['p'])>=np.array(r['threshold']);m=h&sobmask;miss=int(sum(m&~pred));transition=compare(y,np.array(r['p']),np.array(r['threshold']),bp,bt,m);nn=int(sum(m));sob_rows.append(dict(variant=key,n=nn,fn=miss,recall=1-miss/max(1,nn),cost=4*miss,corrected=transition['fn_eliminados'],new=transition['fn_nuevos']))
        sob.append(f"| {key} | {nn} | {miss} | {1-miss/max(1,nn):.4f} | {4*miss} | {int(sum(sobmask&~pred))} | {transition['fn_eliminados']} | {transition['fn_nuevos']} |")
    sob+=['','## Casos que escapan a B0 en desarrollo','', '| ID | Dominio | p B0 | Corte B0 | B4 sujeto | B4 confianza | Variantes que los detectan |','|---|---|---:|---:|---:|---:|---|']
    for i in np.flatnonzero(sobmask&~bold):
        detected=[key for key,r in runs['0'].items() if r['p'][i]>=r['threshold'][i]];sob.append(f"| {cases[i]['id']} | {'H' if h[i] else 'no-H'} | {bp[i]:.4f} | {bt[i]:.2f} | {extra[i]['sujeto_generalizado']:.0f} | {extra[i]['sujeto_generalizado_confianza']:.2f} | {', '.join(detected) or 'ninguna'} |")
    sob+=['','## Seguimiento de FN del protocolo original, semilla 0','', '| ID original reconstruido | En desarrollo | FN B0 actual | Variantes que lo detectan en el nuevo protocolo |','|---|---|---|---|']
    idx={c['id']:i for i,c in enumerate(cases)}
    for caseid in legacy0['sobregeneralizacion_fn_ids']:
        if caseid not in idx:sob.append(f'| {caseid} | no (reserva histórica) | no evaluado OOF B actual | no se usa para elegir |');continue
        i=idx[caseid];detected=[key for key,r in runs['0'].items() if r['p'][i]>=r['threshold'][i]];sob.append(f"| {caseid} | sí | {not bool(bold[i])} | {', '.join(detected) or 'ninguna'} |")
    sob+=['','B4 utiliza coincidencia de acción/propiedad y evidencia de sujeto singular, no el plural aislado. Se abstiene con colectivo respaldado, varios sujetos documentados, nombres enumerados o restricción explícita. La pregunta ayuda a interpretar el caso, pero su plural no demuestra respaldo. Esta heurística no resuelve correferencia general ni todos los cambios de alcance; los tests sintéticos prueban su mecanismo, no su recall en el giro.']
    sob+=['','La revisión B4 añade fracciones estrictas (n de N sujetos o de N sujetos, n...) y restricciones explícitas vinculadas a la misma propiedad. Se distingue el subconjunto del colectivo completo; una fuente que respalda el colectivo completo provoca abstención. Se describe su cobertura real en la tabla, sin tratar los casos del piloto como una prueba ciega.']
    md('diagnostico_sobregeneralizacion.md',sob);save('sobregeneralizacion.json',sob_rows)
    # Robustez: cohortes fijadas por estructura, sin etiquetas de test.
    masks={'una_oracion':~multi,'varias_oraciones':multi,'fuentes_2_3':np.array([2<=len(c['fuentes'])<=3 for c in cases]),'muchas_fuentes_6mas':np.array([len(c['fuentes'])>=6 for c in cases]),'complementarias_2':np.array([e['numero_fuentes_complementarias']==2 for e in extra]),'complementarias_3':np.array([e['numero_fuentes_complementarias']==3 for e in extra]),'enumeraciones':np.array([bool(re.search(r',.+,|\b(?:y|e)\b',c['respuesta'])) for c in cases]),'comparaciones':np.array([bool(re.search(r'\b(?:mas|menos|mayor|menor|frente|mientras)\b',norm(c['respuesta']))) for c in cases]),'condicion_otra_oracion_proxy':multi&np.array([e['matiz_parafrasis_preservado']>0 for e in extra]),'muchos_numeros_10mas':np.array([len(re.findall(r'\d+(?:[.,]\d+)*',' '.join(c['fuentes'])))>=10 for c in cases]),'respuesta_larga_60mas':np.array([len(c['respuesta'].split())>=60 for c in cases])}
    robust=[]
    for key,r in runs['0'].items():
        p=np.array(r['p']);t=np.array(r['threshold'])
        for label,mask in masks.items():
            if mask.any():robust.append(dict(variant=key,cohort=label,**metric(y[mask],p[mask],t[mask])))
    write_csv('robustez.csv',robust)
    general=['# Validación y generalización','',
      'Desarrollo: 1.280 casos, 473 H, 252 componentes de escenario; reserva histórica: 427, excluida de la comparación B. Se reutilizan exactamente los cuatro folds externos del diagnóstico fase 4. Los grupos conectan ID base, pregunta, fuentes idénticas y solape >=80%; así se evita separar escenarios duplicados con distinto ID. Los manifiestos originales de fases 1–3 faltaban y la reconstrucción previa queda como limitación heredada.','',
      'La repetición de estabilidad usa semilla 20261018 y cuatro folds sobre los mismos grupos de desarrollo. Cada fold externo tiene tres internos, semilla 20261017+fold (+100 en la repetición 2). IDs, grupos y tipos solo se usan para particiones, pesos históricos y diagnóstico, nunca como características.','',
      '## Cohortes estructurales','', '| Cohorte | N | Score B0 | AUROC B0 | FP B0 | FN B0 |','|---|---:|---:|---:|---:|---:|']
    for r in robust:
        if r['variant']=='B0':general.append(f"| {r['cohort']} | {r['n']} | {r['score']:.4f} | {r['auroc']:.4f} | {r['fp']} | {r['fn']} |")
    general+=['','Comparación completa de todas las variantes en robustez.csv, incluidos cohortes pequeños que no permiten conclusiones estables. Las enumeraciones, comparaciones y condiciones en otra oración son proxies, no anotaciones exhaustivas. Las cohortes se solapan.','',
      '## Dependencia de longitud y cobertura','', '| Característica | Correlación con nº palabras | % activación una oración | % activación varias oraciones |','|---|---:|---:|---:|']
    words=np.array([len(c['respuesta'].split()) for c in cases]);lengthrows=[]
    for feature in B1+B2+B3+B4:
        v=np.array([e[feature] for e in extra]);corr=float(np.corrcoef(words,v)[0,1]) if np.std(v)>0 else 0.;a=float(np.mean(v[~multi]>0));b=float(np.mean(v[multi]>0));lengthrows.append(dict(feature=feature,correlation_words=corr,activation_single=a,activation_multi=b))
        general.append(f'| {feature} | {corr:+.3f} | {100*a:.1f}% | {100*b:.1f}% |')
    write_csv('longitud_caracteristicas.csv',lengthrows)
    general+=['','B1 limita recuentos a 2 y B2 a 2; B3 normaliza coberturas por raíces únicas y cuenta como máximo tres fuentes; B4 es binaria/confianza. B1_norm comprueba explícitamente sustituir los recuentos de B1 por proporciones. Ninguna correlación prueba por sí sola un atajo ni una mejora.','',
      '## Límites','',
      '- El extractor y la arquitectura v3 ya se diseñaron usando todo train. Reentrenar dentro de folds evita fuga de coeficientes/árboles, pero no borra el sesgo de selección previo.',
      '- La reserva histórica ya fue inspeccionada en fases anteriores y v3 se entrenó con todo train. No se presenta como conjunto ciego ni se usa para seleccionar B.',
      '- Las hipótesis parten de errores ya observados. Dos repartos e intervalos por grupos reducen optimismo, pero no sustituyen datos nuevos; seleccionar entre variantes sigue siendo exploratorio.',
      '- B4 se revisó tras observar cobertura nula en un piloto; se conserva el piloto y se repite el protocolo completo. Esta adaptación añade sesgo de selección y no queda eliminada por Bonferroni entre siete variantes.',
      '- No se han leído ni puntuado test/giro para este experimento, ni se han creado sus etiquetas. Los cohortes de respuestas largas son proxies de robustez, no rendimiento sobre giro.',
      '- Las equivalencias y la selección de cláusulas no resuelven semántica abierta, negación y correferencia en general. Cobertura alta puede coexistir con atribución o relación falsa.']
    md('validacion_generalizacion.md',general)
    # Respuestas explícitas al informe final solicitado.
    summary=dict(selected=selected,baseline_h=base['metrics']['H'],baseline_global=base['metrics']['global'],claimed_counts_verified=False,reconstructed_original_counts=[dict(seed=r['seed'],matices=len(r['matices_ids']),sin_sospecha=len(r['sin_sospecha_ids']),sob_fn=len(r['sobregeneralizacion_fn_ids'])) for r in audit['historical_protocol']],current_cohort_sizes={k:len(v) for k,v in cohort_info.items()},cohort_corrections=cohort_rows,transitions=transitions,gate=gate)
    save('resumen.json',summary)
    rd=['# Experimento B — matices, cobertura y alcance','',
      '**Base: entregas/joseca_v3/mi_bot.py. La versión original y su entrenador se conservan intactos; copia byte a byte en baseline_v3/.**','',
      'SHA-256 v3: `'+audit['bot_sha256']+'`. El reentrenamiento B0 usa la configuración original y reproduce '+str(audit['exact_trees'])+' de 300 árboles exactamente al redondeo original. No hay igualdad completa de árboles; diferencia máxima de probabilidades congeladas/scikit-learn: '+f"{audit['max_frozen_sklearn_difference']:.3g}"+'. No cambia ninguna decisión al corte original en train. La causa de las diferencias numéricas/empates no se ha identificado con certeza; el entorno original no está fijado. La copia congelada conserva la versión exacta. B0 OOF reentrena la misma arquitectura, no reutiliza los árboles congelados. Esta comprobación es sobre entrenamiento, no OOF.','',
      '## Configuraciones','',
      '| ID | Nuevas características |','|---|---|',
      '| B0 | ninguna |','| B1 | matices en cláusulas relevantes |','| B2 | equivalencias fuertes, pérdidas e incertidumbre |','| B3 | cobertura individual/unión y complementariedad |','| B4 | ampliación de sujeto singular a colectivo |','| B5 | B1+B2 |','| B6 | B1+B2+B3 |','| B7 | B1+B2+B3+B4 |','',
      'Las originales se conservan. Cuatro ablaciones auxiliares verifican normalización B1, aporte B3 respecto a NLI y sustitución explícita de matices. No se busca una nueva configuración GB ni se reutilizan árboles v3 con variables adicionales. No se aplica ninguna regla que reduzca automáticamente el riesgo.','',
      '## Resultado y recomendación','',
      ('Se recomienda revisar el candidato '+selected+'; ha superado los controles predefinidos y se exporta como fichero autónomo.' if selected else '**No se recomienda sustituir v3. No se genera mi_bot_candidato_b.py porque ninguna variante supera los controles de estabilidad.**'),'',
      f"B0 OOF agrupado/anidado H: score {base['metrics']['H']['score']:.4f}, AUROC {base['metrics']['H']['auroc']:.4f}, FP {base['metrics']['H']['fp']}, FN {base['metrics']['H']['fn']}. Es una evaluación distinta del protocolo original con corte seleccionado sobre el propio OOF y del leaderboard.",'',
      '## Documentación','',
      '- resultados_ablaciones.md: métricas, transiciones, repetición, IC y criterios de selección.',
      '- diagnostico_falsas_alarmas.md: auditoría 39/70/2, cohortes actuales y seguimiento por ID.',
      '- diagnostico_sobregeneralizacion.md: escapes originales reconstruidos y FN de B0 actual.',
      '- validacion_generalizacion.md: robustez estructural, longitud y limitaciones.',
      '- comparativa_por_tipo.csv, metricas.csv, transiciones.csv y cambios_por_caso.csv: tablas completas.',
      '- predicciones_oof.csv y predicciones_repetidas.json: scores fuera de fold; folds.json incluye particiones internas/externas.',
      '- protocolo.json: configuración, hashes y versiones; seleccion.json: controles; integridad.json y verificacion.json: comprobaciones.',
      '- baseline_v3/: copia original inalterada; caracteristicas_b.py: funciones autónomas; test_caracteristicas_b.py: controles sintéticos.','',
      '## Reproducir','',
      'Se utilizó /private/tmp/guardian-venv/bin/python, con NumPy '+protocol['environment']['numpy']+' y scikit-learn '+protocol['environment']['sklearn']+'. Sustituir python por ese intérprete si corresponde:','',
      '```bash','python -m unittest discover -s kit/experimentos/propuesta_b_matices -p test_caracteristicas_b.py','python kit/experimentos/propuesta_b_matices/experimentar.py','```','',
      'Para regenerar solo informes a partir de resultados ya guardados: `python kit/experimentos/propuesta_b_matices/experimentar.py --solo-informes`. No ejecuta el main del entrenador original. No realiza commits, PR, envíos oficiales, APIs ni descarga de recursos.','',
      'Los tres conteos previos no tienen manifiesto local verificable. Las métricas de entrenamiento, OOF principal, protocolo original retrospectivo y leaderboard se distinguen expresamente. La inferencia de un eventual candidato utiliza solo Python estándar.']
    rd+=['','piloto_sujeto_singular/ conserva la primera ejecución, en la que B4 era constante. La ejecución principal añade el mecanismo de subconjunto numerado que faltaba en esa primera implementación. Se repite toda la comparación y se declara la adaptación posterior al piloto; no es un estudio confirmatorio.']
    md('README.md',rd)
    return selected

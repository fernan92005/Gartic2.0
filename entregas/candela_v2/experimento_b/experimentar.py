#!/usr/bin/env python3
"""Experimento B: comparación emparejada, umbral interno y originales intactos."""
import sys
sys.dont_write_bytecode=True
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from caracteristicas_b import caracteristicas_b,B1,B2,B3,B4

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SEED=20261015
CONFIG=dict(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,min_samples_leaf=20,random_state=0)
PRIMARY=['B0','B1','B2','B3','B4','B5','B6','B7']
VARIANTS={k:sum((v for enabled,v in zip(flags,[B1,B2,B3,B4]) if enabled),[]) for k,flags in {'B0':(0,0,0,0),'B1':(1,0,0,0),'B2':(0,1,0,0),'B3':(0,0,1,0),'B4':(0,0,0,1),'B5':(1,1,0,0),'B6':(1,1,1,0),'B7':(1,1,1,1)}.items()}
VARIANTS.update(B1_norm=['matiz_clausula_ratio_'+k for k in ('restrictivo','condicional','aproximado','parcial')],B0_sin_NLI=[],B3_sin_NLI=B3,B5_sustituye_matices=B1+B2)
TYPES=['distorsion','contradiccion','no_respaldado','sobregeneralizacion']


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def clean(c):return {k:c[k] for k in ('pregunta','fuentes','respuesta','senales') if k in c}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metric(y,p,t):
    y=np.asarray(y);p=np.asarray(p);pred=p>=t;fp=int(sum((y==0)&pred));fn=int(sum((y==1)&~pred));n0=int(sum(y==0))
    return dict(n=len(y),tn=n0-fp,fp=fp,fn=fn,tp=int(sum(y==1))-fn,cost=4*fn+fp,score=1-(4*fn+fp)/n0 if n0 else (1-fn/max(1,int(sum(y==1)))),auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else .5)


def metrics(y,p,t,cases):
    h=np.array([c['id'].startswith('H') for c in cases]);multi=np.array([len(c['senales']['oraciones'])>1 for c in cases]);original_multi=np.array([c['id'][0] in 'GRg' and len(c['senales']['oraciones'])>1 for c in cases]);out={}
    for name,m in [('H',h),('global',np.ones(len(y),dtype=bool)),('no-H',~h),('multi',multi),('multi_GRg',original_multi)]:
        out[name]=metric(y[m],p[m],t if np.isscalar(t) else np.asarray(t)[m])
    return out


def choose(y,p,h):
    assert sum(h&(y==0))>0 and sum(h&(y==1))>0
    return max(np.arange(.02,.80,.01),key=lambda cut:metric(y[h],p[h],cut)['score'])


def fit_predict(X,y,h,tr,va):
    m=GradientBoostingClassifier(**CONFIG);m.fit(X[tr],y[tr],sample_weight=np.where(h[tr],5.,1.))
    return m.predict_proba(X[va])[:,1]


def compare(y,p,t,base,bt,mask):
    pred=p>=t;old=base>=bt;fp=mask&(y==0);pos=mask&(y==1)
    out=dict(fp_eliminados=int(sum(fp&old&~pred)),fp_nuevos=int(sum(fp&~old&pred)),fn_eliminados=int(sum(pos&~old&pred)),fn_nuevos=int(sum(pos&old&~pred)))
    out['delta_coste']=out['fp_nuevos']-out['fp_eliminados']+4*(out['fn_nuevos']-out['fn_eliminados']);return out


def paired(y,p,t,base,bt,h,g):
    rng=np.random.default_rng(SEED);unique=np.unique(g);draws=rng.integers(0,len(unique),size=(3000,len(unique)));a=[]
    pred=p>=t;old=base>=bt
    for group in unique:
        row=[]
        for m in [h,np.ones(len(y),dtype=bool)]:
            mask=(g==group)&m;row+=[int(sum(mask&(y==0))),int(sum(mask&(y==0)&old)+4*sum(mask&(y==1)&~old)),int(sum(mask&(y==0)&pred)+4*sum(mask&(y==1)&~pred))]
        a.append(row)
    sums=np.array(a)[draws].sum(axis=1);out={}
    for j,domain in enumerate(['H','global']):
        delta=(sums[:,3*j+1]-sums[:,3*j+2])/np.maximum(1,sums[:,3*j]);out[domain]=dict(ci95=np.quantile(delta,[.025,.975]).tolist(),ci95_bonferroni7=np.quantile(delta,[.025/7,1-.025/7]).tolist())
    return out


def feature_names(key,base):
    names=list(base)
    if key.endswith('sin_NLI'):
        names=[n for n in names if 'nli' not in n and n!='llm_contra_sin_apoyo']
    if key=='B5_sustituye_matices':names=[n for n in names if not n.startswith(('matiz_','llmok_matiz_'))]
    return names+VARIANTS[key]


def cohort_masks(features):
    nuance=np.array([any(v>0 for k,v in f.items() if k.startswith('matiz_perdido_')) for f in features])
    llm_ok=np.array([f['llm_no_respaldada']==0 and f['llm_contradicha']==0 and f['llm_sin_veredicto']==0 for f in features])
    def suspect(f):
        keys=['num_inexplicados','num_derivado_casual','num_cuenta_falsa','num_proporcion_falsa','entidad_ajena','entidad_ajena_fuerte','nombre_inventado','fecha_plazo_falso','hora_inventada','anadido_universal','anadido_causal','anadido_causal_todas']
        return any(f.get(k,0)>0 for k in keys) or any(v>0 for k,v in f.items() if k.startswith('matiz_perdido_'))
    none=np.array([not suspect(f) for f in features])
    low=np.array([f['nli_resp_fuente_citada_min']<.3 for f in features])
    return dict(matices=nuance,sin_sospecha=llm_ok&none&low,llm_favorable_sin_sospecha=llm_ok&none)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--solo-informes',action='store_true');args=ap.parse_args()
    started=time.time()
    bot=load(OUT/'baseline_v3/mi_bot.py','baseline_v3');trainer=load(OUT/'baseline_v3/entrenar.py','baseline_trainer')
    if args.solo_informes:
        from informes_b import generate
        generate();return
    tracked=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p and (ROOT/p).is_file()]
    # Se preservan también todos los informes previos, aunque no estén en Git.
    protected=tracked+[p for p in (ROOT/'kit/experimentos/fase4_diagnostico').rglob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    save('hashes_antes.json',before)
    feature_hash=sha(OUT/'caracteristicas_b.py')
    save('diseno_previo.json',dict(config=CONFIG,feature_sha256=feature_hash,variants=VARIANTS,primary_gate='3/4 folds H positivos, IC simultáneo7 inferior>0, repetición H positiva, AUROC H no baja, costes global/multi y FN sobregeneralización H no suben',revision='B4 amplía sujeto singular a subconjuntos numerados según el encargo; primera ejecución conservada en piloto_sujeto_singular. Revisión posterior a inspeccionar resultados: estudio exploratorio, no confirmatorio.'))
    assert sha(ROOT/'entregas/joseca_v3/mi_bot.py')==sha(OUT/'baseline_v3/mi_bot.py')
    assert sha(ROOT/'entregas/joseca_v3/entrenar.py')==sha(OUT/'baseline_v3/entrenar.py')
    cases=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()];byid={c['id']:i for i,c in enumerate(cases)}
    old=ROOT/'kit/experimentos/fase4_diagnostico';partition=json.loads((old/'particiones_reconstruidas.json').read_text());oldfolds=json.loads((old/'folds_detalle.json').read_text())
    dev=np.array([byid[i] for i in partition['development']]);dcs=[cases[i] for i in dev];dindex={c['id']:i for i,c in enumerate(dcs)}
    groups=np.array([partition['groups'][c['id']] for c in cases]);dg=groups[dev]
    h=np.array([c['id'].startswith('H') for c in cases]);dh=h[dev];y=np.array([c['etiqueta'] for c in cases]);dy=y[dev]
    base_features=[bot.caracteristicas(clean(c)) for c in cases];extra=[caracteristicas_b(clean(c)) for c in cases];base=bot.NOMBRES
    assert base==[k for k in base_features[0] if k not in trainer.EXCLUIR]
    assert len(bot.ARBOLES)==CONFIG['n_estimators'] and bot.TASA==CONFIG['learning_rate']
    XX={key:np.array([[dict(f,**e)[n] for n in feature_names(key,base)] for f,e in zip(base_features,extra)],dtype=float) for key in VARIANTS}
    # Auditoría exacta del modelo congelado: no se confunde con OOF.
    model=GradientBoostingClassifier(**CONFIG).fit(XX['B0'],y,sample_weight=np.where(h,5.,1.))
    trees=trainer.exportar_arboles(model);tree_equal=trees==bot.ARBOLES
    frozen=np.array([bot.probabilidad(f) for f in base_features]);sk=model.predict_proba(XX['B0'])[:,1];max_diff=float(max(abs(frozen-sk)))
    exact_trees=sum(a==b for a,b in zip(trees,bot.ARBOLES))
    decision_diff=int(sum((frozen>=bot.UMBRAL)!=(sk>=bot.UMBRAL)))
    assert max_diff<.02 and decision_diff==0,('Divergencia material de la baseline',max_diff,decision_diff)
    # Reproduce SOLO el protocolo histórico, sin nueva búsqueda de parámetros.
    historical=[];legacy={}
    for seed in trainer.SEMILLAS:
        p=trainer.validacion_por_familias(trainer.fabrica_gb({k:CONFIG[k] for k in ['n_estimators','max_depth','learning_rate']}),XX['B0'],y,np.array([c['id'].split('::')[0] for c in cases]),np.where(h,5.,1.),semilla=seed)
        historical.append(p);legacy[str(seed)]={c['id']:float(v) for c,v in zip(cases,p)}
    cohort=cohort_masks(base_features);historical_rows=[]
    for seed,p in zip(trainer.SEMILLAS,historical):
        fp=h&(y==0)&(p>=bot.UMBRAL);fn=h&(y==1)&(p<bot.UMBRAL);row=dict(seed=seed,metrics=metrics(y,p,bot.UMBRAL,cases),matices_ids=[c['id'] for i,c in enumerate(cases) if fp[i] and cohort['matices'][i]],sin_sospecha_ids=[c['id'] for i,c in enumerate(cases) if fp[i] and cohort['sin_sospecha'][i]],sobregeneralizacion_fn_ids=[c['id'] for i,c in enumerate(cases) if fn[i] and c.get('tipo')=='sobregeneralizacion'])
        historical_rows.append(row)
    original_best=trainer.mejor_corte(np.concatenate([y[h]]*3),np.concatenate([p[h] for p in historical]))
    save('auditoria_baseline.json',dict(bot_sha256=sha(OUT/'baseline_v3/mi_bot.py'),trainer_sha256=sha(OUT/'baseline_v3/entrenar.py'),tree_equal=tree_equal,exact_trees=exact_trees,decision_difference_fulltrain=decision_diff,max_frozen_sklearn_difference=max_diff,threshold=bot.UMBRAL,original_repeated_best_threshold=original_best[1],original_repeated_best_score=original_best[0],historical_protocol=historical_rows,claim_caveat='No hay manifiesto de los 39/70/2 casos. Cohortes operacionales fijadas y documentadas, no equivalencia demostrada con aquellas cifras.'))
    save('oof_protocolo_original.json',legacy)
    # Folds principales idénticos al diagnóstico anterior; segundo reparto de
    # desarrollo para estabilidad. La reserva histórica nunca selecciona B.
    primary=[(np.array([dindex[i] for i in f['training_ids']]),np.array([dindex[i] for i in f['validation_ids']])) for f in oldfolds]
    secondary=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+3).split(np.zeros(len(dev)),dy+2*dh,dg))
    splitsets=[primary,secondary];allresults={};details=[];allpred={};importance={}
    for repeat,folds in enumerate(splitsets):
        outputs={}
        keys=list(VARIANTS) if repeat==0 else PRIMARY
        for key in keys:
            X=XX[key][dev];p=np.full(len(dev),np.nan);threshold=np.full(len(dev),np.nan);fid=np.full(len(dev),-1)
            for k,(tr,va) in enumerate(folds):
                assert not set(dg[tr])&set(dg[va]);inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+2+k+repeat*100).split(np.zeros(len(tr)),dy[tr]+2*dh[tr],dg[tr]));cal=np.full(len(tr),np.nan)
                for a,b in inner:
                    assert not set(dg[tr[a]])&set(dg[tr[b]])
                    cal[b]=fit_predict(X,dy,dh,tr[a],tr[b])
                assert np.isfinite(cal).all();cut=choose(dy[tr],cal,dh[tr]);p[va]=fit_predict(X,dy,dh,tr,va);threshold[va]=cut;fid[va]=k
                details.append(dict(repeat=repeat,variant=key,fold=k,training_ids=[dcs[i]['id'] for i in tr],validation_ids=[dcs[i]['id'] for i in va],inner=[dict(training_ids=[dcs[tr[i]]['id'] for i in a],validation_ids=[dcs[tr[i]]['id'] for i in b]) for a,b in inner],threshold=float(cut),metrics=metrics(dy[va],p[va],cut,[dcs[i] for i in va])))
                print(f'repetición {repeat} {key} fold {k}: corte {cut:.2f}',flush=True)
            assert np.isfinite(p).all() and np.isfinite(threshold).all()
            outputs[key]=dict(p=p.tolist(),threshold=threshold.tolist(),fold=fid.tolist(),metrics=metrics(dy,p,threshold,dcs),fixed_threshold_metrics=metrics(dy,p,bot.UMBRAL,dcs))
        allpred[str(repeat)]=outputs
    b0=allpred['0']['B0'];bp=np.array(b0['p']);bt=np.array(b0['threshold']);paired_rows=[]
    for repeat,outputs in allpred.items():
        basep=np.array(outputs['B0']['p']);baset=np.array(outputs['B0']['threshold'])
        for key,r in outputs.items():
            p=np.array(r['p']);t=np.array(r['threshold']);r['transitions']={domain:compare(dy,p,t,basep,baset,m) for domain,m in [('H',dh),('global',np.ones(len(dy),dtype=bool))]}
            r['bootstrap_paired']=paired(dy,p,t,basep,baset,dh,dg)
            if repeat=='0':
                for i,c in enumerate(dcs):paired_rows.append(dict(repeat=0,variant=key,id=c['id'],group=int(dg[i]),fold=int(r['fold'][i]),etiqueta=int(dy[i]),tipo=c.get('tipo'),probabilidad=float(p[i]),umbral=float(t[i]),prediccion=int(p[i]>=t[i]),b0_probabilidad=float(bp[i]),b0_umbral=float(bt[i])))
    write_csv('predicciones_oof.csv',paired_rows)
    save('predicciones_repetidas.json',allpred);save('folds.json',details)
    save('datos_diagnostico.json',dict(ids=[c['id'] for c in dcs],groups=dg.tolist(),base_features=[base_features[i] for i in dev],extra_features=[extra[i] for i in dev],base_names=base,variant_names={key:feature_names(key,base) for key in VARIANTS},development=partition['development'],historical_holdout=partition['historical_holdout']))
    save('protocolo.json',dict(config=CONFIG,peso_h=5,threshold_selection='Score oficial H, rejilla 0.02:0.80 paso 0.01, dentro de cada entrenamiento externo',folds_source=str((old/'folds_detalle.json').relative_to(ROOT)),folds_source_sha256=sha(old/'folds_detalle.json'),historical_groups_source_sha256=sha(old/'particiones_reconstruidas.json'),primary_variants=PRIMARY,gate='Δscore H>0 en al menos 3/4 folds principales; IC95% pareado Bonferroni7 H con límite inferior>0; Δscore H repetición 2>0; ΔAUROC H>=0 en ambas repeticiones; coste global y multi no aumentan; FN sobregeneralización H no aumentan. Sin cambiar condiciones tras ver resultados.',environment=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__),feature_sha256=sha(OUT/'caracteristicas_b.py'),elapsed_seconds=time.time()-started))
    # La recomendación y posible exportación se resuelven con la misma regla
    # fijada arriba; no se fuerza un candidato por producir un entregable.
    from analizar_uso import analyze
    analyze()
    from informes_b import generate
    assert sha(OUT/'caracteristicas_b.py')==feature_hash,'Características cambiadas durante la ejecución'
    selected=generate()
    if selected:
        export(selected,XX[selected],y,h,base_features,extra,bot,trainer)
    after={path:sha(ROOT/path) for path in before};assert before==after
    save('integridad.json',dict(originals_unchanged=True,before=before,after=after,test_scored=False,submission_generated=False,candidate=selected))
    from verificar_b import verify
    verify()
    print('Experimento terminado. Candidato:',selected,flush=True)


def export(key,X,y,h,base_features,extra,bot,trainer):
    """Se ejecuta únicamente si pasa la regla de estabilidad documentada."""
    model=GradientBoostingClassifier(**CONFIG).fit(X,y,sample_weight=np.where(h,5.,1.))
    # Corte final obtenido por calibración agrupada de TODOS los datos permitidos.
    meta=json.loads((OUT/'datos_diagnostico.json').read_text());cases=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()]
    groupmap=json.loads((ROOT/'kit/experimentos/fase4_diagnostico/particiones_reconstruidas.json').read_text())['groups'];g=np.array([groupmap[c['id']] for c in cases]);cal=np.zeros(len(y))
    for tr,va in StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+4).split(X,y+2*h,g):cal[va]=fit_predict(X,y,h,tr,va)
    cut=choose(y,cal,h);names=feature_names(key,bot.NOMBRES)
    # Exportación con precisión completa y umbrales compensados a float32:
    # sklearn convierte X a float32 antes de recorrer cada árbol.
    trees=[]
    for estimator in model.estimators_[:,0]:
        t=estimator.tree_;trees.append([t.feature.tolist(),t.threshold.tolist(),t.children_left.tolist(),t.children_right.tolist(),t.value[:,0,0].tolist()])
    init=float(model._raw_predict_init(X[:1])[0,0]);code=(OUT/'baseline_v3/mi_bot.py').read_text()
    import re
    block='# >>> MODELO\nNOMBRES = '+repr(names)+'\nINICIAL = '+repr(init)+'\nTASA = .03\nARBOLES = '+repr(trees)+'\nUMBRAL = '+repr(float(cut))+'\n# <<< MODELO'
    code=re.sub(r'# >>> MODELO.*?# <<< MODELO',lambda _:block,code,flags=re.S)
    code+='\n# Características experimentales B; originales conservadas.\n_caracteristicas_v3 = caracteristicas\n'+(OUT/'caracteristicas_b.py').read_text()+'\ndef caracteristicas(caso):\n    f = _caracteristicas_v3(caso)\n    f.update(caracteristicas_b(caso))\n    return f\n'
    # Conversión float32 con librería estándar, idéntica al recorrido de sklearn.
    code=code.replace('x = [float(feats.get(n, 0.0)) for n in NOMBRES]','x = [struct.unpack("f", struct.pack("f", float(feats.get(n, 0.0))))[0] for n in NOMBRES]')
    code+='\nimport struct\n';path=OUT/'mi_bot_candidato_b.py';path.write_text(code)
    candidate=load(path,'candidate_b');fs=[dict(f,**e) for f,e in zip(base_features,extra)];pure=np.array([candidate.probabilidad(f) for f in fs]);expected=model.predict_proba(X)[:,1];diff=float(max(abs(pure-expected)))
    assert diff<1e-12,diff
    sys.path.insert(0,str(ROOT/'kit'));evaluator=load(ROOT/'kit/evaluar_local.py','candidate_evaluator');result=evaluator.evaluar(candidate.Guardian,cases)
    save('exportacion.json',dict(variant=key,threshold=float(cut),sha256=sha(path),max_sklearn_difference=diff,official_train_in_sample=result,caveat='El evaluador final mide entrenamiento, no validación independiente.'))


if __name__=='__main__':main()

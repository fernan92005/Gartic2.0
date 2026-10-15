#!/usr/bin/env python3
"""Diagnóstico de joseca_v1. No escribe bots, datasets, umbrales ni entregas."""
import sys
sys.dont_write_bytecode=True
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import unicodedata
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from auditar_version import ROOT,OUT,load,clean

SEED=20261015
TYPES=['distorsion','contradiccion','no_respaldado','sobregeneralizacion']


def normalize(s):
    s=unicodedata.normalize('NFKD',str(s or '').lower());return ''.join(c for c in s if not unicodedata.combining(c))


def groups(cases):
    # Reproducción del algoritmo documentado en fases 1–3, no una nueva selección.
    parent=list(range(len(cases)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[root(b)]=root(a)
    seen={};sources=[]
    for i,c in enumerate(cases):
        src=frozenset(normalize(s).strip() for s in c['fuentes']);sources.append(src)
        for key in [('base',c['id'].split('::')[0]),('pregunta',normalize(c['pregunta']).strip()),('fuentes',tuple(sorted(src)))]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    representatives={s:i for i,s in enumerate(sources)};unique=list(representatives)
    for i,a in enumerate(unique):
        for b in unique[:i]:
            if a and b and len(a&b)/min(len(a),len(b))>=.8:union(representatives[a],representatives[b])
    return np.array([root(i) for i in range(len(cases))])


def metric(y,s,t):
    y=np.asarray(y);s=np.asarray(s);p=s>=t
    fp=int(np.sum((y==0)&p));fn=int(np.sum((y==1)&~p));tn=int(np.sum((y==0)&~p));tp=int(np.sum((y==1)&p))
    return dict(n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,cost=fp+4*fn,score=1-(fp+4*fn)/(tn+fp) if tn+fp else 1-fn/max(1,tp+fn),auroc=float(roc_auc_score(y,s)) if len(set(y))>1 else .5)


def metrics(y,s,h,t):
    out={domain:metric(y[m],s[m],t if np.isscalar(t) else np.asarray(t)[m]) for domain,m in [('H',h),('no-H',~h),('todos',np.ones(len(y),dtype=bool))]}
    out['objective']=.7*out['H']['score']+.3*out['no-H']['score'];return out


def curve(y,s,h):
    out=[]
    for cut in np.arange(.02,.80,.01):
        p=s>=cut
        row=dict(threshold=float(cut))
        for domain,mask in [('H',h),('no-H',~h),('todos',np.ones(len(y),dtype=bool))]:
            fp=int(np.sum((y[mask]==0)&p[mask]));fn=int(np.sum((y[mask]==1)&~p[mask]));n0=int(np.sum(y[mask]==0))
            row[domain+'_fp']=fp;row[domain+'_fn']=fn;row[domain+'_cost']=fp+4*fn;row[domain+'_score']=1-(fp+4*fn)/max(1,n0)
        row['objective']=.7*row['H_score']+.3*row['no-H_score'];out.append(row)
    # Exact tie policy of entrenar.py: first (smallest) grid threshold.
    return out,max(out,key=lambda r:r['objective'])['threshold'],max(out,key=lambda r:r['H_score'])['threshold']


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def signal_features(c):
    ss=c.get('senales',{}).get('oraciones') or [];ff=c.get('senales',{}).get('fragmentos') or [];out={}
    def safe(v):
        try:return min(1.,max(0.,float(v))) if math.isfinite(float(v)) else 0.
        except (TypeError,ValueError):return 0.
    risks={}
    for prefix,items in [('s',ss),('f',ff)]:
        rows=[]
        for item in items:
            e=[safe(v) for v in item.get('respaldo_por_fuente',[])];co=[safe(v) for v in item.get('contradiccion_por_fuente',[])];sim=[safe(v) for v in item.get('similitud_por_fuente',[])];joint=safe(item.get('respaldo_todas'));ca=safe(item.get('contradiccion_todas'))
            most=max(range(len(sim)),key=sim.__getitem__) if sim else -1
            rows.append(dict(e_any=max(e,default=0.),e_all=joint,c_any=max(co,default=0.),c_all=ca,similarity=max(sim,default=0.),risk=1-max(joint,max(e,default=0.)),gap=joint-max(e,default=0.),c_relevant=co[most] if 0<=most<len(co) else 0.))
        for key in ['e_any','e_all','c_any','c_all','similarity','risk','gap','c_relevant']:
            a=[r[key] for r in rows] or [0.]
            out[prefix+'_'+key+'_min']=min(a);out[prefix+'_'+key+'_max']=max(a);out[prefix+'_'+key+'_mean']=sum(a)/len(a)
        risks[prefix]=[r['risk'] for r in rows]
    llms=[s.get('llm') or {} for s in ss];vs=[l.get('veredicto') for l in llms];conf=[safe(l.get('confianza')) for l in llms]
    out['llm_all_supported']=float(bool(vs) and all(v=='respaldada' for v in vs));out['llm_bad']=float(any(v in ['no_respaldada','contradicha'] for v in vs));out['llm_missing']=float(not vs or any(v is None for v in vs))
    out['llm_conf_mean']=sum(conf)/max(1,len(conf));out['llm_conf_min']=min(conf,default=0.)
    lr=[(1-c)/2 if v=='respaldada' else (.5+c/2 if v in ['no_respaldada','contradicha'] else .5) for v,c in zip(vs,conf)]
    out['llm_risk_max']=max(lr,default=.5);out['disagreement']=out['s_risk_max']-out['llm_risk_max'];out['f_minus_s_risk']=out['f_risk_max']-out['s_risk_max']
    return out


def bootstrap(cases,y,s,h,g,t):
    pred=s>=t;stats=[]
    for group in np.unique(g):
        mask=g==group;row=[]
        for domain in [h,~h,np.ones(len(y),dtype=bool)]:
            m=mask&domain;row.extend([np.sum(m&(y==0)),np.sum(m&(y==0)&pred)+4*np.sum(m&(y==1)&~pred)])
        for category in TYPES:
            m=mask&np.array([c.get('tipo')==category for c in cases]);row.extend([np.sum(m),np.sum(m&~pred)])
        stats.append(row)
    a=np.asarray(stats,dtype=float);rng=np.random.default_rng(SEED);draws=rng.integers(0,len(a),size=(2000,len(a)));sums=a[draws].sum(axis=1);out={}
    for i,domain in enumerate(['H','no-H','todos']):
        score=1-sums[:,2*i+1]/np.maximum(1,sums[:,2*i]);out[domain]=np.quantile(score,[.025,.975]).tolist()
    obj=.7*(1-sums[:,1]/np.maximum(1,sums[:,0]))+.3*(1-sums[:,3]/np.maximum(1,sums[:,2]));out['objective']=np.quantile(obj,[.025,.975]).tolist()
    for j,category in enumerate(TYPES):
        n=sums[:,6+2*j];fn=sums[:,7+2*j];valid=n>0;out['recall_'+category]=np.quantile(1-fn[valid]/n[valid],[.025,.975]).tolist()
    return out


def bootstrap_difference(y,s,h,g,current,alternative):
    """Bootstrap pareado: mismas muestras de grupos para ambos cortes."""
    costs=[]
    for group in np.unique(g):
        row=[]
        for scope in [h,~h,np.ones(len(y),dtype=bool)]:
            m=(g==group)&scope;neg=np.sum(m&(y==0));row.append(neg)
            for cut in [current,alternative]:
                pred=s>=cut;row.append(np.sum(m&(y==0)&pred)+4*np.sum(m&(y==1)&~pred))
        costs.append(row)
    a=np.asarray(costs,dtype=float);rng=np.random.default_rng(SEED)
    sums=a[rng.integers(0,len(a),size=(2000,len(a)))].sum(axis=1)
    out={};deltas=[]
    for j,domain in enumerate(['H','no-H','todos']):
        diff=(sums[:,3*j+1]-sums[:,3*j+2])/np.maximum(1,sums[:,3*j])
        deltas.append(diff);out[domain]=np.quantile(diff,[.025,.975]).tolist()
    out['objective']=np.quantile(.7*deltas[0]+.3*deltas[1],[.025,.975]).tolist()
    out['replicas']=2000;out['seed']=SEED;out['direction']='alternativo menos actual'
    return out


def main():
    if not (OUT/'procedencia.json').exists():
        import auditar_version;auditar_version.main()
    provenance=json.loads((OUT/'procedencia.json').read_text());before=json.loads((OUT/'hashes_antes.json').read_text())
    for path,digest in before.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    bot=load(ROOT/'entregas/joseca_v1/mi_bot.py','target_joseca');trainer=load(ROOT/'entregas/joseca_v1/entrenar.py','train_joseca');e4=load(ROOT/'entregas/candela_v1/mi_bot.py','reference_e4')
    cases=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()];assert len({c['id'] for c in cases})==len(cases)
    assert all((c['etiqueta']==0 and c.get('tipo') is None) or (c['etiqueta']==1 and c.get('tipo') in TYPES) for c in cases),'Revisar representación de etiquetas múltiples'
    cached=np.load(OUT/'caracteristicas.npz');X=cached['X'];y=cached['y'].astype(int);h=cached['h'];assert cached['names'].tolist()==bot.NOMBRES
    full_feats=json.loads((OUT/'caracteristicas_completas.json').read_text());weight=provenance['config']['weight_h'];l2=provenance['config']['l2'];g=groups(cases)
    split=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED);dev,hold=next(split.split(np.zeros(len(cases)),y+2*h,g));assert not set(g[dev])&set(g[hold])
    dcs=[cases[i] for i in dev];dy=y[dev];dh=h[dev];dg=g[dev];DX=X[dev]
    assert (len(dev),int(sum(dh)),len(hold),int(sum(h[hold])),len(set(g)))==(1280,473,427,158,337),'La reconstrucción no coincide con las comprobaciones históricas'
    folds=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+1).split(np.zeros(len(dev)),dy+2*dh,dg));foldids=np.zeros(len(dev),dtype=int)
    partition=dict(seed=SEED,development=[cases[i]['id'] for i in dev],historical_holdout=[cases[i]['id'] for i in hold],groups={c['id']:int(g[i]) for i,c in enumerate(cases)},caveat='Manifiestos originales ausentes en este checkout; reconstrucción del algoritmo documentado, con controles de tamaño/grupos/métricas históricas.')
    (OUT/'particiones_reconstruidas.json').write_text(json.dumps(partition,indent=2)+'\n')
    configs=provenance['config'];oof=np.zeros(len(dev));nested=np.zeros(len(dev));nested_h=np.zeros(len(dev));nested_global=np.zeros(len(dev));logit_groups={key:np.zeros(len(dev)) for key in ['llm','nli','numeros','matices','texto','otros']};ablations={key:np.zeros(len(dev)) for key in logit_groups};fold_detail=[];models=[]
    def family(name):
        n=name.replace('llmok_','')
        if n.startswith(('num_','llm_ok_y_num','nombre_')):return 'numeros'
        if n.startswith(('matiz_','anadido_','huecos_')):return 'matices'
        if n.startswith('palabras_'):return 'texto'
        if n.startswith('nli_') or 'nli_bajo' in n:return 'nli'
        if n.startswith('llm_'):return 'llm'
        return 'otros'
    for k,(tr,va) in enumerate(folds):
        assert not set(dg[tr])&set(dg[va]);foldids[va]=k
        model=trainer.ajustar(DX[tr],dy[tr],np.where(dh[tr],weight,1.),l2);oof[va]=trainer.predecir(DX[va],model);models.append(model)
        mean,scale,coef,bias=model;contrib=(DX[va]-mean)/scale*coef
        for key in logit_groups:
            cols=[j for j,n in enumerate(bot.NOMBRES) if family(n)==key]
            logit_groups[key][va]=contrib[:,cols].sum(axis=1) if cols else 0.
            z=((DX[va]-mean)/scale)@coef+bias
            # Feature-to-zero diagnostic with coefficients fixed, not retraining.
            if cols:z-=np.sum(DX[va][:,cols]/scale[cols]*coef[cols],axis=1)
            ablations[key][va]=1/(1+np.exp(-np.clip(z,-30,30)))
        calibration=np.zeros(len(tr))
        inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+2+k).split(np.zeros(len(tr)),dy[tr]+2*dh[tr],dg[tr]))
        for a,b in inner:
            assert not set(dg[tr[a]])&set(dg[tr[b]])
            m=trainer.ajustar(DX[tr[a]],dy[tr[a]],np.where(dh[tr[a]],weight,1.),l2);calibration[b]=trainer.predecir(DX[tr[b]],m)
        rows,t,th=curve(dy[tr],calibration,dh[tr]);nested[va]=t;nested_h[va]=th
        tg=max(rows,key=lambda r:r['todos_score'])['threshold'];nested_global[va]=tg
        fold_detail.append(dict(fold=k,training_ids=[dcs[i]['id'] for i in tr],validation_ids=[dcs[i]['id'] for i in va],inner_folds=[dict(training_ids=[dcs[tr[j]]['id'] for j in a],validation_ids=[dcs[tr[j]]['id'] for j in b]) for a,b in inner],raw_threshold_current=bot.UMBRAL,raw_threshold_nested_weighted=t,raw_threshold_nested_H=th,current=metrics(dy[va],oof[va],dh[va],bot.UMBRAL),nested=metrics(dy[va],oof[va],dh[va],t),parameters=dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),intercept=float(bias))))
        print('JOSECA fold',k,'current',fold_detail[-1]['current']['objective'],'nested',t,flush=True)
        fold_detail[-1].update(raw_threshold_nested_official=tg,nested_official=metrics(dy[va],oof[va],dh[va],tg))
    # Candela E4 is a separate reference, never injected into Joseca.
    cf=[e4.features_for(clean(c),'sin_numeros') for c in dcs];names=e4._PARAMETERS['features'];CX=np.array([[f[n] for n in names] for f in cf]);base=np.zeros(len(dev));rel=np.array([e4.sci_relation_risk(clean(c)) for c in dcs]);checks=[e4.sci_relations(clean(c)) for c in dcs]
    hp=e4._PARAMETERS['config']
    for tr,va in folds:
        m=HistGradientBoostingClassifier(max_iter=hp['iterations'],max_leaf_nodes=hp['leaves'],learning_rate=.06,min_samples_leaf=20,l2_regularization=hp['l2'],early_stopping=False,random_state=SEED)
        m.fit(CX[tr],dy[tr],sample_weight=np.where(dh[tr],1.,hp['non_h_weight']));base[va]=m.predict_proba(CX[va])[:,1]
    corrected=np.round(np.maximum(base,rel),12);et=e4._SCI_THRESHOLD
    # Verify reconstructed HGB data/protocol against previously documented OOF.
    bm=metrics(dy,base,dh,et);assert abs(bm['H']['score']-.3092369477911646)<1e-12
    frozen_j=np.array([bot.Guardian().score(clean(c)) for c in cases]);frozen_c=np.array([e4.Guardian().score(clean(c)) for c in cases])
    # Reproduce author's fixed architecture, without new configuration search.
    repeated=[];author_oof={}
    for seed in trainer.SEMILLAS:
        p=trainer.validacion_por_grupos(X,y,[c['id'] for c in cases],h,weight,l2,semilla=seed)
        repeated.append(dict(seed=seed,scope='H OOF del protocolo autor; no-H siempre entrena',**metric(y[h],p[h],bot.UMBRAL)));author_oof[str(seed)]={cases[i]['id']:float(p[i]) for i in np.flatnonzero(h)}
    result=dict(provenance=provenance,counts=dict(total=len(cases),development=len(dev),h_development=int(sum(dh)),historical=len(hold),h_historical=int(sum(h[hold])),groups=len(set(g)),groups_development=len(set(dg))),joseca_oof=metrics(dy,oof,dh,bot.UMBRAL),joseca_nested_weighted=metrics(dy,oof,dh,nested),joseca_nested_H=metrics(dy,oof,dh,nested_h),candela_hgb_oof=bm,candela_e4_oof=metrics(dy,corrected,dh,et),joseca_frozen_train=metrics(y,frozen_j,h,.5),joseca_frozen_historical=metrics(y[hold],frozen_j[hold],h[hold],.5),candela_frozen_train=metrics(y,frozen_c,h,.5),candela_frozen_historical=metrics(y[hold],frozen_c[hold],h[hold],.5),author_repeated_h=repeated,bootstrap_current=bootstrap(dcs,dy,oof,dh,dg,bot.UMBRAL),bootstrap_nested=bootstrap(dcs,dy,oof,dh,dg,nested))
    import sklearn
    result['environment']=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    result['bootstrap_paired_delta']=bootstrap_difference(dy,oof,dh,dg,bot.UMBRAL,nested)
    result['joseca_nested_official']=metrics(dy,oof,dh,nested_global)
    result['bootstrap_official']=bootstrap(dcs,dy,oof,dh,dg,nested_global)
    result['bootstrap_paired_official_delta']=bootstrap_difference(dy,oof,dh,dg,bot.UMBRAL,nested_global)
    for key,p in ablations.items():result.setdefault('feature_zero_ablation',{})[key]=metrics(dy,p,dh,bot.UMBRAL)
    rows,tw,th=curve(dy,oof,dh);write_csv(OUT/'curva_coste_oof.csv',rows);result['descriptive_global_best']=dict(weighted_threshold=tw,h_threshold=th,weighted=metrics(dy,oof,dh,tw),h=metrics(dy,oof,dh,th),caveat='Selección retrospectiva en OOF, no estimación independiente y no se adopta')
    (OUT/'resumen_metricas.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'folds_detalle.json').write_text(json.dumps(fold_detail,indent=2)+'\n');(OUT/'autor_oof_h.json').write_text(json.dumps(author_oof,indent=2)+'\n')
    records=[]
    for j,i in enumerate(dev):
        c=cases[i];records.append(dict(id=c['id'],group=int(dg[j]),fold=int(foldids[j]),etiqueta=int(dy[j]),tipo=c.get('tipo'),joseca_raw=float(oof[j]),joseca_score=bot.reescalar(float(oof[j]),bot.UMBRAL),nested_threshold=float(nested[j]),nested_h_threshold=float(nested_h[j]),hgb_raw=float(base[j]),e4_raw=float(corrected[j]),e4_threshold=et,relations=checks[j],signals=signal_features(c),features=full_feats[i],logit_contributions={key:float(v[j]) for key,v in logit_groups.items()}))
        records[-1]['nested_official_threshold']=float(nested_global[j])
    (OUT/'predicciones_oof.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    write_csv(OUT/'predicciones_oof.csv',[{k:v for k,v in r.items() if k not in ['relations','signals','features','logit_contributions']} for r in records])
    # Every table is generated from these explicit OOF records.
    from informes import generate
    generate(cases,dev,hold,records,result,fold_detail,metric,metrics,write_csv)
    after={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in before};assert before==after
    (OUT/'integridad.json').write_text(json.dumps(dict(unchanged=True,files=after,bots_modified=False,thresholds_modified=False,test_scored=False,submission_generated=False),indent=2)+'\n')
    from verificar_diagnostico import verify
    verify()
    print('JOSECA OOF',result['joseca_oof'],flush=True);print('JOSECA NESTED',result['joseca_nested_weighted'],flush=True);print('INTEGRIDAD OK',flush=True)

if __name__=='__main__':main()

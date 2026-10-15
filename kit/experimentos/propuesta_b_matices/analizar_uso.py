"""Audita uso de variables reconstituyendo los modelos externos ya evaluados.
No busca parámetros ni modifica predicciones, reglas o umbrales.
"""
import sys
sys.dont_write_bytecode=True
import collections
import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from experimentar import OUT,ROOT,CONFIG,PRIMARY,save,write_csv
from caracteristicas_b import B1,B2,B3,B4


def analyze():
    data=json.loads((OUT/'datos_diagnostico.json').read_text());runs=json.loads((OUT/'predicciones_repetidas.json').read_text());folds=json.loads((OUT/'folds.json').read_text());byid={c['id']:c for c in [json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()]};index={caseid:i for i,caseid in enumerate(data['ids'])};y=np.array([byid[i]['etiqueta'] for i in data['ids']]);h=np.array([i.startswith('H') for i in data['ids']]);features=[dict(f,**e) for f,e in zip(data['base_features'],data['extra_features'])];rows=[];details=[]
    for key in PRIMARY:
        names=data['variant_names'][key];X=np.array([[f[n] for n in names] for f in features]);uses=collections.Counter();importance=collections.defaultdict(list)
        for f in [f for f in folds if f['repeat']==0 and f['variant']==key]:
            tr=np.array([index[i] for i in f['training_ids']]);va=np.array([index[i] for i in f['validation_ids']]);model=GradientBoostingClassifier(**CONFIG).fit(X[tr],y[tr],sample_weight=np.where(h[tr],5.,1.));p=model.predict_proba(X[va])[:,1];assert np.allclose(p,np.array(runs['0'][key]['p'])[va],atol=1e-14,rtol=0.)
            split=collections.Counter(names[int(v)] for estimator in model.estimators_[:,0] for v in estimator.tree_.feature if v>=0);uses.update(split)
            details.append(dict(variant=key,fold=f['fold'],splits=dict(split)))
            for j,n in enumerate(names):importance[n].append(float(model.feature_importances_[j]))
        for name in names:
            if name not in B1+B2+B3+B4:continue
            vals=np.array([f[name] for f in features]);rows.append(dict(variant=key,feature=name,activations=int(sum(vals>0)),n_unique=len(np.unique(vals)),splits_across_four_outer_models=uses[name],mean_impurity_importance=float(np.mean(importance[name])),caveat='Uso descriptivo y correlacionado; no prueba valor causal ni generalización'))
    write_csv('uso_caracteristicas.csv',rows);save('uso_caracteristicas.json',dict(rows=rows,folds=details,all_reconstructed_predictions_match=True))
    print('Auditoría de uso: scores externos reproducidos.',flush=True)
    return rows


if __name__=='__main__':analyze()

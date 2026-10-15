"""Verifica las comparaciones sin volver a entrenar ni modificar originales."""
import sys
sys.dont_write_bytecode=True
import csv
import hashlib
import json
import numpy as np
from experimentar import OUT,ROOT,load,metric


def verify():
    data=json.loads((OUT/'datos_diagnostico.json').read_text());runs=json.loads((OUT/'predicciones_repetidas.json').read_text());folds=json.loads((OUT/'folds.json').read_text());ids=set(data['ids']);group=dict(zip(data['ids'],data['groups']));positions={caseid:i for i,caseid in enumerate(data['ids'])}
    sys.path.insert(0,str(ROOT/'kit'));official=load(ROOT/'kit/evaluar_local.py','verify_official_b');cases=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()];byid={c['id']:c for c in cases};y=np.array([byid[i]['etiqueta'] for i in data['ids']]);h=np.array([i.startswith('H') for i in data['ids']]);count=0
    for repeat,outputs in runs.items():
        for key,r in outputs.items():
            details=[f for f in folds if f['repeat']==int(repeat) and f['variant']==key];seen=[]
            for f in details:
                tr=set(f['training_ids']);va=set(f['validation_ids']);seen+=f['validation_ids'];assert not tr&va and tr|va==ids
                assert not {group[i] for i in tr}&{group[i] for i in va};innerseen=[]
                for inner in f['inner']:
                    it=set(inner['training_ids']);iv=set(inner['validation_ids']);innerseen+=inner['validation_ids'];assert not it&iv and it|iv==tr
                    assert not {group[i] for i in it}&{group[i] for i in iv}
                assert set(innerseen)==tr and len(innerseen)==len(tr)
                for caseid in va:
                    i=positions[caseid];assert r['threshold'][i]==f['threshold'] and r['fold'][i]==f['fold']
            assert len(seen)==len(ids) and set(seen)==ids
            p=np.array(r['p']);t=np.array(r['threshold']);pred=(p>=t).astype(int)
            for domain,m in [('H',h),('global',np.ones(len(y),dtype=bool))]:
                assert abs(official.puntuacion(y[m],pred[m])-r['metrics'][domain]['score'])<1e-12
                assert abs(official.auroc(y[m],p[m])-r['metrics'][domain]['auroc'])<1e-12
                delta=r['transitions'][domain]['delta_coste'];assert delta==r['metrics'][domain]['cost']-outputs['B0']['metrics'][domain]['cost']
            count+=1
    with (OUT/'comparativa_por_tipo.csv').open() as f:rows=list(csv.DictReader(f))
    for repeat,outputs in runs.items():
        for key,r in outputs.items():
            for domain in ['H','global']:
                costs=sum(int(row['coste_fn']) for row in rows if row['repeat']==repeat and row['variant']==key and row['domain']==domain)
                assert costs+r['metrics'][domain]['fp']==r['metrics'][domain]['cost']
    before=json.loads((OUT/'hashes_antes.json').read_text());assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value for path,value in before.items())
    selected=json.loads((OUT/'seleccion.json').read_text())['selected'];assert (OUT/'mi_bot_candidato_b.py').exists()==bool(selected)
    save=dict(passed=True,variant_repeat_pairs=count,group_disjoint_outer_inner=True,validation_once=True,threshold_only_inner=True,official_metric_parity=True,per_type_cost_identity=True,transition_cost_identity=True,originals_unchanged=True,candidate=selected)
    (OUT/'verificacion.json').write_text(json.dumps(save,indent=2)+'\n');print('VERIFICACIÓN OK',flush=True)
    return save


if __name__=='__main__':verify()

"""Comprueba métrica oficial, reconstrucción de scores y separación de folds."""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import numpy as np
from auditar_version import ROOT,OUT,load,clean


def verify():
    sys.path.insert(0,str(ROOT/'kit'))
    official=load(ROOT/'kit/evaluar_local.py','official_eval')
    bot=load(ROOT/'entregas/joseca_v1/mi_bot.py','verify_joseca')
    cases=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()]
    records=json.loads((OUT/'predicciones_oof.json').read_text())
    res=json.loads((OUT/'resumen_metricas.json').read_text())
    partitions=json.loads((OUT/'particiones_reconstruidas.json').read_text())
    folds=json.loads((OUT/'folds_detalle.json').read_text())
    byid={r['id']:r for r in records};group=partitions['groups']
    assert len(byid)==len(records)==1280
    assert set(byid)==set(partitions['development'])
    assert not set(group[i] for i in byid)&set(group[i] for i in partitions['historical_holdout'])
    y=np.array([r['etiqueta'] for r in records]);scores=np.array([r['joseca_score'] for r in records])
    raw=np.array([r['joseca_raw'] for r in records]);pred=(scores>=.5).astype(int)
    assert np.array_equal(pred,raw>=bot.UMBRAL)
    assert abs(official.puntuacion(y,pred)-res['joseca_oof']['todos']['score'])<1e-12
    assert abs(official.auroc(y,scores)-res['joseca_oof']['todos']['auroc'])<1e-12
    cache=np.load(OUT/'caracteristicas.npz');index={c['id']:i for i,c in enumerate(cases)}
    validation=[];max_error=0.
    for fold in folds:
        tr=set(fold['training_ids']);va=set(fold['validation_ids']);validation.extend(va)
        assert not tr&va and tr|va==set(byid)
        assert not {group[i] for i in tr}&{group[i] for i in va}
        inner_seen=[]
        for inner in fold['inner_folds']:
            it=set(inner['training_ids']);iv=set(inner['validation_ids']);inner_seen.extend(iv)
            assert it|iv==tr and not it&iv
            assert not {group[i] for i in it}&{group[i] for i in iv}
        assert len(inner_seen)==len(tr) and set(inner_seen)==tr
        params=fold['parameters'];mean=np.array(params['mean']);scale=np.array(params['scale']);coef=np.array(params['coef'])
        ordered=list(va);X=cache['X'][[index[i] for i in ordered]]
        z=np.clip(((X-mean)/scale)@coef+params['intercept'],-30,30)
        expected=1/(1+np.exp(-z));actual=np.array([byid[i]['joseca_raw'] for i in ordered])
        max_error=max(max_error,float(np.max(np.abs(actual-expected))))
        assert np.allclose(actual,expected,atol=1e-14,rtol=0.)
    assert len(validation)==len(byid) and set(validation)==set(byid)
    # Compatibilidad del bot congelado con el evaluador, explícitamente in-sample.
    frozen=official.evaluar(bot.Guardian,cases)
    reference=res['joseca_frozen_train']['todos']
    for key,other in [('puntuacion','score'),('auroc','auroc'),('fp','fp'),('fn','fn')]:
        assert abs(frozen[key]-reference[other])<1e-12,(key,frozen,reference)
    before=json.loads((OUT/'hashes_antes.json').read_text())
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value for path,value in before.items())
    result=dict(passed=True,official_metric_oof=True,official_evaluator_frozen_in_sample={key:frozen[key] for key in ['puntuacion','auroc','fp','fn']},outer_and_inner_group_disjoint=True,validation_once_per_case=True,max_oof_parameter_reconstruction_error=max_error,original_files_unchanged=True)
    (OUT/'verificacion.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFICACIÓN OK: métrica oficial, folds externos/internos, scores y hashes.',flush=True)
    return result


if __name__=='__main__':verify()

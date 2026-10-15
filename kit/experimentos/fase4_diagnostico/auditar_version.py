"""Identifica versión/configuración sin ejecutar el main del entrenamiento."""
import sys
sys.dont_write_bytecode=True
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile
import xml.etree.ElementTree as ET
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def clean(c):return {k:c[k] for k in ('pregunta','fuentes','respuesta','senales') if k in c}


def main():
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and 'experimentos' not in p.parts and '__pycache__' not in p.parts]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'hashes_antes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    bot=load(ROOT/'entregas/joseca_v1/mi_bot.py','joseca');trainer=load(ROOT/'entregas/joseca_v1/entrenar.py','trainer')
    e4=load(ROOT/'entregas/candela_v1/mi_bot.py','candela_e4')
    c=[json.loads(l) for l in (ROOT/'kit/train.jsonl').read_text().splitlines() if l.strip()]
    feats=[bot.caracteristicas(clean(x)) for x in c];names=bot.NOMBRES
    assert len(names)==len(bot.MEDIAS)==len(bot.ESCALAS)==len(bot.PESOS)
    X=np.array([[r[n] for n in names] for r in feats]);y=np.array([x['etiqueta'] for x in c],dtype=float);h=np.array([x['id'].startswith('H') for x in c])
    rows=[]
    for weight in [1.,3.,5.]:
        for l2 in [3.,10.,30.]:
            mean,scale,coef,bias=trainer.ajustar(X,y,np.where(h,weight,1.),l2)
            delta=max(np.max(np.abs(mean-bot.MEDIAS)),np.max(np.abs(scale-bot.ESCALAS)),np.max(np.abs(coef-bot.PESOS)),abs(bias-bot.SESGO))
            exact=(np.array_equal(np.round(mean,6),bot.MEDIAS) and np.array_equal(np.round(scale,6),bot.ESCALAS) and np.array_equal(np.round(coef,6),bot.PESOS) and round(float(bias),6)==bot.SESGO)
            rows.append(dict(weight_h=weight,l2=l2,max_parameter_error=float(delta),matches_rounded_parameters=exact));print(rows[-1],flush=True)
    matched=[r for r in rows if r['matches_rounded_parameters']]
    assert len(matched)==1,('No se identifica una configuración única',rows)
    git_blob=subprocess.run(['git','show','b6e5a97:kit/mi_bot.py'],cwd=ROOT,capture_output=True,check=True).stdout
    git_message=subprocess.run(['git','show','--format=%B','--no-patch','b6e5a97'],cwd=ROOT,capture_output=True,check=True,text=True).stdout
    assert hashlib.sha256(git_blob).hexdigest()==hashes['entregas/joseca_v1/mi_bot.py']
    with zipfile.ZipFile(ROOT/'Enunciado_El_Guardian.docx') as z:
        xml=ET.fromstring(z.read('word/document.xml'));text='\n'.join(''.join(x.itertext()) for x in xml.findall('.//{*}p'))
    (OUT/'enunciado_extraido.txt').write_text(text)
    provenance=dict(target='entregas/joseca_v1/mi_bot.py',sha256=hashes['entregas/joseca_v1/mi_bot.py'],features=names,feature_count=len(names),threshold=bot.UMBRAL,config=matched[0],config_audit=rows,leaderboard_evidence_commit='b6e5a9787f4de02b984ca982fd6556b8f047866c',leaderboard_evidence_message=git_message,leaderboard_caveat='Coincidencia exacta con el archivo del commit que registra las métricas; no se dispone del recibo del servidor ni de etiquetas de test.',e4_sha256=hashes['entregas/candela_v1/mi_bot.py'],kit_mi_bot_exists=(ROOT/'kit/mi_bot.py').exists(),previous_experiments_present=(ROOT/'kit/experimentos/particiones.json').exists())
    (OUT/'procedencia.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    np.savez_compressed(OUT/'caracteristicas.npz',X=X,y=y,h=h,names=np.array(names))
    (OUT/'caracteristicas_completas.json').write_text(json.dumps(feats,ensure_ascii=False)+'\n')
    print('Identificación exacta:',matched[0],flush=True)

if __name__=='__main__':main()

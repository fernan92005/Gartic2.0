"""pi_train = prevalencia ponderada (peso_h=5 en H, 1 en no-H) del train, sin familias del hold-out."""
import json, os
AQ=os.path.dirname(os.path.abspath(__file__)); R=os.path.abspath(os.path.join(AQ,*[".."]*5))
hold={l.strip() for l in open(os.path.join(AQ,"..","..","holdout_familias.txt")) if l.strip()}
sw=sy=0.0
for l in open(os.path.join(R,"kit","train.jsonl"),encoding="utf-8"):
    if not l.strip(): continue
    c=json.loads(l); h=c["id"].startswith("H")
    if h and c["id"].split("::")[0] in hold: continue
    w=5.0 if h else 1.0; sw+=w; sy+=w*c["etiqueta"]
print(sy/sw)

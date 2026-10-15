"""Comprueba: n<=n0 => features idénticas a v4; cuenta excepciones en train/test/giro."""
import json, os, importlib.util, sys
AQ=os.path.dirname(os.path.abspath(__file__)); R=os.path.abspath(os.path.join(AQ,*[".."]*5))
def load(p,n):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
a=load(os.path.join(R,"entregas","joseca_v4","mi_bot.py"),"v4"); b=load(os.path.join(AQ,"mi_bot.py"),"c1")
n_ig=n_dif=exc=0; dif_nombres=set()
for fn in ("train","test_publico","giro_publico"):
    for l in open(os.path.join(R,"kit",fn+".jsonl"),encoding="utf-8"):
        if not l.strip(): continue
        c=json.loads(l); c={k:c[k] for k in ("pregunta","fuentes","respuesta","senales") if k in c}
        try: fb=b.caracteristicas(c)
        except Exception as e: exc+=1; print("EXC",fn,e); continue
        fa=a.caracteristicas(c); s=c["senales"]; no=len(s["oraciones"]); nf=len(s["fragmentos"])
        nr=sum(1 for o in s["oraciones"] if (o.get("llm") or {}).get("veredicto")=="respaldada")
        dif={k for k in fa if abs(fa[k]-fb[k])>1e-12}
        if no<=1 and nf<=2:   # n<=n0 en todos los tipos (citada<=1 si no<=1)
            assert not dif,(fn,dif)
        if dif: n_dif+=1; dif_nombres|=dif
        else: n_ig+=1
print("excepciones",exc,"casos iguales",n_ig,"con cambios",n_dif); print(sorted(dif_nombres))

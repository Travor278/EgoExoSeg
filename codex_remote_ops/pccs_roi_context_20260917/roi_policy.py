import numpy as np
from policies import BASE,DENSE,EXPERTS,allowed
from roi_evidence import KEYS
GROUPS={
 'global':BASE+DENSE,
 'local15':BASE+tuple('local15_'+k for k in KEYS),
 'local20':BASE+tuple('local20_'+k for k in KEYS),
 'object20':BASE+tuple('object20_'+k for k in KEYS),
 'global_local20':BASE+DENSE+tuple('local20_'+k for k in KEYS),
 'global_object20':BASE+DENSE+tuple('object20_'+k for k in KEYS),
 'global_multiscale':BASE+DENSE+tuple(p+'_'+k for p in ('local15','local20') for k in KEYS),
}
def permitted(r,e,group):
    b=r['baseline'];x=r['evidence'][e]
    if not (e!=b and r['source_valid'] and x['quality_valid'] and r['mask_sha256'][e]!=r['mask_sha256'][b]):return False
    if group=='global':return allowed(r,e)
    modes=[p for p in ('local15','local20','object20') if any(k.startswith(p+'_') for k in GROUPS[group])]
    local=any(x[p+'_valid'] and x[p+'_log_forward']>0 and x[p+'_log_backward']>0 for p in modes)
    return local or allowed(r,e)
def features(r,e,group):
    c=np.asarray([r['evidence'][e][k] for k in GROUPS[group]],float);b=np.asarray([r['evidence'][r['baseline']][k] for k in GROUPS[group]],float)
    return np.r_[c,b,c-b,[float(e==x) for x in EXPERTS],[float(r['baseline']==x) for x in EXPERTS]]
def predict(r,model,group,threshold,strength=1.):
    if strength==0:return r['baseline']
    options=[e for e in EXPERTS if permitted(r,e,group)]
    if not options:return r['baseline']
    values=model.predict(np.asarray([features(r,e,group) for e in options]));assert np.isfinite(values).all();i=int(np.argmax(values));return options[i] if values[i]>threshold else r['baseline']

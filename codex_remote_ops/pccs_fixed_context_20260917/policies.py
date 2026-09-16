"""PCCS confidence correction requires a valid challenger and improved native cycle."""
import numpy as np
EXPERTS=('visual','anchor','fusion')
BASE=('hard_votes','hard_distance','has_points','sam_iou','area','components')
DENSE=('forward_mass','backward_mass','cycle_mass','soft_cycle','object_cos')
CONTEXT=tuple('ctx'+s+'_'+k for s in ('15','20') for k in ('valid','context_cos','cross_bg','contrast','residual_cos','cycle_contrast'))
GROUPS={'base':BASE,'cycle':BASE+DENSE,'context':BASE+CONTEXT,'full':BASE+DENSE+CONTEXT}
HAND=[{'name':'softcycle_m005','kind':'soft','margin':.005}, {'name':'softcycle_m02','kind':'soft','margin':.02},
      {'name':'contrast15','kind':'contrast','scale':'15','margin':.03}, {'name':'contrast20','kind':'contrast','scale':'20','margin':.03},
      {'name':'residual15','kind':'residual','scale':'15','margin':.03}, {'name':'residual20','kind':'residual','scale':'20','margin':.03},
      {'name':'joint15','kind':'joint','scale':'15','margin':.02}, {'name':'joint20','kind':'joint','scale':'20','margin':.02}]

def allowed(record,expert):
    base=record['baseline'];c=record['evidence'][expert];b=record['evidence'][base]
    return expert!=base and record['mask_sha256'][expert]!=record['mask_sha256'][base] and record['source_valid'] and c['quality_valid'] and c['valid'] and c['soft_cycle']>max(.01,b['soft_cycle']+.005)

def hand_select(record,cfg,strength=1.,wrong=False):
    base=record['baseline']
    if strength==0:return base
    b=record['evidence'][base];options=[]
    for e in EXPERTS:
        if not allowed(record,e):continue
        c=record['evidence'][e]
        if cfg['kind']=='soft':gain=c['soft_cycle']-b['soft_cycle']
        else:
            p=('wrong' if wrong else 'ctx')+cfg['scale']+'_'
            if not c[p+'valid'] or not b[p+'valid']:continue
            key={'contrast':'contrast','residual':'residual_cos','joint':'cycle_contrast'}[cfg['kind']]
            gain=c[p+key]-b[p+key]
            if cfg['kind']=='joint' and c[p+'contrast']<=b[p+'contrast']:continue
        if gain>cfg['margin']:options.append((gain,e))
    return max(options,key=lambda x:x[0])[1] if options else base

def features(record,e,group,wrong=False):
    b=record['baseline'];cols=GROUPS[group]
    def vector(expert):
        x=record['evidence'][expert]
        return np.asarray([x[(k.replace('ctx','wrong',1) if k.startswith('ctx') else 'wrong_'+k) if wrong and (k.startswith('ctx') or k.startswith(('px','scale','om100'))) else k] for k in cols],float)
    c=vector(e);v=vector(b)
    return np.r_[c,v,c-v,[float(e==x) for x in EXPERTS],[float(b==x) for x in EXPERTS]]

def calibrated_select(record,model,group,threshold,wrong=False,strength=1.):
    base=record['baseline']
    if strength==0:return base
    options=[e for e in EXPERTS if allowed(record,e)]
    if not options:return base
    pred=model.predict(np.asarray([features(record,e,group,wrong) for e in options]));assert np.isfinite(pred).all()
    i=int(np.argmax(pred));return options[i] if pred[i]>threshold else base

from fixed_context import VARIANTS,KEYS
GROUPS={'cycle':BASE+DENSE,'old_ring15':BASE+DENSE+tuple('ctx15_'+k for k in KEYS),'old_ring20':BASE+DENSE+tuple('ctx20_'+k for k in KEYS),**{v:BASE+DENSE+tuple(v+'_'+k for k in KEYS) for v in VARIANTS}}
HAND=[]

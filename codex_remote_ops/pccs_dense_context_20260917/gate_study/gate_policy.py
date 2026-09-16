from policies import EXPERTS,allowed,features
import numpy as np
GATES=('dominance','present','context_override')
def admits(row,e,gate):
    b=row['baseline'];x=row['evidence'][e];base=row['evidence'][b]
    valid=e!=b and row['source_valid'] and x['quality_valid'] and x['valid'] and row['mask_sha256'][e]!=row['mask_sha256'][b]
    if not valid:return False
    if gate=='dominance':return allowed(row,e)
    if x['soft_cycle']<=.001:return False
    if gate=='present':return True
    assert gate=='context_override'
    if allowed(row,e) or not base['valid']:return True
    if x['soft_cycle']<.5*base['soft_cycle'] or x['object_cos']<base['object_cos']-.03:return False
    return all(x['ctx'+s+'_valid'] and base['ctx'+s+'_valid'] and x['ctx'+s+'_contrast']>=base['ctx'+s+'_contrast']+.03 for s in ('15','20'))
def choose(row,predictions,gate,threshold,strength=1.):
    if strength==0:return row['baseline']
    options=[e for e in predictions if admits(row,e,gate)]
    if not options:return row['baseline']
    e=max(options,key=lambda e:predictions[e]);return e if predictions[e]>threshold else row['baseline']
def scores(row,model,group,wrong=False):
    options=[e for e in EXPERTS if e!=row['baseline'] and row['source_valid'] and row['evidence'][e]['quality_valid'] and row['mask_sha256'][e]!=row['mask_sha256'][row['baseline']]]
    if not options:return {}
    values=model.predict(np.asarray([features(row,e,group,wrong) for e in options]));assert np.isfinite(values).all();return dict(zip(options,map(float,values)))

import math,numpy as np
from policies import EXPERTS,allowed,features as original_features
GROUPS=('cycle_control','lift','lift_context')
GATES=('dominance','positive_lift','relative_lift')
def lift(x):
    if not x['valid'] or x['q_area']<=0 or x['s_area']<=0:return np.zeros(5)
    log=lambda a:math.log(max(float(a),1e-30))
    q=log(x['q_area']);s=log(x['s_area'])
    return np.clip([log(x['forward_mass'])-q,log(x['backward_mass'])-s,log(x['cycle_mass'])-q-s,q,s],-12,12)
def features(row,e,group):
    v=original_features(row,e,'full' if group=='lift_context' else 'cycle')
    if group=='cycle_control':return v
    c=lift(row['evidence'][e]);b=lift(row['evidence'][row['baseline']]);return np.r_[v,c,b,c-b]
def admits(row,e,gate):
    b=row['baseline'];x=row['evidence'][e]
    if not (e!=b and row['source_valid'] and x['quality_valid'] and x['valid'] and row['mask_sha256'][e]!=row['mask_sha256'][b]):return False
    if gate=='dominance':return allowed(row,e)
    c=lift(x);base=lift(row['evidence'][b])
    if c[0]<=0 or c[1]<=0:return False
    if gate=='positive_lift':return True
    assert gate=='relative_lift';return c[2]>base[2]+.1
def scores(row,model,group,wrong=False):
    options=[e for e in EXPERTS if e!=row['baseline'] and row['source_valid'] and row['evidence'][e]['quality_valid'] and row['mask_sha256'][e]!=row['mask_sha256'][row['baseline']]]
    if not options:return {}
    x=model.predict(np.asarray([features(row,e,group) for e in options]));assert np.isfinite(x).all();return dict(zip(options,map(float,x)))
def choose(row,pred,gate,threshold,strength=1.):
    if strength==0:return row['baseline']
    options=[e for e in pred if admits(row,e,gate)]
    if not options:return row['baseline']
    e=max(options,key=lambda e:pred[e]);return e if pred[e]>threshold else row['baseline']
def test():
    for q,s in ((.1,.2),(1e-7,.01),(.7,.6)):
        x={'valid':True,'q_area':q,'s_area':s,'forward_mass':q,'backward_mass':s,'cycle_mass':q*s};assert np.max(np.abs(lift(x)[:3]))<1e-12
    assert choose({'baseline':'fusion'},None,None,None,strength=0)=='fusion'
if __name__=='__main__':test();print('CHANCE_LEVEL_AND_ZERO_TESTS_PASS')

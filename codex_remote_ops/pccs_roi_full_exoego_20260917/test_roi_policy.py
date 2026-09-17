import copy
from types import SimpleNamespace
from roi_policy import GROUPS,permitted,predict
from policies import EXPERTS
from native_bridge import native_select
r={'baseline':'fusion','source_valid':True,'mask_sha256':dict(zip(EXPERTS,('v','a','f'))),'evidence':{e:{'quality_valid':True,'valid':True,'soft_cycle':.02} for e in EXPERTS},'metrics':None}
for gate in ('dominance','present'):
    original=[permitted(r,e,gate) for e in EXPERTS];changed=copy.deepcopy(r)
    for e in EXPERTS:changed['evidence'][e].update(local20_valid=True,local20_log_forward=100.,local20_log_backward=100.)
    assert original==[permitted(changed,e,gate) for e in EXPERTS]
assert predict(r,None,'global',.03,strength=0)=='fusion'
assert native_select(SimpleNamespace(native_dense_config={'strength':0}),None,None,0,'fusion')=='fusion'
r['source_valid']=False;assert not any(permitted(r,e,'present') for e in EXPERTS)
print('ROI_COMMON_GATE_AND_ZERO_STRENGTH_PASS')

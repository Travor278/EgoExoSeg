from policies import hand_select,calibrated_select,allowed
class ForbiddenModel:
    def predict(self,x):raise AssertionError('Disabled correction called model')
r={'baseline':'fusion'}
assert hand_select(r,{},strength=0)=='fusion'
assert calibrated_select(r,ForbiddenModel(),'full',.03,strength=0)=='fusion'
r.update(source_valid=True,mask_sha256={'fusion':'f','visual':'v','anchor':'a'},evidence={'fusion':{'soft_cycle':.1},'visual':{'soft_cycle':.09,'quality_valid':True,'valid':True},'anchor':{'soft_cycle':.2,'quality_valid':False,'valid':True}})
assert not allowed(r,'visual') and not allowed(r,'anchor')
assert hand_select(r,{'kind':'soft','margin':.005})=='fusion'
r['evidence']['anchor']['quality_valid']=True
assert hand_select(r,{'kind':'soft','margin':.005})=='anchor'
r['source_valid']=False;assert hand_select(r,{'kind':'soft','margin':.005})=='fusion'
r['source_valid']=True
from policies import BASE
for e in r['evidence']:
    for k in BASE:r['evidence'][e][k]=0.
class ConstantModel:
    def predict(self,x):return [0.2]*len(x)
first=calibrated_select(r,ConstantModel(),'base',.03)
r['metrics']={'not_accessible_ground_truth':'deliberately invalid for scoring'}
assert calibrated_select(r,ConstantModel(),'base',.03)==first=='anchor'
print('POLICY_OFF_AND_POSITIVE_CYCLE_GATES_PASS')

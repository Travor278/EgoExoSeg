import numpy as np
from native_context import ring,evidence_from_matches,recheck_fusion,weighted_count
m=np.zeros((8,8),bool);m[3:5,3:5]=1;r=ring(m);assert not (r&m).any() and r.any()
identity=np.arange(64);valid=np.ones(64,bool)
x=evidence_from_matches(m,m,identity,identity,valid,valid);assert x['support']==1 and x['reliability']>0
none=evidence_from_matches(m,m,identity,identity,np.zeros(64,bool),np.zeros(64,bool));assert none['reliability']==0
wrong=evidence_from_matches(m,m,identity,identity,valid,valid,wrong=True);assert wrong['support']<x['support']
assert not ring(np.zeros((8,8))).any()
class Metric:
    def _count_points_in_mask(self,p,m):return 2
metric=Metric();bad={'support':0.,'reliability':1.}
assert weighted_count(metric,None,None,bad,{'cycle':True,'strength':0})==2
assert weighted_count(metric,None,None,none,{'cycle':True,'strength':.5})==2
assert weighted_count(metric,None,None,bad,{'cycle':True,'strength':.5})==1
assert recheck_fusion({'fusion':bad},{'review':True,'strength':.5})
assert not recheck_fusion({'fusion':none},{'review':True,'strength':.5})
assert not recheck_fusion({'fusion':bad},{'review':True,'strength':0})
print('NATIVE_CONTEXT_UNIT_PASS')

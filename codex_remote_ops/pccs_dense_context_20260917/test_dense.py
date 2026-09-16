import torch,numpy as np
from dense_context import extract
x=torch.eye(4).T.reshape(4,2,2);source=np.zeros((1,2,2),np.uint8);source[0,0,0]=1
good=source.copy();bad=np.zeros_like(source);bad[0,1,1]=1
r={'features_query':x,'features_target':x}
a=extract(r,good,source)[0];b=extract(r,bad,source)[0];z=extract(r,np.zeros_like(source),source)[0]
assert a['soft_cycle']>.99 and b['soft_cycle']<1e-4
assert not z['valid'] and z['soft_cycle']==0 and all(np.isfinite(v) for v in z.values())
assert a['cycle_mass']>.99 and b['cycle_mass']<1e-4
print('DENSE_IDENTITY_WRONG_OBJECT_EMPTY_TESTS_PASS')

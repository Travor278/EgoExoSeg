import numpy as np
from fixed_context import region
m=np.zeros((768,1024),bool);m[300:400,400:500]=True
r=region(m,(48,64),'px100_box')
assert abs(r.sum()*256-300*300)<.1
r50=region(m,(48,64),'px50_box');r150=region(m,(48,64),'px150_box')
assert np.all(r50<=r) and np.all(r<=r150)
# 100px is independent of object size; scale2 is not.
s=region(m,(48,64),'scale20_box');assert abs(s.sum()*256-200*200)<.1
m[:]=False;m[:10,:10]=True
assert abs(region(m,(48,64),'px100_box').sum()*256-110*110)<.1
assert region(m*False,(48,64),'px100_box').sum()==0
# Canonical O-MaMa source and target radii differ after native resizing.
m[:]=False;m[300:400,400:500]=True
assert not np.allclose(region(m,(48,64),'om100_box',True),region(m,(48,64),'om100_box',False))
print('FIXED_CONTEXT_GEOMETRY_PASSED')

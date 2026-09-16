import numpy as np
from PIL import Image
from roi_views import crop_view
im=Image.fromarray(np.full((100,200,3),200,np.uint8));m=np.zeros((100,200),np.uint8);m[2:7,185:195]=1
for scale in (1.5,2.):
    a=crop_view(im,m,scale);b=crop_view(im,m,scale,True);assert a['image'].width==a['image'].height and a['mask'].sum()==50
    assert np.array_equal(np.asarray(a['image'])[a['mask']>0],np.asarray(b['image'])[b['mask']>0]);assert np.any(np.asarray(a['image'])[a['mask']==0]!=np.asarray(b['image'])[b['mask']==0])
assert crop_view(im,np.zeros_like(m)) is None
large=np.zeros_like(m);large[20:22,1:199]=1;assert crop_view(im,large) is None
print('ROI_FOREGROUND_PRESERVATION_AND_CONTEXT_CONTROL_PASS')
